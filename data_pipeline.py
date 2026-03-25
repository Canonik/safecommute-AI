import os
import requests
import zipfile
import io
import random
import numpy as np
import librosa
import soundata
import torch
import torchaudio.transforms as T

# constants stuff for preparing data
SAMPLE_RATE = 16000
DURATION_SEC = 3.0  # 3-second context window 
TARGET_LENGTH = int(SAMPLE_RATE * DURATION_SEC)
N_MELS = 64
RAW_DIR = "raw_data"
OUTPUT_DIR = "prepared_data"

# Setup Output Structure
for split in ['train', 'val']:
    for class_id in ['0_safe', '1_unsafe']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_id), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, 'ravdess'), exist_ok=True)

# download and caching logic + check if datasets already exist
def download_ravdess():
    """
    Downloads RAVDESS if not already present.
    """
    target_dir = os.path.join(RAW_DIR, 'ravdess')
    # Check if we already have files
    if len(os.listdir(target_dir)) > 50:
        print("_______________RAVDESS already downloaded. Skipping download._______________")
        return target_dir

    print("_______________Downloading RAVDESS dataset (approx 200MB subset)_______________")
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(target_dir)
    print("_______________RAVDESS download complete_______________")  # audio datasets download
    return target_dir

def initialize_urbansound():
    """
    Uses Soundata to fetch background clutter.
    """
    print("_______________Checking UrbanSound8K (Background clutter)_______________")
    urbansound = soundata.initialize('urbansound8k')
    urbansound.download() # Soundata inherently skips if already downloaded
    return urbansound

# we preprocess and augment data with background noises
def pad_or_truncate(y):
    if len(y) > TARGET_LENGTH:
        start = random.randint(0, len(y) - TARGET_LENGTH)
        return y[start : start + TARGET_LENGTH]
    return np.pad(y, (0, TARGET_LENGTH - len(y)), 'constant')

def mix_audio(signal, noise, snr_db=5):
    """
    Mixes a foreground signal with background noise to simulate station ambience
    """
    signal, noise = pad_or_truncate(signal), pad_or_truncate(noise)
    p_signal, p_noise = np.mean(signal ** 2), np.mean(noise ** 2)
    
    if p_noise == 0: return signal
        
    multiplier = np.sqrt((p_signal / (10 ** (snr_db / 10))) / p_noise)
    mixed = signal + (noise * multiplier)
    
    # Peak normalization
    max_val = np.max(np.abs(mixed))
    return mixed / max_val if max_val > 0 else mixed

def extract_features(y):
    """
    Converts raw audio into non-reconstructible Mel-spectrograms for GDPR compliance
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
    
    # Apply Time/Frequency Masking for model robustness 
    if random.random() < 0.5:
        freq_masking = T.FrequencyMasking(freq_mask_param=10)
        time_masking = T.TimeMasking(time_mask_param=20)
        tensor = freq_masking(tensor)
        tensor = time_masking(tensor)
        
    return tensor

def main():
    print(f"{"_"*15}Data Acquisition{"_"*15}")
    ravdess_dir = download_ravdess()
    urbansound = initialize_urbansound()

    print(f"\n{"_"*15}Extracting Background Clutter{"_"*15}")
    background_audios = []
    # Extract safe background noises (street music, engine idling) 
    for clip_id in urbansound.clip_ids:
        clip = urbansound.clip(clip_id)
        if clip.tags.labels[0] in ['street_music', 'engine_idling', 'children_playing']:
            y, sr = clip.audio
            if y is not None:
                if sr != SAMPLE_RATE: y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
                if len(y.shape) > 1: y = librosa.to_mono(y)
                background_audios.append(y)
                
                # Save some clean background noise to the Safe class (loud but safe) 
                split = 'train' if random.random() < 0.8 else 'val'
                features = extract_features(pad_or_truncate(y))
                torch.save(features, os.path.join(OUTPUT_DIR, split, '0_safe', f"bg_{clip_id}.pt"))

    if not background_audios:
        print(f"{"_"*15}\n\n\n Warning: No background audios extracted. Mixing will be skipped.\n\n\n {"_"*15}")

    print(f"\n{"_"*15}Processing Emotional Escalation Cues{"_"*15}")
    for root, _, files in os.walk(ravdess_dir):
        for file in files:
            if file.endswith('.wav'):
                # Decode RAVDESS Filename (e.g., 03-01-05-01-01-01-01.wav)
                parts = file.replace('.wav', '').split('-')
                if len(parts) < 3: continue
                
                emotion = parts[2]
                # 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
                if emotion in ['05', '06']:
                    is_unsafe = True  # Angry/Fear = Unsafe 
                elif emotion in ['01', '02', '03']:
                    is_unsafe = False # Neutral/Calm/Happy = Safe 
                else:
                    continue # Skip sad/disgust to keep binary classes sharp (also useless for our model)

                filepath = os.path.join(root, file)
                y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                
                # Mix with transit noise (SNR between 0 and 15 dB)
                bg_noise = random.choice(background_audios) if background_audios else np.zeros(TARGET_LENGTH)
                mixed_audio = mix_audio(y, bg_noise, snr_db=random.uniform(0.0, 15.0))
                
                features = extract_features(mixed_audio)
                
                class_dir = '1_unsafe' if is_unsafe else '0_safe'
                split = 'train' if random.random() < 0.8 else 'val'
                torch.save(features, os.path.join(OUTPUT_DIR, split, class_dir, f"ravdess_{file}.pt"))

    print("\n{"_"*25}Data is ready for immediate fine-tuning.{"_"*25}")
    print(f"\nCheck the '{OUTPUT_DIR}' directory for your .pt tensors.")

if __name__ == "__main__":
    main()