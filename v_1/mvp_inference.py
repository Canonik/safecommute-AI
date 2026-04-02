import pyaudio
import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import time
import os

# system and audio configuration,  VERY DEVICE DEPENDENT, FOR TROUBLESHOOTING CHATGPT IT 
SAMPLE_RATE = 16000
CONTEXT_WINDOW_SEC = 3  # 3 seconds of context 
STRIDE_SEC = 1          # Predict every 1 second 
CHUNK_SIZE = int(SAMPLE_RATE * STRIDE_SEC)
BUFFER_SIZE = int(SAMPLE_RATE * CONTEXT_WINDOW_SEC)
N_MELS = 64
MODEL_PATH = "safecommute_edge_model.pth"

# model architecture (make sure it matches training)
class SafeCommuteCNN(nn.Module):
    def __init__(self):
        super(SafeCommuteCNN, self).__init__()
        self.backbone = mobilenet_v2(weights=None) # No pre-trained weights needed here, we load our own
        
        # 1-channel Mel-spectrogram input
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        # Binary classifier (Safe vs Unsafe)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)

# inference engine
def preprocess_live_audio(audio_buffer):
    """Converts the rolling audio array into a PyTorch tensor."""
    # Convert raw audio to Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_buffer, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Shape for PyTorch: (Batch=1, Channel=1, Mels=64, TimeSteps)
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def main():
    print(f"{"_"*15}Loading SafeCommute AI Edge Model{"_"*15}")
    
    # Edge inference usually runs best on CPU to avoid latency spikes in moving data
    device = torch.device("cpu") 
    model = SafeCommuteCNN()
    
    # Load your freshly trained weights
    if not os.path.exists(MODEL_PATH):
        print(f"{"_"*15} Error: Could not find {MODEL_PATH}. Did you run the training script?{"_"*15}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval() # Set to evaluation mode
    print(f"{"_"*15}Model loaded successfully.{"_"*15}")

    # Initialize PyAudio for live microphone streaming
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=8,  # double check this for your device, sometimes is 4 sometimes 6 or 7
                        frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"Microphone Error: {e}")
        print("Ensure your microphone is connected.")
        p.terminate()
        return

    print("\n" + "="*50)
    print("🎙️  SAFECOMMUTE AI LIVE INFERENCE ACTIVE")
    print("🔒 GDPR-Mode: Processing in RAM. No audio saved.")
    print("="*50 + "\n")
    
    # The 3-second rolling buffer 
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

    try:
        while True:
            # 1. Read 1 second of live audio (the stride)
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            new_audio = np.frombuffer(data, dtype=np.float32)
            
            # 2. Slide the buffer: discard oldest 1 second, append newest 1 second
            audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = new_audio
            
            # 3. Extract features 
            features = preprocess_live_audio(audio_buffer).to(device)
            
            # 4. Run Inference 
            with torch.no_grad():
                outputs = model(features)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Class 1 is 'Unsafe'
                unsafe_prob = probabilities[0][1].item()
            
            # 5. Visual MVP Interface [cite: 184]
            # Emit a compact alert signal 
            timestamp = time.strftime('%H:%M:%S')
            
            # Thresholds for operational safeguards 
            if unsafe_prob < 0.40:
                print(f"[{timestamp}] 🟢 GREEN (Safe)       | Risk: {unsafe_prob:.2f}")
            elif unsafe_prob < 0.75:
                print(f"[{timestamp}] 🟠 AMBER (Warning)    | Risk: {unsafe_prob:.2f} (Monitoring)")
            else:
                print(f"[{timestamp}] 🔴 RED   (ALERT)      | Risk: {unsafe_prob:.2f} ⚠️ ESCALATION DETECTED!")

    except KeyboardInterrupt:
        print("\n🛑 Stopping SafeCommute AI...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("✅ Gracefully shut down. No data was stored.")

if __name__ == "__main__":
    main()