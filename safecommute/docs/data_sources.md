# Data Sources — SafeCommute AI v2

## AudioSet (Strongly-labeled subset)

- **Citation**: Gemmeke, J.F., Ellis, D.P.W., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M. and Ritter, M., 2017. Audio Set: An ontology and human-labeled dataset for audio events. *Proc. IEEE ICASSP*, pp.776-780.
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0)
- **URL**: https://research.google.com/audioset/
- **Samples used**: 
  - Threat (unsafe): Screaming (204), Shout (250), Yell (211), Gunshot (213), Explosion (238), Glass breaking (241) = 1,357 raw clips, ~7,940 chunks after 3s windowing
  - Safe (hard negatives): Laughter (217), Crowd (246), Speech (236), Music (229), Applause (265), Cheering (262), Singing (229) = 1,684 raw clips, ~10,982 chunks
- **Label mapping**: Threat categories -> unsafe (label=1); Safe categories -> safe (label=0)
- **Filtering**: Only strongly-labeled segments used. Downloaded via yt-dlp from YouTube. ~40% failure rate due to deleted/private videos.
- **Excluded categories**: Siren (contextual confound), Crying/sobbing (distress != threat), Fighting (topic label, <20 clips in strongly-labeled)

## ESC-50 (Environmental Sound Classification)

- **Citation**: Piczak, K.J., 2015. ESC: Dataset for Environmental Sound Classification. *Proc. ACM Multimedia*, pp.1015-1018.
- **License**: Creative Commons Attribution Non-Commercial 3.0 (CC BY-NC 3.0)
- **URL**: https://github.com/karolpiczak/ESC-50
- **Samples used**: 800 clips (from 2000 total)
- **Label mapping**: All used categories -> safe (label=0). Categories: rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, toilet_flush, clock_alarm, train, helicopter, church_bells, airplane, washing_machine (ambient) + siren, fireworks, chainsaw, thunderstorm, hand_saw (hard negatives)
- **Split**: Predefined 5 folds (1-3 train, 4 val, 5 test)

## UrbanSound8K

- **Citation**: Salamon, J., Jacoby, C. and Bello, J.P., 2014. A Dataset and Taxonomy for Urban Sound Research. *Proc. ACM Multimedia*, pp.1041-1044.
- **License**: Creative Commons Attribution Non-Commercial 4.0 (CC BY-NC 4.0)
- **URL**: https://urbansounddataset.weebly.com/urbansound8k.html
- **Samples used**: ~6,429 clips
- **Label mapping**: All used categories -> safe (label=0). Safe background: street_music, engine_idling, children_playing. Hard negatives: jackhammer, drilling, air_conditioner, car_horn.
- **Split**: Predefined 10 folds (1-7 train, 8 val, 9-10 test)

## YouTube Real Screams

- **Citation**: Curated by project team (no published dataset)
- **License**: Fair use (research purposes)
- **URL**: Downloaded via yt-dlp from public YouTube videos
- **Samples used**: 57 source clips -> 1,223 chunks (3s windows, 1.5s hop)
- **Label mapping**: All -> unsafe (label=1)
- **Filtering**: Validated with automated quality checks (validate_youtube_data.py). Quarantined files with music, news content, or insufficient energy.

## YouTube Metro Ambient

- **Citation**: Curated by project team (no published dataset)
- **License**: Fair use (research purposes)
- **URL**: Downloaded via yt-dlp from public YouTube metro ride compilations
- **Samples used**: 58 source clips -> 3,489 chunks (3s windows, 1.5s hop)
- **Label mapping**: All -> safe (label=0)
- **Filtering**: Same quality validation as YouTube screams. Quarantined files with music or excessive speech.

## Violence Detection Dataset

- **Citation**: Available on HuggingFace (Hemg/audio-based-violence-dataset)
- **License**: Research use
- **URL**: https://huggingface.co/datasets/Hemg/audio-based-violence-dataset
- **Samples used**: 2,000 source clips -> 2,012 chunks. Label 0 (non-violent) -> safe, Label 1 (violent) -> unsafe.
- **Label mapping**: violence_IDX_0.wav -> safe; violence_IDX_1.wav -> unsafe
- **Split**: sha256 hash of source filename (deterministic)

## Dropped Datasets (v1 only, not in v2)

The following acted speech emotion datasets were used in v1 but dropped in v2 due to poor generalization (acted emotions != real threat audio):

- **CREMA-D** (Cao et al., 2014): 7,442 clips, 91 actors. Test accuracy 45.2%.
- **RAVDESS** (Livingstone & Russo, 2018): 1,440 clips, 24 actors. Test accuracy 52.1%.
- **TESS** (Dupuis & Pichora-Fuller, 2010): 2,800 clips, 2 speakers. Test accuracy 98.8% (speaker memorization).
- **SAVEE** (Haq & Jackson, 2010): 480 clips, 4 speakers. Test accuracy 35.7%.
