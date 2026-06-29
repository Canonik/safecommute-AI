# Privacy attack evaluation -- SafeCommute PCEN tiles

Corpus: `librispeech_devclean_3s`, n=200 clips, 16000 Hz.

Threat model: the attacker observes the PCEN tile the classifier consumes plus the public feature-extraction config (`librosa.pcen` defaults, prescale `2**31`). They try to recover intelligible audio or speaker identity.

**Recoveries**

- `pcen_oracle`: exact closed-form inverse given the running-mean state M from the forward pass. Strongest attacker upper bound.

- `pcen_blind`: iterative inverse, M estimated from the PCEN output itself. Realistic attacker.

- `mel_baseline`: feed the raw mel directly to the attack, skipping PCEN. Ablation: quantifies how much privacy PCEN adds over plain mel.

**Attacks**

- `griffin_lim`: power-mel -> waveform via 60 iterations of Griffin-Lim phase estimation. No pretrained model.

- `hifigan`: 64-mel -> 80-mel (closed-form pinv adapter) -> log-mel -> SpeechBrain `tts-hifigan-libritts-16kHz` vocoder. Off-the-shelf, no fine-tuning.

## Metrics per configuration

All values are mean with bootstrap 95 % CI in brackets.

| Recovery | Attack | WER | Speaker cosine (recon vs orig) | Chance cosine | PESQ-wb | STOI |
|---|---|---|---|---|---|---|
| `pcen_oracle` | `griffin_lim` | 0.101  [0.081, 0.122]  (n=200) | 0.841  [0.835, 0.847]  (n=200) | 0.114  [0.095, 0.135]  (n=200) | 1.839  [1.792, 1.886]  (n=200) | 0.899  [0.895, 0.902]  (n=200) |
| `pcen_oracle` | `hifigan` | 0.267  [0.229, 0.305]  (n=200) | 0.550  [0.539, 0.562]  (n=200) | 0.114  [0.095, 0.135]  (n=200) | 1.581  [1.551, 1.610]  (n=200) | 0.853  [0.848, 0.857]  (n=200) |
| `pcen_blind` | `griffin_lim` | 0.150  [0.123, 0.177]  (n=200) | 0.797  [0.789, 0.804]  (n=200) | 0.114  [0.095, 0.135]  (n=200) | 1.742  [1.708, 1.777]  (n=200) | 0.883  [0.880, 0.886]  (n=200) |
| `pcen_blind` | `hifigan` | 0.206  [0.175, 0.238]  (n=200) | 0.631  [0.621, 0.642]  (n=200) | 0.114  [0.095, 0.135]  (n=200) | 1.288  [1.273, 1.303]  (n=200) | 0.844  [0.840, 0.848]  (n=200) |
| `mel_baseline` | `griffin_lim` | 0.091  [0.072, 0.112]  (n=200) | 0.838  [0.832, 0.845]  (n=200) | 0.114  [0.095, 0.135]  (n=200) | 1.847  [1.799, 1.897]  (n=200) | 0.899  [0.896, 0.903]  (n=200) |
| `mel_baseline` | `hifigan` | 0.267  [0.229, 0.305]  (n=200) | 0.550  [0.539, 0.562]  (n=200) | 0.114  [0.095, 0.135]  (n=200) | 1.581  [1.551, 1.610]  (n=200) | 0.853  [0.848, 0.857]  (n=200) |

## Verdict

### Bottom line

Under the **Griffin-Lim** attack, WER from plain mel is 0.09 (95 % CI [0.07, 0.11]); WER from PCEN tiles (oracle attacker) is 0.10 (95 % CI [0.08, 0.12]). PCEN minus mel: +0.010 WER points.

Under the **HiFi-GAN** attack, WER from plain mel is 0.27; WER from PCEN (oracle) is 0.27. PCEN minus mel: +0.000 WER points.

The **blind** PCEN attacker (no oracle running-mean) achieves WER 0.15 (95 % CI [0.12, 0.18]) under Griffin-Lim.


**Interpretation:**

- PCEN tiles yield partially intelligible speech under the strongest attacker (WER lower bound 0.08). PCEN should not be treated as privacy-preserving or non-invertible.

- Speaker identity leaks: reconstructed cosine (0.84) exceeds chance (0.11) by +0.73.

## Transcript spot-checks

The public snapshot retains the raw JSON metrics and transcript examples, not reconstructed WAV files. Local evaluation runs may write sample WAVs under `tests/privacy/reports/samples/`; those files inherit upstream data/model terms and are ignored by git.

| Configuration | Clip ID | Reference (LibriSpeech) | Whisper-tiny hypothesis |
|---|---|---|---|
| `pcen_oracle/griffin_lim` | `3081-166546-0046` | been over the ground, studied the affair | Then over the ground, studied to the affair |
| `pcen_oracle/griffin_lim` | `1272-141231-0026` | Breathing deeply, Breonna's softly | Breathing deeply, beyond softness. |
| `pcen_oracle/griffin_lim` | `174-168635-0009` | To meet was to find each other. | To meet was to find each other. |
| `pcen_oracle/griffin_lim` | `6313-66129-0034` | If the moment when he freed his left foot from the start, he threw his | If the moment when he freed his left foot from the star, he threw his |
| `pcen_oracle/griffin_lim` | `3576-138058-0013` | Sancho replied that all the trees were full of | Sontoreplied that all the trees were full of |
| `pcen_oracle/griffin_lim` | `7850-281318-0013` | She began to show them how to weave the bits of things. | She began to show them how to weave the bits of things. |
| `pcen_oracle/griffin_lim` | `8297-275154-0012` | Randall, he said. | Randall, he said. |
| `pcen_oracle/griffin_lim` | `1988-148538-0013` | They stand unmoved in their solitary greatness. | They stand unmoved in their solitary greatness. |
| `pcen_oracle/griffin_lim` | `5895-34629-0001` | The dome of St. Paul's was a delight to Ursus. | The dome of St. Paul's was a delight to Earth's |
| `pcen_oracle/griffin_lim` | `5694-64025-0011` | I'd be feeding me in all the mornings if I had stolen | I'd be feeling mean all the mornings if I had stolen |
| `pcen_oracle/hifigan` | `3081-166546-0046` | been over the ground, studied the affair | Then over the ground, study to the affair. |
| `pcen_oracle/hifigan` | `1272-141231-0026` | Breathing deeply, Breonna's softly | Breathing deeply, bring out a soft case. |
| `pcen_oracle/hifigan` | `174-168635-0009` | To meet was to find each other. | To meet was to find each other. |
| `pcen_oracle/hifigan` | `6313-66129-0034` | If the moment when he freed his left foot from the start, he threw his | If the moment when he freight his left foot from a stirup he threw his |
| `pcen_oracle/hifigan` | `3576-138058-0013` | Sancho replied that all the trees were full of | Something replied that all the trees were full of |
| `pcen_oracle/hifigan` | `7850-281318-0013` | She began to show them how to weave the bits of things. | She began to show them how to weave the bits of things. |
| `pcen_oracle/hifigan` | `8297-275154-0012` | Randall, he said. | Randall, he said. |
| `pcen_oracle/hifigan` | `1988-148538-0013` | They stand unmoved in their solitary greatness. | This time, unmoved and there are solitary greatness. |
| `pcen_oracle/hifigan` | `5895-34629-0001` | The dome of St. Paul's was a delight to Ursus. | The dome of St. Paul's was a delight to Ursus. |
| `pcen_oracle/hifigan` | `5694-64025-0011` | I'd be feeding me in all the mornings if I had stolen | By defeating me in all the mornings if I had stolen |
| `pcen_blind/griffin_lim` | `3081-166546-0046` | been over the ground, studied the affair | Then over the ground, studied the affair. |
| `pcen_blind/griffin_lim` | `1272-141231-0026` | Breathing deeply, Breonna's softly | Reusing deeply, re-unsoftless. |
| `pcen_blind/griffin_lim` | `174-168635-0009` | To meet was to find each other. | To meet was to find each other. |
| `pcen_blind/griffin_lim` | `6313-66129-0034` | If the moment when he freed his left foot from the start, he threw his | If the moment when he freed his left foot from the start, he threw his |
| `pcen_blind/griffin_lim` | `3576-138058-0013` | Sancho replied that all the trees were full of | Sondho replied that all the trees were full of |
| `pcen_blind/griffin_lim` | `7850-281318-0013` | She began to show them how to weave the bits of things. | She began to show them how to weave the bits of things. |
| `pcen_blind/griffin_lim` | `8297-275154-0012` | Randall, he said. | Randall, he said. |
| `pcen_blind/griffin_lim` | `1988-148538-0013` | They stand unmoved in their solitary greatness. | They stand unmoved in their solitary greatness. |
| `pcen_blind/griffin_lim` | `5895-34629-0001` | The dome of St. Paul's was a delight to Ursus. | The Dome of St. Paul's was a delight to Earth's. |
| `pcen_blind/griffin_lim` | `5694-64025-0011` | I'd be feeding me in all the mornings if I had stolen | I've been feeding me in all the mornings if I had stolen |
| `pcen_blind/hifigan` | `3081-166546-0046` | been over the ground, studied the affair | Then over the ground, studied to the affair. |
| `pcen_blind/hifigan` | `1272-141231-0026` | Breathing deeply, Breonna's softly | Breathing deeply, free out of the soft place. |
| `pcen_blind/hifigan` | `174-168635-0009` | To meet was to find each other. | To meet was to find each other. |
| `pcen_blind/hifigan` | `6313-66129-0034` | If the moment when he freed his left foot from the start, he threw his | If the moment when he freed his left foot from a stirrup, he threw his |
| `pcen_blind/hifigan` | `3576-138058-0013` | Sancho replied that all the trees were full of | Saunter replied that all the trees were full of |
| `pcen_blind/hifigan` | `7850-281318-0013` | She began to show them how to weave the bits of things. | She began to show them how to weave the bits of things. |
| `pcen_blind/hifigan` | `8297-275154-0012` | Randall, he said. | Randall, he said. |
| `pcen_blind/hifigan` | `1988-148538-0013` | They stand unmoved in their solitary greatness. | They stand unmoved in their solitary greatness. |
| `pcen_blind/hifigan` | `5895-34629-0001` | The dome of St. Paul's was a delight to Ursus. | The dome of St. Paul's was a delight to Earth's. |
| `pcen_blind/hifigan` | `5694-64025-0011` | I'd be feeding me in all the mornings if I had stolen | I've been fitting me in all the mornings if I had stolen |
| `mel_baseline/griffin_lim` | `3081-166546-0046` | been over the ground, studied the affair | Then over the ground, studied to the afheric |
| `mel_baseline/griffin_lim` | `1272-141231-0026` | Breathing deeply, Breonna's softly | Breathing deeply, beyond softness. |
| `mel_baseline/griffin_lim` | `174-168635-0009` | To meet was to find each other. | To meet was to find each other. |
| `mel_baseline/griffin_lim` | `6313-66129-0034` | If the moment when he freed his left foot from the start, he threw his | If the moment when he freed his left foot from the star, he threw his |
| `mel_baseline/griffin_lim` | `3576-138058-0013` | Sancho replied that all the trees were full of | Sontoreplied that all the trees were full of |
| `mel_baseline/griffin_lim` | `7850-281318-0013` | She began to show them how to weave the bits of things. | She began to show them how to weave the bits of things |
| `mel_baseline/griffin_lim` | `8297-275154-0012` | Randall, he said. | Randall, he said. |
| `mel_baseline/griffin_lim` | `1988-148538-0013` | They stand unmoved in their solitary greatness. | They stand unmoved in their solitary greatness. |
| `mel_baseline/griffin_lim` | `5895-34629-0001` | The dome of St. Paul's was a delight to Ursus. | The dome of St. Paul's was a delight to Earth's |
| `mel_baseline/griffin_lim` | `5694-64025-0011` | I'd be feeding me in all the mornings if I had stolen | I've been feeling mean all the mornings if I had stolen |
| `mel_baseline/hifigan` | `3081-166546-0046` | been over the ground, studied the affair | Then over the ground, study to the affair. |
| `mel_baseline/hifigan` | `1272-141231-0026` | Breathing deeply, Breonna's softly | Breathing deeply, bring out a soft case. |
| `mel_baseline/hifigan` | `174-168635-0009` | To meet was to find each other. | To meet was to find each other. |
| `mel_baseline/hifigan` | `6313-66129-0034` | If the moment when he freed his left foot from the start, he threw his | If the moment when he freight his left foot from a stirup he threw his |
| `mel_baseline/hifigan` | `3576-138058-0013` | Sancho replied that all the trees were full of | Something replied that all the trees were full of |
| `mel_baseline/hifigan` | `7850-281318-0013` | She began to show them how to weave the bits of things. | She began to show them how to weave the bits of things. |
| `mel_baseline/hifigan` | `8297-275154-0012` | Randall, he said. | Randall, he said. |
| `mel_baseline/hifigan` | `1988-148538-0013` | They stand unmoved in their solitary greatness. | This time, unmoved and there are solitary greatness. |
| `mel_baseline/hifigan` | `5895-34629-0001` | The dome of St. Paul's was a delight to Ursus. | The dome of St. Paul's was a delight to Ursus. |
| `mel_baseline/hifigan` | `5694-64025-0011` | I'd be feeding me in all the mornings if I had stolen | By defeating me in all the mornings if I had stolen |


## Hidden-phrase sub-evaluation

A separate corpus of 5 synthesised probe clips (speechbrain Tacotron2 + HiFi-GAN on LJSpeech, resampled to 16000 Hz) was generated with planted phrases such as `"the password is fortepiano"`. We push each clip through the same six (recovery x attack) pipeline and report per-clip Whisper transcripts so a reader can spot-check whether specific keywords survive.

| Configuration | Planted phrase | Whisper hypothesis |
|---|---|---|
| `pcen_oracle/griffin_lim` | The password is 4 to PM. | The password is 4 to P.N. |
| `pcen_oracle/griffin_lim` | credit card number is 1-2-3-4-5-6. | credit card number is 1-2-3-4-5-6. |
| `pcen_oracle/griffin_lim` | safe combination is 47 16. | This safe combination is 4716. |
| `pcen_oracle/griffin_lim` | France for 2000 Euros to account number 19. | France for 2000 years to account number 19. |
| `pcen_oracle/griffin_lim` | The axis code for the vault is AlphaTango Breve. | The axis code for the vault is AlphaTango Brewe. |
| `pcen_oracle/hifigan` | The password is 4 to PM. | The Peniswork is 4 to PM. |
| `pcen_oracle/hifigan` | credit card number is 1-2-3-4-5-6. | credit card number is 1-2-3-4-5-6. |
| `pcen_oracle/hifigan` | safe combination is 47 16. | safe combination is 4716. |
| `pcen_oracle/hifigan` | France for 2000 Euros to account number 19. | for 2,000 viewers to account number 90. |
| `pcen_oracle/hifigan` | The axis code for the vault is AlphaTango Breve. | the axis code for the vault is alpha tangle rather. |
| `pcen_blind/griffin_lim` | The password is 4 to PM. | The password is 4 to PM. |
| `pcen_blind/griffin_lim` | credit card number is 1-2-3-4-5-6. | credit card number is 1-2-3-4-5-6. |
| `pcen_blind/griffin_lim` | safe combination is 47 16. | safe combination is 47 16. |
| `pcen_blind/griffin_lim` | France for 2000 Euros to account number 19. | for 2,000 years to account number 19. |
| `pcen_blind/griffin_lim` | The axis code for the vault is AlphaTango Breve. | The axis code for the vault is AlphaTango Brev. |
| `pcen_blind/hifigan` | The password is 4 to PM. | The password is for to PM. |
| `pcen_blind/hifigan` | credit card number is 1-2-3-4-5-6. | credit card number is 1-2-3-4-5-6. |
| `pcen_blind/hifigan` | safe combination is 47 16. | safe combination is 4716. |
| `pcen_blind/hifigan` | France for 2000 Euros to account number 19. | for 2,000 years to account number 19. |
| `pcen_blind/hifigan` | The axis code for the vault is AlphaTango Breve. | the access code for the vault is AlphaTangle Grub. |
| `mel_baseline/griffin_lim` | The password is 4 to PM. | The password is 4 to PM. |
| `mel_baseline/griffin_lim` | credit card number is 1-2-3-4-5-6. | credit card number is 1-2-3-4-5-6. |
| `mel_baseline/griffin_lim` | safe combination is 47 16. | safe combination is 4716. |
| `mel_baseline/griffin_lim` | France for 2000 Euros to account number 19. | for 2,000 years to account number 19. |
| `mel_baseline/griffin_lim` | The axis code for the vault is AlphaTango Breve. | The axis code for the vault is AlphaTango Brev. |
| `mel_baseline/hifigan` | The password is 4 to PM. | The Peniswork is 4 to PM. |
| `mel_baseline/hifigan` | credit card number is 1-2-3-4-5-6. | credit card number is 1-2-3-4-5-6. |
| `mel_baseline/hifigan` | safe combination is 47 16. | safe combination is 4716. |
| `mel_baseline/hifigan` | France for 2000 Euros to account number 19. | for 2,000 viewers to account number 90. |
| `mel_baseline/hifigan` | The axis code for the vault is AlphaTango Breve. | the axis code for the vault is alpha tangle rather. |

Aggregate metrics for the probe-phrase corpus (n is small, so CIs are wide):

| Recovery | Attack | WER | Speaker cosine | PESQ-wb | STOI |
|---|---|---|---|---|---|
| `pcen_oracle` | `griffin_lim` | 0.201  [0.058, 0.410]  (n=5) | 0.836  [0.816, 0.861]  (n=5) | 1.722  [1.627, 1.851]  (n=5) | 0.908  [0.904, 0.912]  (n=5) |
| `pcen_oracle` | `hifigan` | 0.302  [0.122, 0.458]  (n=5) | 0.581  [0.567, 0.595]  (n=5) | 1.631  [1.547, 1.702]  (n=5) | 0.892  [0.884, 0.900]  (n=5) |
| `pcen_blind` | `griffin_lim` | 0.097  [0.000, 0.247]  (n=5) | 0.803  [0.772, 0.834]  (n=5) | 1.802  [1.564, 2.046]  (n=5) | 0.899  [0.880, 0.911]  (n=5) |
| `pcen_blind` | `hifigan` | 0.277  [0.113, 0.413]  (n=5) | 0.567  [0.534, 0.611]  (n=5) | 1.228  [1.169, 1.285]  (n=5) | 0.851  [0.826, 0.876]  (n=5) |
| `mel_baseline` | `griffin_lim` | 0.177  [0.022, 0.332]  (n=5) | 0.830  [0.800, 0.861]  (n=5) | 1.726  [1.628, 1.853]  (n=5) | 0.907  [0.899, 0.915]  (n=5) |
| `mel_baseline` | `hifigan` | 0.302  [0.122, 0.458]  (n=5) | 0.581  [0.567, 0.595]  (n=5) | 1.631  [1.547, 1.702]  (n=5) | 0.892  [0.884, 0.900]  (n=5) |

## How to reproduce

```bash
PYTHONPATH=. python tests/privacy/data/download_librispeech.py
PYTHONPATH=. python tests/privacy/run_attack_eval.py \
    --corpus tests/privacy/data/librispeech_devclean_3s \
    --out-dir tests/privacy/reports
# Hidden-phrase sub-evaluation:
PYTHONPATH=. python tests/privacy/data/synthesize_hidden_phrases.py
PYTHONPATH=. python tests/privacy/run_attack_eval.py \
    --corpus tests/privacy/data/hidden_phrases \
    --out-dir tests/privacy/reports/hidden_phrases
PYTHONPATH=. python tests/privacy/build_report.py
```
