# First real training run — v0.1 tiny CPU

This run used all 13 supplied FNF songs as the first proof that the stem-aware model can learn a non-trivial distribution from the dataset.

## Setup

- 13 songs total
- song-level split: 11 train / 2 validation
- validation songs: `forgotten_words_old`, `hi_jon`
- generated/synthesized stem mix used for strict instrumental + vocals pairing in this tiny experiment
- 8 kHz mono
- 2-second crops
- 64 mel bins / 128 model frames
- 286,019 trainable parameters in the local tiny CVAE
- 50 epochs
- seed 1337 for the split/training setup
- same latent seed (`20260824`) used to compare generated milestone samples

## Result

Validation loss fell sharply at the start and continued improving more gradually:

| Epoch | Validation loss |
| ---: | ---: |
| 1 | 0.385493 |
| 5 | 0.216711 |
| 10 | 0.192120 |
| 20 | 0.178374 |
| 30 | 0.169335 |
| 35 | **0.163686** |
| 40 | 0.164826 |
| 50 | 0.165385 |

Best validation checkpoint: **epoch 35**.

The validation loss improved by about 57.5% from epoch 1 to the best checkpoint while validation songs were excluded from training. After epoch 35, training loss kept falling but validation stopped improving consistently, which is the first indication of overfitting in this small-data / tiny-model regime.

## Rendering bug discovered by the run

Training completed, but the original `torchaudio.transforms.InverseMelScale` renderer failed on generated spectrograms because the mel filterbank can be rank-deficient. A second issue was that padded model frames did not necessarily equal the exact centered-STFT frame count implied by the requested waveform length.

The renderer was changed to use a Moore-Penrose pseudo-inverse and to crop/pad to the exact Griffin-Lim frame count before waveform reconstruction. A regression test now covers this path.

## What this proves (and what it does not)

This run proves that the v0.1 pipeline learns measurable structure that transfers to held-out songs. It does **not** prove high-quality music generation yet: 8 kHz + 2 seconds + mel/Griffin-Lim is intentionally a cheap diagnostic configuration.

The next useful quality step is to keep the stem-aware objective but move away from Griffin-Lim toward a learned audio codec/vocoder and then add a temporal generative model capable of longer musical structure.
