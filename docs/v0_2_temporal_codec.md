# v0.2 temporal codec experiment

The v0.1 mel-CVAE produced perceptually unusable samples even when reconstruction metrics looked reasonable. v0.2 therefore changes the representation before attempting a Transformer.

## Architecture

- shared mono waveform codec for mix, instrumental and vocals
- six stride-2 temporal downsampling stages
- total waveform hop: 64 samples
- at 8 kHz: 125 latent tokens/second
- 32-dimensional latent vectors
- 64-entry discrete codebook
- waveform L1 + STFT reconstruction objective

The first direct end-to-end VQ run collapsed to roughly 1–1.5 perplexity, so v0.2 uses a staged procedure:

1. train the temporal autoencoder continuously;
2. collect continuous training latents;
3. fit a 64-centroid k-means codebook;
4. quantize latents with the fitted codebook;
5. reject quantized fine-tuning if held-out reconstruction gets worse.

## First real run

Dataset: all 13 supplied FNF songs, with the same song-level split used by v0.1.

- 11 train songs
- held out: `forgotten_words_old`, `hi_jon`
- 8 kHz mono
- 1-second diagnostic crops
- 99 energetic training crops sampled from mix/instrumental/vocals
- 8 energetic held-out mix crops
- 255,041 model parameters
- 20 continuous-autoencoder epochs

Continuous validation loss improved from 1.6980 at epoch 1 to **1.4125 at epoch 20**.

After k-means fitting:

- training codebook fitting used all 64 codes
- `hi_jon` 1-second example used 37/64 codes
- `hi_jon` token perplexity: 25.30
- aggregate held-out code usage: 54/64
- aggregate held-out perplexity: 33.75
- mean held-out continuous waveform MAE: 0.1620
- mean held-out quantized waveform MAE: 0.1734
- mean held-out continuous waveform correlation: 0.4345
- mean held-out quantized waveform correlation: 0.2795

For the selected `hi_jon` crop, quantization increased MAE from 0.1439 to 0.1605, which is a much smaller degradation than the v0.1 prior failure.

A five-epoch quantized fine-tune with a frozen k-means codebook was tested. It degraded validation, so that checkpoint was rejected. The current prototype deliberately keeps the k-means-fitted codebook without the rejected fine-tuning.

## Gate before a Transformer

Do not train the token Transformer until the held-out reconstruction is audibly recognizable. Numerical loss alone is not an acceptance criterion anymore.

Once the codec passes that listening test, the next phase is an autoregressive or masked temporal Transformer over code IDs, conditioned by stem type and later by style/prompt tags.
