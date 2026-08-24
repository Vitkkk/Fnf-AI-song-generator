# v0.1 architecture

## Goal

Test whether a compact model trained from scratch on a small FNF dataset can learn a generative latent space for synchronized instrumental and vocal behavior.

## Representation

Training audio is converted to 16 kHz mono PCM and sliced into random 4-second crops. Each crop becomes three normalized log-mel spectrograms: instrumental, vocals and mix.

## Model

`StemCVAE` uses a shared convolutional encoder over the three spectrogram channels. The encoder produces `mu` and `logvar` for a latent Gaussian. A tag vector is embedded and concatenated into both encoder and decoder paths, allowing later semantic conditioning without changing the core model.

The decoder reconstructs/generates three channels simultaneously:

- instrumental
- vocals
- mix

## Loss

The objective combines:

- L1 spectrogram reconstruction
- beta-weighted KL divergence
- stem consistency: the decoder's mix should approximately agree with the combined mel-domain energy of its generated instrumental and vocals

Dataset `training_weight` values scale each sample's objective.

## Generation

A fixed random seed samples a latent vector. The decoder produces the three mel spectrograms, which are converted back to waveform with inverse mel scaling + Griffin-Lim. `mix_stems.wav` is also exported by directly summing the generated instrumental and vocal waveforms.

## Why this is only v0.1

A VAE plus Griffin-Lim is deliberately small and easy to train, but it is not the final quality target. If the checkpoint comparison shows that the model is learning meaningful FNF structure, the next version should separate representation learning from sequence generation: a learned neural audio codec/vocoder for fidelity and a temporal latent Transformer or diffusion model for long-range composition.
