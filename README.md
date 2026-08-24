# FNF AI Song Generator — v0.1

Experimental **from-scratch** music generator specialized for Friday Night Funkin'-style audio.

The first version deliberately does not try to clone Suno or generate full songs. It answers a smaller question first: can a compact model trained on a small FNF dataset learn a useful latent space and generate new short FNF-like stem pairs?

## v0.1 design

The model is a conditional convolutional VAE trained jointly on three synchronized log-mel spectrograms:

1. instrumental
2. vocals
3. mix

The decoder predicts all three. A consistency term encourages the predicted mix to agree with the energy of the predicted instrumental + vocals. Song-level `training_weight` values from the dataset manifest are preserved.

Current generation length is **4 seconds** at 16 kHz mono. This is intentional for the proof of concept; later versions can move to a learned audio codec/latent Transformer or diffusion model for longer and higher-quality output.

## Dataset handling

Do **not** commit training audio to this repository.

`prepare_dataset.py` merges the split ZIP parts and normalizes every stem to PCM WAV with a common sample rate and timeline. It prefers `mix_exact.wav` when present, uses a normal mix only when its duration is plausibly aligned, otherwise synthesizes a training mix from the stems, and pads shorter stems with silence rather than truncating legitimate outros.

## Setup

Requirements: Python 3.10+, FFmpeg/ffprobe on `PATH`, PyTorch + torchaudio. CUDA is recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Prepare the six dataset ZIPs

```bash
python scripts/prepare_dataset.py \
  /path/FNF_Dataset_Part1.zip \
  /path/FNF_Dataset_Part2.zip \
  /path/FNF_Dataset_Part3.zip \
  /path/FNF_Dataset_Part4.zip \
  /path/FNF_Dataset_Part5.zip \
  /path/FNF_Dataset_Part6.zip \
  --output dataset/v0_1 \
  --sample-rate 16000
```

The result is `dataset.json` plus `songs/<song_id>/{instrumental,vocals,mix}.wav`.

## Optional style labels

The current dataset does not contain semantic labels such as `epic`, `dark`, `sad` or `boss`. The architecture already supports them, but the model should not pretend to understand words it was never labeled with. Pass an optional JSON mapping song IDs to style tags with `--styles` when preparing the dataset. Every song automatically receives the generic `fnf` tag.

## Train

```bash
python scripts/train.py --dataset dataset/v0_1 --config configs/v0_1.json --output checkpoints/v0_1
```

The default run is 50 epochs and saves `last.pt`, `best.pt`, and epoch checkpoints every five epochs. Validation is split **by song**, not by random crop, to reduce leakage.

## Generate

```bash
python scripts/generate.py \
  --checkpoint checkpoints/v0_1/best.pt \
  --prompt "fnf" \
  --seed 1234 \
  --output generated/test
```

Outputs: `instrumental.wav`, `vocals.wav`, `mix_decoder.wav`, and `mix_stems.wav`.

## Compare checkpoints

```bash
python scripts/compare_checkpoints.py \
  --checkpoints checkpoints/v0_1 \
  --seed 1234 \
  --output generated/checkpoint_comparison
```

This uses the same latent seed at every saved epoch so the training progression can be heard directly.

## What counts as success for v0.1?

Not studio quality. The first milestone is reached if later checkpoints produce new audio that is audibly more structured than early checkpoints and if instrumental/vocal generations begin to show FNF-like rhythmic or melodic behavior without simply reproducing one training crop.

## Next milestones

- label the 13 songs with semantic style tags;
- increase the dataset;
- replace Griffin-Lim with a learned decoder/vocoder;
- move from one-shot VAE sampling to a temporal latent Transformer or diffusion model;
- generate longer sections and eventually full songs;
- add explicit controls for BPM, energy, call-and-response and character-vocal behavior.
