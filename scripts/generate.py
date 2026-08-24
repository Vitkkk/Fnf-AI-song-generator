#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from fnf_ai.audio import mel_to_waveform, save_audio, sum_stems
from fnf_ai.model import StemCVAE


def parse_args():
    p = argparse.ArgumentParser(description="Generate a short FNF snippet from a trained v0.1 checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="generated/sample")
    p.add_argument("--prompt", default="fnf")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def prompt_to_tags(prompt: str, vocab: list[str]) -> tuple[torch.Tensor, list[str]]:
    words = set(re.findall(r"[a-z0-9_]+", prompt.lower()))
    active = [tag for tag in vocab if tag == "fnf" or tag.lower() in words]
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for tag in active:
        vec[vocab.index(tag)] = 1.0
    return vec, active


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    vocab = ckpt["tag_vocab"]
    model = StemCVAE(
        n_mels=cfg["n_mels"], spec_frames=cfg["spec_frames"], latent_dim=cfg["latent_dim"],
        base_channels=cfg["base_channels"], num_tags=len(vocab),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, vocab, ckpt


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, cfg, vocab, ckpt = load_model(args.checkpoint, device)
    tags, active = prompt_to_tags(args.prompt, vocab)
    print("checkpoint epoch:", ckpt.get("epoch"))
    print("active tags:", ", ".join(active))
    unknown_words = [w for w in re.findall(r"[a-z0-9_]+", args.prompt.lower()) if w not in vocab]
    if unknown_words and len(vocab) == 1:
        print("note: this checkpoint has no semantic style labels yet; prompt words beyond 'fnf' are ignored")

    with torch.no_grad():
        specs = model.sample(tags.unsqueeze(0).to(device), seed=args.seed, temperature=args.temperature)[0].cpu()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    length = int(cfg["sample_rate"] * cfg["segment_seconds"])
    names = ["instrumental", "vocals", "mix_decoder"]
    waves = []
    for i, name in enumerate(names):
        wav = mel_to_waveform(
            specs[i], cfg["sample_rate"], cfg["n_fft"], cfg["hop_length"], cfg["n_mels"], length
        )
        save_audio(out / f"{name}.wav", wav, cfg["sample_rate"])
        waves.append(wav)
    mix_sum = sum_stems(waves[0], waves[1])
    save_audio(out / "mix_stems.wav", mix_sum, cfg["sample_rate"])
    print(f"wrote generated audio to {out}")


if __name__ == "__main__":
    main()
