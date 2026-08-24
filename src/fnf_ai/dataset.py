from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import load_audio, make_mel_transform, waveform_to_scaled_mel


@dataclass
class SongEntry:
    song_id: str
    instrumental: Path
    vocals: Path
    mix: Path
    weight: float
    tags: list[str]


def load_dataset_manifest(root: str | Path) -> dict:
    with open(Path(root) / "dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_tag_vocab(manifest: dict) -> list[str]:
    tags = {"fnf"}
    for song in manifest["songs"]:
        tags.update(song.get("style_tags", []))
    return sorted(tags)


class FNFStemDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        song_ids: list[str],
        config: dict,
        tag_vocab: list[str],
        training: bool,
    ):
        self.root = Path(root)
        self.config = config
        self.training = training
        self.tag_vocab = tag_vocab
        manifest = load_dataset_manifest(root)
        by_id = {s["id"]: s for s in manifest["songs"]}
        self.songs: list[SongEntry] = []
        for song_id in song_ids:
            s = by_id[song_id]
            base = self.root / "songs" / song_id
            self.songs.append(
                SongEntry(
                    song_id=song_id,
                    instrumental=base / "instrumental.wav",
                    vocals=base / "vocals.wav",
                    mix=base / "mix.wav",
                    weight=float(s.get("training_weight", 1.0)),
                    tags=sorted(set(["fnf", *s.get("style_tags", [])])),
                )
            )
        self.samples_per_song = int(config["samples_per_song_per_epoch"] if training else 4)
        self.segment_samples = int(config["sample_rate"] * config["segment_seconds"])
        self.mel = make_mel_transform(
            config["sample_rate"], config["n_fft"], config["hop_length"], config["n_mels"]
        )

    def __len__(self) -> int:
        return len(self.songs) * self.samples_per_song

    def __getitem__(self, index: int) -> dict:
        song_index = index // self.samples_per_song
        sample_index = index % self.samples_per_song
        song = self.songs[song_index]
        inst, sr_i = load_audio(song.instrumental)
        voc, sr_v = load_audio(song.vocals)
        mix, sr_m = load_audio(song.mix)
        expected_sr = int(self.config["sample_rate"])
        if {sr_i, sr_v, sr_m} != {expected_sr}:
            raise RuntimeError(f"Prepared audio must be {expected_sr} Hz: {song.song_id}")

        length = min(inst.shape[-1], voc.shape[-1], mix.shape[-1])
        inst, voc, mix = inst[..., :length], voc[..., :length], mix[..., :length]

        if length <= self.segment_samples:
            start = 0
        elif self.training:
            start = random.randint(0, length - self.segment_samples)
        else:
            frac = (sample_index + 1) / (self.samples_per_song + 1)
            start = int((length - self.segment_samples) * frac)
        end = start + self.segment_samples

        def crop_pad(x: torch.Tensor) -> torch.Tensor:
            y = x[..., start:end]
            if y.shape[-1] < self.segment_samples:
                y = torch.nn.functional.pad(y, (0, self.segment_samples - y.shape[-1]))
            return y

        inst, voc, mix = crop_pad(inst), crop_pad(voc), crop_pad(mix)
        specs = torch.stack([
            waveform_to_scaled_mel(inst, self.mel, self.config["spec_frames"]),
            waveform_to_scaled_mel(voc, self.mel, self.config["spec_frames"]),
            waveform_to_scaled_mel(mix, self.mel, self.config["spec_frames"]),
        ], dim=0)

        tag_vec = torch.zeros(len(self.tag_vocab), dtype=torch.float32)
        for tag in song.tags:
            if tag in self.tag_vocab:
                tag_vec[self.tag_vocab.index(tag)] = 1.0

        return {
            "specs": specs,
            "tags": tag_vec,
            "weight": torch.tensor(song.weight, dtype=torch.float32),
            "song_id": song.song_id,
        }
