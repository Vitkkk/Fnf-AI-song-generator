from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
import torchaudio


DB_FLOOR = -80.0
DB_CEIL = 0.0


def ffprobe_duration(path: str | Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return float(json.loads(out)["format"]["duration"])


def decode_with_ffmpeg(
    src: str | Path,
    dst: str | Path,
    sample_rate: int,
    channels: int,
    duration: float | None = None,
) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    if duration is not None:
        cmd += ["-af", "apad", "-t", f"{duration:.6f}"]
    cmd += ["-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, sample_rate


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.detach().cpu().clamp(-1.0, 1.0)
    torchaudio.save(str(path), waveform, sample_rate)


def sum_stems(instrumental: torch.Tensor, vocals: torch.Tensor) -> torch.Tensor:
    length = max(instrumental.shape[-1], vocals.shape[-1])
    inst = torch.nn.functional.pad(instrumental, (0, length - instrumental.shape[-1]))
    voc = torch.nn.functional.pad(vocals, (0, length - vocals.shape[-1]))
    mix = inst + voc
    peak = mix.abs().amax().clamp_min(1.0)
    return mix / peak


def make_mel_transform(sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        center=True,
    )


def waveform_to_scaled_mel(
    waveform: torch.Tensor,
    mel_transform,
    spec_frames: int,
) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    mel = mel_transform(waveform)
    mel = mel.clamp_min(1e-10)
    db = 10.0 * torch.log10(mel)
    db = db - db.amax(dim=(-2, -1), keepdim=True)
    db = db.clamp(DB_FLOOR, DB_CEIL)
    scaled = ((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)) * 2.0 - 1.0
    if scaled.shape[-1] < spec_frames:
        scaled = torch.nn.functional.pad(scaled, (0, spec_frames - scaled.shape[-1]), value=-1.0)
    elif scaled.shape[-1] > spec_frames:
        scaled = scaled[..., :spec_frames]
    return scaled


def scaled_mel_to_power(scaled: torch.Tensor) -> torch.Tensor:
    db = ((scaled + 1.0) * 0.5) * (DB_CEIL - DB_FLOOR) + DB_FLOOR
    return torch.pow(10.0, db / 10.0)


def power_to_scaled_mel(power: torch.Tensor) -> torch.Tensor:
    db = 10.0 * torch.log10(power.clamp_min(1e-10))
    db = db.clamp(DB_FLOOR, DB_CEIL)
    return ((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)) * 2.0 - 1.0


def mel_to_waveform(
    scaled_mel: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    length: int,
) -> torch.Tensor:
    """Invert a generated mel spectrogram robustly enough for v0.1 listening tests.

    Torchaudio's InverseMelScale relies on a least-squares solve that can fail
    when the mel filterbank is rank-deficient. Generated spectrograms also use
    a padded frame count, while Griffin-Lim with a fixed waveform length expects
    the exact centered-STFT frame count. A Moore-Penrose pseudo-inverse plus an
    explicit frame crop avoids both failure modes.
    """
    power_mel = scaled_mel_to_power(scaled_mel.detach().cpu())
    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    ).transpose(0, 1)
    inverse_fb = torch.linalg.pinv(fb)
    linear_power = torch.matmul(inverse_fb, power_mel).clamp_min(1e-10)

    expected_frames = 1 + length // hop_length
    if linear_power.shape[-1] < expected_frames:
        linear_power = torch.nn.functional.pad(
            linear_power, (0, expected_frames - linear_power.shape[-1]), value=1e-10
        )
    elif linear_power.shape[-1] > expected_frames:
        linear_power = linear_power[..., :expected_frames]

    magnitude = torch.sqrt(linear_power)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        power=1.0,
        n_iter=32,
        length=length,
    )
    return griffin_lim(magnitude)
