from __future__ import annotations

import torch
import torch.nn.functional as F


def _stft(x: torch.Tensor, n_fft: int):
    hop = n_fft // 4
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=True, return_complex=True)


def multi_resolution_stft_loss(pred: torch.Tensor, target: torch.Tensor, ffts=(256,)):
    total = pred.new_zeros(())
    mag_total = pred.new_zeros(())
    complex_total = pred.new_zeros(())
    for n_fft in ffts:
        p = _stft(pred, n_fft)
        t = _stft(target, n_fft)
        pm, tm = p.abs(), t.abs()
        spectral_convergence = torch.linalg.vector_norm(pm - tm) / torch.linalg.vector_norm(tm).clamp_min(1e-6)
        log_mag = F.l1_loss(torch.log(pm + 1e-5), torch.log(tm + 1e-5))
        complex_l1 = F.l1_loss(torch.view_as_real(p), torch.view_as_real(t))
        mag_part = spectral_convergence + log_mag
        total = total + mag_part + 0.15 * complex_l1
        mag_total = mag_total + mag_part
        complex_total = complex_total + complex_l1
    n = float(len(ffts))
    return total / n, mag_total / n, complex_total / n


def codec_loss(pred: torch.Tensor, target: torch.Tensor, vq_loss: torch.Tensor, waveform_weight: float = 1.0, stft_weight: float = 1.0, vq_weight: float = 0.25):
    wave = F.l1_loss(pred, target)
    stft, mag, complex_l1 = multi_resolution_stft_loss(pred, target)
    total = waveform_weight * wave + stft_weight * stft + vq_weight * vq_loss
    return total, {"loss": float(total.detach()), "wave_l1": float(wave.detach()), "stft": float(stft.detach()), "stft_mag": float(mag.detach()), "stft_complex": float(complex_l1.detach()), "vq": float(vq_loss.detach())}
