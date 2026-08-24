from __future__ import annotations

import torch

from .audio import power_to_scaled_mel, scaled_mel_to_power


def weighted_cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    weights: torch.Tensor,
    beta_kl: float,
    lambda_consistency: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    rec_per = (recon - target).abs().mean(dim=(1, 2, 3))
    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)

    inst_power = scaled_mel_to_power(recon[:, 0])
    voc_power = scaled_mel_to_power(recon[:, 1])
    synthesized_mix = power_to_scaled_mel(inst_power + voc_power)
    consistency_per = (recon[:, 2] - synthesized_mix).abs().mean(dim=(1, 2))

    w = weights / weights.mean().clamp_min(1e-6)
    total_per = rec_per + beta_kl * kl_per + lambda_consistency * consistency_per
    total = (total_per * w).mean()
    return total, {
        "loss": float(total.detach()),
        "reconstruction": float(rec_per.mean().detach()),
        "kl": float(kl_per.mean().detach()),
        "consistency": float(consistency_per.mean().detach()),
    }
