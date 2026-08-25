from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8 if channels >= 8 else 1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 128, embedding_dim: int = 48, commitment: float = 0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment = float(commitment)
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        bound = 1.0 / max(self.codebook_size, 1)
        nn.init.uniform_(self.embedding.weight, -bound, bound)

    def forward(self, z_e: torch.Tensor):
        b, d, t = z_e.shape
        flat = z_e.permute(0, 2, 1).contiguous().view(-1, d)
        emb = self.embedding.weight
        distances = flat.pow(2).sum(dim=1, keepdim=True) + emb.pow(2).sum(dim=1).unsqueeze(0) - 2.0 * flat @ emb.t()
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(b, t, d).permute(0, 2, 1).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment * commitment_loss
        z_st = z_e + (z_q - z_e).detach()
        with torch.no_grad():
            counts = torch.bincount(indices, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            usage = (counts > 0).float().mean()
        return z_st, indices.view(b, t), vq_loss, perplexity, usage

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        q = self.embedding(indices)
        return q.permute(0, 2, 1).contiguous()


class TemporalVQCodec(nn.Module):
    """Shared mono waveform codec for FNF mix/instrumental/vocals.

    Six stride-2 downsamples give a hop of 64 waveform samples. At 8 kHz this is
    125 discrete tokens per second.
    """

    def __init__(self, base_channels: int = 32, latent_dim: int = 48, codebook_size: int = 128, commitment: float = 0.25):
        super().__init__()
        c = int(base_channels)
        self.latent_dim = int(latent_dim)
        self.hop_length = 64
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c, 7, padding=3), nn.SiLU(), ResBlock1d(c, 1),
            nn.Conv1d(c, c * 2, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 2, 1),
            nn.Conv1d(c * 2, c * 3, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 3, 2),
            nn.Conv1d(c * 3, c * 4, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 4, 2),
            nn.Conv1d(c * 4, c * 4, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 4, 4),
            nn.Conv1d(c * 4, c * 4, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 4, 4),
            nn.Conv1d(c * 4, latent_dim, 4, stride=2, padding=1),
        )
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, commitment)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, c * 4, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 4, 4),
            nn.ConvTranspose1d(c * 4, c * 4, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 4, 4),
            nn.ConvTranspose1d(c * 4, c * 4, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 4, 2),
            nn.ConvTranspose1d(c * 4, c * 3, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 3, 2),
            nn.ConvTranspose1d(c * 3, c * 2, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c * 2, 1),
            nn.ConvTranspose1d(c * 2, c, 4, stride=2, padding=1), nn.SiLU(), ResBlock1d(c, 1),
            nn.Conv1d(c, 1, 7, padding=3), nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        z_e = self.encoder(x)
        return self.quantizer(z_e)

    def decode(self, z_q: torch.Tensor, length: int | None = None) -> torch.Tensor:
        y = self.decoder(z_q)
        if length is not None:
            if y.shape[-1] < length:
                y = F.pad(y, (0, length - y.shape[-1]))
            y = y[..., :length]
        return y

    def forward(self, x: torch.Tensor):
        z_q, indices, vq_loss, perplexity, usage = self.encode(x)
        y = self.decode(z_q, x.shape[-1])
        return y, indices, vq_loss, perplexity, usage

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z_q, *_ = self.encode(x)
        return self.decode(z_q, x.shape[-1])

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        _, indices, *_ = self.encode(x)
        return indices

    @torch.no_grad()
    def decode_tokens(self, indices: torch.Tensor, length: int | None = None) -> torch.Tensor:
        z_q = self.quantizer.decode_indices(indices)
        return self.decode(z_q, length)
