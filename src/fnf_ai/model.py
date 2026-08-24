from __future__ import annotations

import torch
from torch import nn


class StemCVAE(nn.Module):
    """Small conditional VAE over instrumental/vocal/mix log-mel spectrograms."""

    def __init__(
        self,
        n_mels: int = 128,
        spec_frames: int = 256,
        latent_dim: int = 256,
        base_channels: int = 32,
        num_tags: int = 1,
        tag_dim: int = 64,
    ):
        super().__init__()
        if n_mels % 16 or spec_frames % 16:
            raise ValueError("n_mels and spec_frames must be divisible by 16")
        self.n_mels = n_mels
        self.spec_frames = spec_frames
        self.latent_dim = latent_dim
        self.num_tags = num_tags
        self.tag_dim = tag_dim

        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.GroupNorm(8, c * 4), nn.SiLU(),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.GroupNorm(8, c * 8), nn.SiLU(),
        )
        self.h = n_mels // 16
        self.w = spec_frames // 16
        flat = c * 8 * self.h * self.w
        self.tag_encoder = nn.Sequential(nn.Linear(num_tags, tag_dim), nn.SiLU())
        self.mu = nn.Linear(flat + tag_dim, latent_dim)
        self.logvar = nn.Linear(flat + tag_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + tag_dim, flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1), nn.GroupNorm(8, c * 4), nn.SiLU(),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.GroupNorm(8, c * 2), nn.SiLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1),
            nn.Tanh(),
        )
        self.base_channels = c

    def encode(self, x: torch.Tensor, tags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        t = self.tag_encoder(tags)
        h = torch.cat([h, t], dim=1)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        t = self.tag_encoder(tags)
        h = self.decoder_input(torch.cat([z, t], dim=1))
        h = h.view(z.shape[0], self.base_channels * 8, self.h, self.w)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, tags: torch.Tensor):
        mu, logvar = self.encode(x, tags)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, tags)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, tags: torch.Tensor, seed: int | None = None, temperature: float = 1.0) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator(device=tags.device)
            generator.manual_seed(seed)
            z = torch.randn((tags.shape[0], self.latent_dim), generator=generator, device=tags.device)
        else:
            z = torch.randn((tags.shape[0], self.latent_dim), device=tags.device)
        return self.decode(z * temperature, tags)
