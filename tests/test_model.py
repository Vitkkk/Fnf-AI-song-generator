import torch

from fnf_ai.audio import mel_to_waveform
from fnf_ai.losses import weighted_cvae_loss
from fnf_ai.model import StemCVAE


def test_model_shapes_and_loss():
    model = StemCVAE(n_mels=32, spec_frames=32, latent_dim=16, base_channels=8, num_tags=3, tag_dim=8)
    x = torch.randn(2, 3, 32, 32).tanh()
    tags = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    recon, mu, logvar = model(x, tags)
    assert recon.shape == x.shape
    loss, metrics = weighted_cvae_loss(recon, x, mu, logvar, torch.ones(2), 0.001, 0.1)
    assert torch.isfinite(loss)
    assert metrics["loss"] > 0


def test_sampling_is_seeded():
    model = StemCVAE(n_mels=32, spec_frames=32, latent_dim=16, base_channels=8, num_tags=1, tag_dim=8)
    tags = torch.ones(1, 1)
    a = model.sample(tags, seed=42)
    b = model.sample(tags, seed=42)
    assert torch.allclose(a, b)


def test_mel_to_waveform_handles_padded_model_frames():
    # The model can emit a padded spectrogram width that is larger than the
    # centered STFT width implied by the requested waveform length.
    scaled_mel = torch.zeros(16, 32)
    waveform = mel_to_waveform(
        scaled_mel,
        sample_rate=8000,
        n_fft=64,
        hop_length=32,
        n_mels=16,
        length=800,
    )
    assert waveform.shape[-1] == 800
    assert torch.isfinite(waveform).all()
