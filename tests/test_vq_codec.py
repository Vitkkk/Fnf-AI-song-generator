import torch
from fnf_ai.vq_codec import TemporalVQCodec
from fnf_ai.codec_losses import codec_loss


def test_temporal_vq_codec_roundtrip_shapes_and_tokens():
    model=TemporalVQCodec(base_channels=8,latent_dim=16,codebook_size=32)
    x=torch.randn(2,1,4096).tanh()
    y,tokens,vq,ppl,usage=model(x)
    assert y.shape==x.shape
    assert tokens.shape==(2,64)
    assert torch.isfinite(vq) and torch.isfinite(ppl) and torch.isfinite(usage)
    loss,_=codec_loss(y,x,vq)
    assert torch.isfinite(loss)


def test_decode_tokens_matches_length():
    model=TemporalVQCodec(base_channels=8,latent_dim=16,codebook_size=32)
    tokens=torch.zeros(1,50,dtype=torch.long)
    y=model.decode_tokens(tokens,length=3000)
    assert y.shape==(1,1,3000)
