import pytest
import torch
import numpy as np
from core.models.text_encoder import TextEncoder
from core.models.audio_encoder import AudioEncoder
from core.models.meta_encoder import MetadataEncoder
from core.models.fusion import GatedFusion
from core.models.losses import info_nce, triplet_loss


def test_text_encoder():
    """Test text encoder."""
    encoder = TextEncoder(output_dim=256)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = encoder(input_ids, attention_mask)
    
    assert output.shape == (batch_size, 256)
    assert not torch.isnan(output).any()


def test_audio_encoder():
    """Test audio encoder."""
    encoder = AudioEncoder(output_dim=256)
    
    # Test forward pass
    batch_size = 2
    n_mels = 128
    time_steps = 1000
    mel_spec = torch.randn(batch_size, n_mels, time_steps)
    
    output = encoder(mel_spec)
    
    assert output.shape == (batch_size, 256)
    assert not torch.isnan(output).any()


def test_metadata_encoder():
    """Test metadata encoder."""
    encoder = MetadataEncoder(
        artist_vocab_size=100,
        genre_vocab_size=10,
        output_dim=64
    )
    
    # Test forward pass
    batch_size = 2
    artist_ids = torch.randint(0, 100, (batch_size,))
    genre_ids = torch.randint(0, 10, (batch_size,))
    year_norms = torch.randn(batch_size)
    
    output = encoder(artist_ids, genre_ids, year_norms)
    
    assert output.shape == (batch_size, 64)
    assert not torch.isnan(output).any()


def test_gated_fusion():
    """Test gated fusion."""
    fusion = GatedFusion(d=256)
    
    # Test forward pass
    batch_size = 2
    z_text = torch.randn(batch_size, 256)
    z_audio = torch.randn(batch_size, 256)
    z_meta = torch.randn(batch_size, 64)
    
    output = fusion(z_text, z_audio, z_meta)
    
    assert output.shape == (batch_size, 256)
    assert not torch.isnan(output).any()


def test_info_nce_loss():
    """Test InfoNCE loss."""
    batch_size = 4
    dim = 256
    
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    
    loss = info_nce(z1, z2)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_triplet_loss():
    """Test triplet loss."""
    batch_size = 4
    dim = 256
    
    anchor = torch.randn(batch_size, dim)
    positive = torch.randn(batch_size, dim)
    negative = torch.randn(batch_size, dim)
    
    loss = triplet_loss(anchor, positive, negative)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__])
