import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class WeightedSumFusion(nn.Module):
    """Weighted sum fusion with learnable weights."""
    
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # Learnable weights for each modality
        self.weights = nn.Parameter(torch.ones(3) / 3)  # text, audio, metadata
        
        # Softmax to ensure weights sum to 1
        self.softmax = nn.Softmax(dim=0)
        
        logger.info(f"WeightedSumFusion initialized with dimension {d}")
    
    def forward(self, z_text: torch.Tensor, z_audio: torch.Tensor, z_meta: torch.Tensor) -> torch.Tensor:
        """Forward pass through weighted sum fusion."""
        # Normalize weights
        weights = self.softmax(self.weights)
        
        # Weighted combination
        z_fused = weights[0] * z_text + weights[1] * z_audio + weights[2] * z_meta
        
        return z_fused


class GatedFusion(nn.Module):
    """Gated fusion with learnable gates."""
    
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Sigmoid()
        )
        
        logger.info(f"GatedFusion initialized with dimension {d}")
    
    def forward(self, z_text: torch.Tensor, z_audio: torch.Tensor, z_meta: torch.Tensor) -> torch.Tensor:
        """Forward pass through gated fusion."""
        # Concatenate all modalities
        z_concat = torch.cat([z_text, z_audio, z_meta], dim=-1)
        
        # Compute gate
        gate = self.gate_net(z_concat)
        
        # Gated combination (text and audio are primary, metadata is auxiliary)
        z_fused = gate * z_text + (1 - gate) * z_audio + 0.1 * z_meta
        
        return z_fused


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism."""
    
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d * 4, d)
        )
        
        logger.info(f"AttentionFusion initialized with dimension {d}")
    
    def forward(self, z_text: torch.Tensor, z_audio: torch.Tensor, z_meta: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention fusion."""
        batch_size = z_text.shape[0]
        
        # Stack modalities as sequence
        modalities = torch.stack([z_text, z_audio, z_meta], dim=1)  # [batch, 3, d]
        
        # Self-attention
        attended, _ = self.attention(modalities, modalities, modalities)
        
        # Residual connection and layer norm
        attended = self.layer_norm(attended + modalities)
        
        # Feed-forward
        ffn_out = self.ffn(attended)
        
        # Final residual connection
        output = self.layer_norm(ffn_out + attended)
        
        # Average over modalities
        z_fused = torch.mean(output, dim=1)  # [batch, d]
        
        return z_fused


class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion."""
    
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # Cross-attention layers
        self.text_audio_attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.audio_text_attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d, d)
        )
        
        logger.info(f"CrossModalFusion initialized with dimension {d}")
    
    def forward(self, z_text: torch.Tensor, z_audio: torch.Tensor, z_meta: torch.Tensor) -> torch.Tensor:
        """Forward pass through cross-modal fusion."""
        batch_size = z_text.shape[0]
        
        # Reshape for attention
        z_text_seq = z_text.unsqueeze(1)  # [batch, 1, d]
        z_audio_seq = z_audio.unsqueeze(1)  # [batch, 1, d]
        
        # Cross-attention
        text_attended, _ = self.text_audio_attn(z_text_seq, z_audio_seq, z_audio_seq)
        audio_attended, _ = self.audio_text_attn(z_audio_seq, z_text_seq, z_text_seq)
        
        # Remove sequence dimension
        text_attended = text_attended.squeeze(1)
        audio_attended = audio_attended.squeeze(1)
        
        # Combine all modalities
        combined = torch.cat([text_attended, audio_attended, z_meta], dim=-1)
        
        # Final fusion
        z_fused = self.fusion(combined)
        
        return z_fused


def create_fusion_module(fusion_type: str, d: int) -> nn.Module:
    """Create fusion module based on type."""
    if fusion_type == "weighted_sum":
        return WeightedSumFusion(d)
    elif fusion_type == "gated":
        return GatedFusion(d)
    elif fusion_type == "attention":
        return AttentionFusion(d)
    elif fusion_type == "cross_modal":
        return CrossModalFusion(d)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
