import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional

from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .meta_encoder import MetadataEncoder
from .fusion import create_fusion_module
from .losses import MultiModalLoss

logger = logging.getLogger(__name__)


class Lyric2VecModel(nn.Module):
    """Main Lyric2Vec model combining text, audio, and metadata encoders."""
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        audio_encoder: AudioEncoder,
        metadata_encoder: Optional[MetadataEncoder],
        fusion_module: nn.Module,
        loss_function: Optional[MultiModalLoss] = None
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.metadata_encoder = metadata_encoder
        self.fusion_module = fusion_module
        self.loss_function = loss_function
        
        # Check if metadata encoder is available
        self.use_metadata = metadata_encoder is not None
        
        logger.info(f"Lyric2VecModel initialized (metadata: {self.use_metadata})")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_spec: torch.Tensor,
        artist_id: Optional[torch.Tensor] = None,
        genre_id: Optional[torch.Tensor] = None,
        year_norm: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Text encoding
        z_text = self.text_encoder(input_ids, attention_mask)
        
        # Audio encoding
        z_audio = self.audio_encoder(mel_spec)
        
        # Metadata encoding (if available)
        if self.use_metadata and artist_id is not None:
            z_meta = self.metadata_encoder(artist_id, genre_id, year_norm)
        else:
            # Create dummy metadata embedding if not available
            batch_size = z_text.size(0)
            z_meta = torch.zeros(batch_size, self.metadata_encoder.get_embedding_dim(), 
                               device=z_text.device)
        
        # Fusion
        z_fused = self.fusion_module(z_text, z_audio, z_meta)
        
        return {
            "z_text": z_text,
            "z_audio": z_audio,
            "z_meta": z_meta,
            "z_fused": z_fused
        }
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text only."""
        return self.text_encoder(input_ids, attention_mask)
    
    def encode_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Encode audio only."""
        return self.audio_encoder(mel_spec)
    
    def encode_metadata(
        self, 
        artist_id: torch.Tensor, 
        genre_id: torch.Tensor, 
        year_norm: torch.Tensor
    ) -> torch.Tensor:
        """Encode metadata only."""
        if self.metadata_encoder is None:
            raise ValueError("Metadata encoder not available")
        return self.metadata_encoder(artist_id, genre_id, year_norm)
    
    def encode_fused(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_spec: torch.Tensor,
        artist_id: Optional[torch.Tensor] = None,
        genre_id: Optional[torch.Tensor] = None,
        year_norm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode and fuse all modalities."""
        outputs = self.forward(input_ids, attention_mask, mel_spec, artist_id, genre_id, year_norm)
        return outputs["z_fused"]
    
    def compute_losses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_spec: torch.Tensor,
        artist_id: Optional[torch.Tensor] = None,
        genre_id: Optional[torch.Tensor] = None,
        year_norm: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        if self.loss_function is None:
            raise ValueError("Loss function not available")
        
        # Forward pass
        outputs = self.forward(input_ids, attention_mask, mel_spec, artist_id, genre_id, year_norm)
        
        # Compute losses
        losses = self.loss_function(
            outputs["z_text"],
            outputs["z_audio"],
            outputs["z_meta"],
            outputs["z_fused"]
        )
        
        return losses
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.fusion_module.d
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "use_metadata": self.use_metadata,
                "embedding_dim": self.get_embedding_dim()
            }
        }
        
        if metadata:
            checkpoint["metadata"] = metadata
        
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Model checkpoint loaded from {path}")
        return checkpoint.get("metadata", {})


def create_lyric2vec_model(
    config,
    artist_vocab_size: int = 1000,
    genre_vocab_size: int = 100,
    use_metadata: bool = True
) -> Lyric2VecModel:
    """Create complete Lyric2Vec model from config."""
    # Create encoders
    text_encoder = TextEncoder(
        model_name=config.hf_text_model,
        output_dim=config.d,
        freeze_backbone=False
    )
    
    audio_encoder = AudioEncoder(
        input_channels=1,
        n_mels=config.n_mels,
        output_dim=config.d,
        hidden_dim=config.audio_dim,
        num_layers=2
    )
    
    # Create metadata encoder if needed
    metadata_encoder = None
    if use_metadata:
        metadata_encoder = MetadataEncoder(
            artist_vocab_size=artist_vocab_size,
            genre_vocab_size=genre_vocab_size,
            output_dim=config.meta_dim,
            embedding_dim=32
        )
    
    # Create fusion module
    fusion_module = create_fusion_module(config.fusion_type, config.d)
    
    # Create loss function
    loss_function = MultiModalLoss(
        loss_weights=config.loss_weights,
        temperature=config.temperature,
        triplet_margin=config.triplet_margin,
        use_triplet=True
    )
    
    # Create main model
    model = Lyric2VecModel(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        metadata_encoder=metadata_encoder,
        fusion_module=fusion_module,
        loss_function=loss_function
    )
    
    return model
