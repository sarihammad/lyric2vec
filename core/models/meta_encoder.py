import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MetadataEncoder(nn.Module):
    """Metadata encoder for artist/genre/year information."""
    
    def __init__(
        self,
        artist_vocab_size: int,
        genre_vocab_size: int,
        output_dim: int = 64,
        embedding_dim: int = 32
    ):
        super().__init__()
        
        self.artist_vocab_size = artist_vocab_size
        self.genre_vocab_size = genre_vocab_size
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.artist_embedding = nn.Embedding(artist_vocab_size, embedding_dim)
        self.genre_embedding = nn.Embedding(genre_vocab_size, embedding_dim)
        
        # Year processing (single value)
        self.year_projection = nn.Linear(1, embedding_dim)
        
        # Fusion MLP
        total_dim = embedding_dim * 3  # artist + genre + year
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim // 2, output_dim)
        )
        
        logger.info(f"MetadataEncoder initialized: {artist_vocab_size} artists, {genre_vocab_size} genres -> {output_dim}D")
    
    def forward(
        self, 
        artist_ids: torch.Tensor, 
        genre_ids: torch.Tensor, 
        year_norms: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through metadata encoder."""
        # Get embeddings
        artist_emb = self.artist_embedding(artist_ids)  # [batch, embedding_dim]
        genre_emb = self.genre_embedding(genre_ids)     # [batch, embedding_dim]
        
        # Project year
        year_emb = self.year_projection(year_norms.unsqueeze(-1))  # [batch, embedding_dim]
        
        # Concatenate all features
        combined = torch.cat([artist_emb, genre_emb, year_emb], dim=-1)  # [batch, embedding_dim*3]
        
        # Fusion MLP
        metadata_embedding = self.fusion_mlp(combined)  # [batch, output_dim]
        
        return metadata_embedding
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


class SimpleMetadataEncoder(nn.Module):
    """Simplified metadata encoder using one-hot encoding."""
    
    def __init__(
        self,
        artist_vocab_size: int,
        genre_vocab_size: int,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.artist_vocab_size = artist_vocab_size
        self.genre_vocab_size = genre_vocab_size
        self.output_dim = output_dim
        
        # One-hot encoding dimensions
        total_input_dim = artist_vocab_size + genre_vocab_size + 1  # +1 for year
        
        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_input_dim // 2, output_dim)
        )
        
        logger.info(f"SimpleMetadataEncoder initialized: {total_input_dim}D -> {output_dim}D")
    
    def forward(
        self, 
        artist_ids: torch.Tensor, 
        genre_ids: torch.Tensor, 
        year_norms: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through simple metadata encoder."""
        batch_size = artist_ids.shape[0]
        
        # One-hot encoding
        artist_onehot = F.one_hot(artist_ids, num_classes=self.artist_vocab_size).float()
        genre_onehot = F.one_hot(genre_ids, num_classes=self.genre_vocab_size).float()
        
        # Combine features
        combined = torch.cat([
            artist_onehot,
            genre_onehot,
            year_norms.unsqueeze(-1)
        ], dim=-1)
        
        # MLP
        metadata_embedding = self.mlp(combined)
        
        return metadata_embedding


def create_metadata_encoder(
    config, 
    artist_vocab_size: int, 
    genre_vocab_size: int,
    use_simple: bool = False
) -> nn.Module:
    """Create metadata encoder from config."""
    if use_simple:
        return SimpleMetadataEncoder(
            artist_vocab_size=artist_vocab_size,
            genre_vocab_size=genre_vocab_size,
            output_dim=config.meta_dim
        )
    else:
        return MetadataEncoder(
            artist_vocab_size=artist_vocab_size,
            genre_vocab_size=genre_vocab_size,
            output_dim=config.meta_dim,
            embedding_dim=32
        )
