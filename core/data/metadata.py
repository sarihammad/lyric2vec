import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """Metadata processing for artist/genre embeddings."""
    
    def __init__(self, meta_dim: int = 64):
        self.meta_dim = meta_dim
        self.artist_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.year_scaler = StandardScaler()
        
        self.artist_vocab_size = 0
        self.genre_vocab_size = 0
        self.is_fitted = False
    
    def fit(self, artists: List[str], genres: List[str], years: List[int]):
        """Fit encoders on training data."""
        logger.info("Fitting metadata encoders")
        
        # Fit encoders
        self.artist_encoder.fit(artists)
        self.genre_encoder.fit(genres)
        self.year_scaler.fit(np.array(years).reshape(-1, 1))
        
        # Store vocab sizes
        self.artist_vocab_size = len(self.artist_encoder.classes_)
        self.genre_vocab_size = len(self.genre_encoder.classes_)
        
        self.is_fitted = True
        
        logger.info(f"Artist vocab size: {self.artist_vocab_size}")
        logger.info(f"Genre vocab size: {self.genre_vocab_size}")
    
    def transform(self, artists: List[str], genres: List[str], years: List[int]) -> np.ndarray:
        """Transform metadata to encoded format."""
        if not self.is_fitted:
            raise ValueError("MetadataProcessor must be fitted before transform")
        
        # Encode categorical features
        artist_ids = self.artist_encoder.transform(artists)
        genre_ids = self.genre_encoder.transform(genres)
        
        # Normalize year
        years_normalized = self.year_scaler.transform(np.array(years).reshape(-1, 1)).flatten()
        
        # Combine features
        metadata = np.column_stack([artist_ids, genre_ids, years_normalized])
        
        return metadata
    
    def process_metadata(
        self, 
        artist: str, 
        genre: str, 
        year: int
    ) -> Tuple[int, int, float]:
        """Process single metadata entry."""
        if not self.is_fitted:
            raise ValueError("MetadataProcessor must be fitted before processing")
        
        # Handle unknown artists/genres
        try:
            artist_id = self.artist_encoder.transform([artist])[0]
        except ValueError:
            # Unknown artist - use most common class
            artist_id = 0
        
        try:
            genre_id = self.genre_encoder.transform([genre])[0]
        except ValueError:
            # Unknown genre - use most common class
            genre_id = 0
        
        # Normalize year
        year_normalized = self.year_scaler.transform([[year]])[0, 0]
        
        return artist_id, genre_id, year_normalized
    
    def get_metadata_embedding(
        self, 
        artist_id: int, 
        genre_id: int, 
        year_normalized: float
    ) -> np.ndarray:
        """Get metadata embedding (placeholder - would use learned embeddings)."""
        # This is a placeholder implementation
        # In practice, you would use learned embedding layers
        
        # Create a simple embedding based on IDs and normalized year
        embedding = np.zeros(self.meta_dim)
        
        # Use artist and genre IDs to set some dimensions
        embedding[artist_id % self.meta_dim] = 1.0
        embedding[(genre_id + 10) % self.meta_dim] = 1.0
        
        # Use normalized year for other dimensions
        embedding[20:30] = year_normalized * 0.1
        
        # Add some noise for diversity
        embedding += np.random.normal(0, 0.1, self.meta_dim)
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def save(self, path: str):
        """Save fitted encoders."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'artist_encoder': self.artist_encoder,
                'genre_encoder': self.genre_encoder,
                'year_scaler': self.year_scaler,
                'artist_vocab_size': self.artist_vocab_size,
                'genre_vocab_size': self.genre_vocab_size,
                'meta_dim': self.meta_dim,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Saved metadata processor to {path}")
    
    def load(self, path: str):
        """Load fitted encoders."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.artist_encoder = data['artist_encoder']
        self.genre_encoder = data['genre_encoder']
        self.year_scaler = data['year_scaler']
        self.artist_vocab_size = data['artist_vocab_size']
        self.genre_vocab_size = data['genre_vocab_size']
        self.meta_dim = data['meta_dim']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Loaded metadata processor from {path}")


def create_metadata_processor(config) -> MetadataProcessor:
    """Create metadata processor from config."""
    return MetadataProcessor(meta_dim=config.meta_dim)
