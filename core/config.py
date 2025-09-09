import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration."""
    # Text encoder
    hf_text_model: str = "distilbert-base-uncased"
    text_dim: int = 768
    
    # Audio encoder
    audio_sr: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    audio_dim: int = 256
    
    # Metadata encoder
    meta_dim: int = 64
    
    # Fusion
    fusion_type: str = "gated"  # "weighted_sum" or "gated"
    d: int = 256  # Final embedding dimension
    
    # Loss weights
    loss_weights: List[float] = None
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = [1.0, 0.5, 0.2]  # text-audio, text-meta, audio-meta


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 10
    
    # Training features
    fp16: bool = True
    gradient_clip_val: float = 1.0
    
    # Loss parameters
    temperature: float = 0.07
    triplet_margin: float = 0.2
    
    # Ray configuration
    ray_num_workers: int = 4
    ray_use_gpu: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    artifacts_path: str = "data/artifacts"
    
    # Dataset splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Audio preprocessing
    max_audio_length: float = 30.0  # seconds
    min_audio_length: float = 3.0   # seconds
    
    # Text preprocessing
    max_text_length: int = 512


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        "model": ModelConfig(
            hf_text_model=os.getenv("HF_TEXT_MODEL", "distilbert-base-uncased"),
            audio_sr=int(os.getenv("AUDIO_SR", "16000")),
            d=int(os.getenv("EMB_D", "256")),
            loss_weights=[float(x) for x in os.getenv("LOSS_WEIGHTS", "1.0,0.5,0.2").split(",")]
        ),
        "training": TrainingConfig(
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            learning_rate=float(os.getenv("LEARNING_RATE", "1e-4")),
            num_epochs=int(os.getenv("NUM_EPOCHS", "10")),
            fp16=os.getenv("FP16", "true").lower() == "true",
            ray_num_workers=int(os.getenv("RAY_NUM_WORKERS", "4")),
            ray_use_gpu=os.getenv("RAY_USE_GPU", "false").lower() == "true"
        ),
        "data": DataConfig(
            raw_path=os.getenv("DATA_RAW_PATH", "data/raw"),
            processed_path=os.getenv("DATA_PROCESSED_PATH", "data/processed"),
            artifacts_path=os.getenv("DATA_ARTIFACTS_PATH", "data/artifacts")
        )
    }
