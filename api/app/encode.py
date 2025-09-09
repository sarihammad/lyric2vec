import base64
import io
import time
import logging
from typing import Optional
import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel
import librosa
from .metrics import record_encode_latency

logger = logging.getLogger(__name__)

# Global model instances (lazy loaded)
_text_tokenizer = None
_text_model = None
_audio_pipeline = None
_audio_model = None


def load_text_encoder(model_name: str = "distilbert-base-uncased"):
    """Load text tokenizer and model."""
    global _text_tokenizer, _text_model
    
    if _text_tokenizer is None or _text_model is None:
        logger.info(f"Loading text encoder: {model_name}")
        _text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _text_model = AutoModel.from_pretrained(model_name)
        _text_model.eval()
        logger.info("Text encoder loaded successfully")
    
    return _text_tokenizer, _text_model


def load_audio_encoder():
    """Load audio processing pipeline and model."""
    global _audio_pipeline, _audio_model
    
    if _audio_pipeline is None or _audio_model is None:
        logger.info("Loading audio encoder")
        # This would load your trained audio model
        # For now, we'll create a placeholder
        _audio_pipeline = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        # Placeholder model - replace with your actual trained model
        _audio_model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 256)
        )
        _audio_model.eval()
        logger.info("Audio encoder loaded successfully")
    
    return _audio_pipeline, _audio_model


def encode_text(query: str) -> np.ndarray:
    """Encode text query to embedding."""
    start_time = time.time()
    
    try:
        tokenizer, model = load_text_encoder()
        
        # Tokenize input
        inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        duration_ms = (time.time() - start_time) * 1000
        record_encode_latency("text", duration_ms)
        
        return embedding[0]  # Return first (and only) embedding
        
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        raise


def encode_audio(audio_b64: str) -> np.ndarray:
    """Encode base64 audio to embedding."""
    start_time = time.time()
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_io = io.BytesIO(audio_bytes)
        
        # Load audio with librosa
        audio, sr = librosa.load(audio_io, sr=16000)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Load audio pipeline and model
        pipeline, model = load_audio_encoder()
        
        # Process audio
        with torch.no_grad():
            # Convert to mel spectrogram
            mel_spec = pipeline(audio_tensor.unsqueeze(0))
            
            # Add channel dimension
            mel_spec = mel_spec.unsqueeze(0)
            
            # Get embedding
            embedding = model(mel_spec).numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        
        duration_ms = (time.time() - start_time) * 1000
        record_encode_latency("audio", duration_ms)
        
        return embedding[0]  # Return first (and only) embedding
        
    except Exception as e:
        logger.error(f"Error encoding audio: {e}")
        raise


def get_fused_by_id(track_id: str) -> np.ndarray:
    """Get fused embedding by track ID."""
    # This would load from your pre-computed embeddings
    # For now, return a placeholder
    logger.warning(f"get_fused_by_id not implemented for track_id: {track_id}")
    return np.random.randn(256).astype(np.float32)
