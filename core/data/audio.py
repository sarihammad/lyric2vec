import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing pipeline for music embeddings."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        max_length: float = 30.0,
        min_length: float = 3.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_length = max_length
        self.min_length = min_length
        
        # Audio transforms
        self.resample = T.Resample(orig_freq=44100, new_freq=sample_rate)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # SpecAugment for training
        self.spec_augment = T.SpecAugment(
            freq_mask_param=27,
            time_mask_param=100,
            num_freq_masks=2,
            num_time_masks=2
        )
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load audio file and return waveform."""
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                waveform = self.resample(waveform)
            
            return waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def preprocess_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Preprocess audio waveform."""
        # Trim silence
        waveform = torchaudio.functional.trim(waveform)[0]
        
        # Check length constraints
        duration = len(waveform) / self.sample_rate
        
        if duration < self.min_length:
            # Pad if too short
            pad_length = int(self.min_length * self.sample_rate) - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif duration > self.max_length:
            # Truncate if too long
            max_samples = int(self.max_length * self.sample_rate)
            waveform = waveform[:max_samples]
        
        return waveform
    
    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram."""
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db
    
    def apply_spec_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment for training."""
        return self.spec_augment(mel_spec)
    
    def process_audio_file(
        self, 
        file_path: str, 
        apply_augmentation: bool = False
    ) -> torch.Tensor:
        """Process audio file end-to-end."""
        # Load audio
        waveform = self.load_audio(file_path)
        
        # Preprocess
        waveform = self.preprocess_audio(waveform)
        
        # Convert to mel spectrogram
        mel_spec = self.waveform_to_mel(waveform)
        
        # Apply augmentation if training
        if apply_augmentation:
            mel_spec = self.apply_spec_augment(mel_spec)
        
        return mel_spec
    
    def batch_process(
        self, 
        file_paths: list, 
        apply_augmentation: bool = False
    ) -> torch.Tensor:
        """Process multiple audio files in batch."""
        mel_specs = []
        
        for file_path in file_paths:
            try:
                mel_spec = self.process_audio_file(file_path, apply_augmentation)
                mel_specs.append(mel_spec)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue
        
        if not mel_specs:
            raise ValueError("No valid audio files processed")
        
        # Pad to same length
        max_length = max(spec.shape[1] for spec in mel_specs)
        padded_specs = []
        
        for spec in mel_specs:
            if spec.shape[1] < max_length:
                pad_length = max_length - spec.shape[1]
                spec = torch.nn.functional.pad(spec, (0, pad_length))
            padded_specs.append(spec)
        
        return torch.stack(padded_specs)


def create_audio_processor(config) -> AudioProcessor:
    """Create audio processor from config."""
    return AudioProcessor(
        sample_rate=config.audio_sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        max_length=config.max_audio_length,
        min_length=config.min_audio_length
    )
