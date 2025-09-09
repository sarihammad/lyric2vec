import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """Audio encoder using CNN + BiGRU architecture."""
    
    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        output_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.n_mels = n_mels
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Keep time dimension
        )
        
        # Calculate CNN output channels
        cnn_output_channels = 256
        
        # BiGRU for temporal modeling
        self.gru = nn.GRU(
            input_size=cnn_output_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Projection head
        gru_output_dim = hidden_dim * 2  # Bidirectional
        self.projection = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gru_output_dim // 2, output_dim)
        )
        
        logger.info(f"AudioEncoder initialized: {n_mels}D mel-spec -> {output_dim}D")
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass through audio encoder."""
        batch_size = mel_spec.shape[0]
        
        # Add channel dimension if needed
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, n_mels, time]
        
        # CNN feature extraction
        cnn_features = self.cnn(mel_spec)  # [batch, 256, 1, time]
        
        # Reshape for GRU: [batch, time, channels]
        cnn_features = cnn_features.squeeze(2).transpose(1, 2)  # [batch, time, 256]
        
        # GRU for temporal modeling
        gru_output, _ = self.gru(cnn_features)  # [batch, time, hidden_dim*2]
        
        # Global average pooling over time
        pooled_output = torch.mean(gru_output, dim=1)  # [batch, hidden_dim*2]
        
        # Project to output dimension
        embedding = self.projection(pooled_output)  # [batch, output_dim]
        
        return embedding
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


class CRNNAudioEncoder(nn.Module):
    """Alternative CRNN-based audio encoder."""
    
    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        output_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.n_mels = n_mels
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])
        
        # Calculate conv output size
        conv_output_size = 128 * (n_mels // 8)  # After 3 maxpool layers
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass through CRNN encoder."""
        batch_size = mel_spec.shape[0]
        
        # Add channel dimension if needed
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        
        # Convolutional layers
        x = mel_spec
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for RNN: [batch, time, features]
        x = x.view(batch_size, x.size(3), -1)
        
        # RNN
        rnn_output, _ = self.rnn(x)
        
        # Global average pooling
        pooled = torch.mean(rnn_output, dim=1)
        
        # Projection
        embedding = self.projection(pooled)
        
        return embedding


def create_audio_encoder(config) -> AudioEncoder:
    """Create audio encoder from config."""
    return AudioEncoder(
        input_channels=1,
        n_mels=config.n_mels,
        output_dim=config.d,
        hidden_dim=config.audio_dim,
        num_layers=2
    )
