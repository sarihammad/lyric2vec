import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Text encoder using pre-trained transformer."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dim: int = 256,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pre-trained model
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Frozen backbone parameters for {model_name}")
        
        # Get hidden dimension
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, output_dim)
        )
        
        logger.info(f"TextEncoder initialized: {model_name} -> {output_dim}D")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through text encoder."""
        # Get transformer outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Project to output dimension
        embedding = self.projection(pooled_output)  # [batch_size, output_dim]
        
        return embedding
    
    def encode_text(self, text: str, tokenizer) -> torch.Tensor:
        """Encode single text string."""
        self.eval()
        with torch.no_grad():
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to same device as model
            device = next(self.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Encode
            embedding = self.forward(input_ids, attention_mask)
            
            return embedding.squeeze(0)  # Remove batch dimension
    
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


def create_text_encoder(config) -> TextEncoder:
    """Create text encoder from config."""
    return TextEncoder(
        model_name=config.hf_text_model,
        output_dim=config.d,
        freeze_backbone=False
    )
