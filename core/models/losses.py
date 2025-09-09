import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE loss for contrastive learning."""
    # Normalize embeddings
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(z1, z2.t()) / temperature
    
    # Labels are diagonal (positive pairs)
    labels = torch.arange(z1.size(0), device=z1.device)
    
    # Symmetric loss
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.t(), labels)
    
    return 0.5 * (loss_1 + loss_2)


def triplet_loss(
    anchor: torch.Tensor, 
    positive: torch.Tensor, 
    negative: torch.Tensor, 
    margin: float = 0.2
) -> torch.Tensor:
    """Triplet loss for metric learning."""
    # Compute distances
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    
    # Triplet loss
    loss = F.relu(pos_dist - neg_dist + margin)
    
    return loss.mean()


def contrastive_loss(
    z1: torch.Tensor, 
    z2: torch.Tensor, 
    temperature: float = 0.07
) -> torch.Tensor:
    """Contrastive loss (simplified InfoNCE)."""
    # Normalize embeddings
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute similarities
    similarities = torch.matmul(z1, z2.t()) / temperature
    
    # Create labels (diagonal is positive)
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z1.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(similarities, labels)
    
    return loss


def cosine_similarity_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss (maximize similarity)."""
    # Normalize embeddings
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute cosine similarities
    similarities = torch.sum(z1 * z2, dim=-1)
    
    # Maximize similarity (minimize negative similarity)
    loss = -similarities.mean()
    
    return loss


def mse_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(z1, z2)


class MultiModalLoss(nn.Module):
    """Combined loss for multimodal learning."""
    
    def __init__(
        self,
        loss_weights: list = [1.0, 0.5, 0.2],
        temperature: float = 0.07,
        triplet_margin: float = 0.2,
        use_triplet: bool = True
    ):
        super().__init__()
        
        self.loss_weights = loss_weights
        self.temperature = temperature
        self.triplet_margin = triplet_margin
        self.use_triplet = use_triplet
        
        logger.info(f"MultiModalLoss initialized with weights: {loss_weights}")
    
    def forward(
        self,
        z_text: torch.Tensor,
        z_audio: torch.Tensor,
        z_meta: torch.Tensor,
        z_fused: torch.Tensor
    ) -> dict:
        """Compute multimodal losses."""
        losses = {}
        
        # Text-Audio contrastive loss
        losses["text_audio"] = info_nce(z_text, z_audio, self.temperature)
        
        # Text-Metadata contrastive loss
        losses["text_meta"] = info_nce(z_text, z_meta, self.temperature)
        
        # Audio-Metadata contrastive loss
        losses["audio_meta"] = info_nce(z_audio, z_meta, self.temperature)
        
        # Fused embedding losses
        losses["fused_text"] = cosine_similarity_loss(z_fused, z_text)
        losses["fused_audio"] = cosine_similarity_loss(z_fused, z_audio)
        losses["fused_meta"] = cosine_similarity_loss(z_fused, z_meta)
        
        # Triplet loss (if enabled)
        if self.use_triplet:
            # Create triplets (anchor=text, positive=audio, negative=random)
            batch_size = z_text.size(0)
            indices = torch.randperm(batch_size, device=z_text.device)
            z_audio_neg = z_audio[indices]
            
            losses["triplet"] = triplet_loss(z_text, z_audio, z_audio_neg, self.triplet_margin)
        else:
            losses["triplet"] = torch.tensor(0.0, device=z_text.device)
        
        # Weighted total loss
        total_loss = (
            self.loss_weights[0] * losses["text_audio"] +
            self.loss_weights[1] * losses["text_meta"] +
            self.loss_weights[2] * losses["audio_meta"] +
            0.1 * (losses["fused_text"] + losses["fused_audio"] + losses["fused_meta"]) +
            0.1 * losses["triplet"]
        )
        
        losses["total"] = total_loss
        
        return losses


class HardNegativeMiningLoss(nn.Module):
    """Loss with hard negative mining."""
    
    def __init__(self, temperature: float = 0.07, hard_ratio: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.hard_ratio = hard_ratio
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute loss with hard negative mining."""
        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Compute similarity matrix
        similarities = torch.matmul(z1, z2.t()) / self.temperature
        
        # Get hard negatives (highest similarities excluding diagonal)
        batch_size = z1.size(0)
        mask = ~torch.eye(batch_size, dtype=bool, device=z1.device)
        
        # Select hard negatives
        num_hard = int(batch_size * self.hard_ratio)
        hard_negatives = torch.topk(similarities[mask], num_hard, largest=True)[1]
        
        # Create labels
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute loss only on hard negatives
        loss = F.cross_entropy(similarities, labels)
        
        return loss


def create_loss_function(config) -> MultiModalLoss:
    """Create loss function from config."""
    return MultiModalLoss(
        loss_weights=config.loss_weights,
        temperature=config.temperature,
        triplet_margin=config.triplet_margin,
        use_triplet=True
    )
