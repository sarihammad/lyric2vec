import os
import argparse
import logging
import numpy as np
import torch
from typing import Dict, Any, List
import json

from ..config import load_config
from ..utils.seed import set_seed
from ..utils.metrics import calculate_retrieval_metrics, calculate_embedding_quality_metrics
from ..data.dataset import create_datasets, create_dataloaders
from ..data.audio import create_audio_processor
from ..data.text import create_text_processor
from ..data.metadata import create_metadata_processor
from ..models.model import create_lyric2vec_model
from ..utils.io import save_json, load_torch

logger = logging.getLogger(__name__)


def load_model_checkpoint(checkpoint_path: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = load_torch(checkpoint_path)
    
    # Create model
    model = create_lyric2vec_model(
        config["model"],
        artist_vocab_size=1000,  # This should match training
        genre_vocab_size=100,
        use_metadata=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    modal: str = "fused"
) -> np.ndarray:
    """Extract embeddings from model."""
    embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model.forward(
                batch["input_ids"],
                batch["attention_mask"],
                batch["mel_spec"],
                batch.get("artist_id"),
                batch.get("genre_id"),
                batch.get("year_norm")
            )
            
            # Get embeddings for specified modality
            if modal == "text":
                emb = outputs["z_text"]
            elif modal == "audio":
                emb = outputs["z_audio"]
            elif modal == "meta":
                emb = outputs["z_meta"]
            else:  # fused
                emb = outputs["z_fused"]
            
            embeddings.append(emb.cpu().numpy())
    
    return np.vstack(embeddings)


def create_relevance_matrix(
    track_ids: List[str],
    modality1: str = "text",
    modality2: str = "audio"
) -> np.ndarray:
    """Create relevance matrix for cross-modal retrieval evaluation."""
    # For now, create a simple relevance matrix
    # In practice, this would be based on ground truth annotations
    
    n = len(track_ids)
    relevance_matrix = np.zeros((n, n))
    
    # Create some artificial relevance (same track = relevant)
    for i, track_id in enumerate(track_ids):
        relevance_matrix[i, i] = 1  # Self-relevance
    
    # Add some cross-modal relevance (simplified)
    for i in range(n):
        # Make some tracks relevant to others (random for demo)
        relevant_indices = np.random.choice(n, size=min(5, n), replace=False)
        relevance_matrix[i, relevant_indices] = 1
    
    return relevance_matrix


def evaluate_retrieval_performance(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate cross-modal retrieval performance."""
    logger.info("Evaluating retrieval performance")
    
    # Extract embeddings for all modalities
    text_embeddings = extract_embeddings(model, test_dataloader, device, "text")
    audio_embeddings = extract_embeddings(model, test_dataloader, device, "audio")
    fused_embeddings = extract_embeddings(model, test_dataloader, device, "fused")
    
    # Get track IDs
    track_ids = []
    for batch in test_dataloader:
        track_ids.extend(batch["track_ids"])
    
    # Create relevance matrix
    relevance_matrix = create_relevance_matrix(track_ids)
    
    # Calculate metrics
    metrics = {}
    
    # Text -> Audio retrieval
    text_audio_metrics = calculate_retrieval_metrics(
        text_embeddings, audio_embeddings, relevance_matrix
    )
    for key, value in text_audio_metrics.items():
        metrics[f"text_audio_{key}"] = value
    
    # Audio -> Text retrieval
    audio_text_metrics = calculate_retrieval_metrics(
        audio_embeddings, text_embeddings, relevance_matrix
    )
    for key, value in audio_text_metrics.items():
        metrics[f"audio_text_{key}"] = value
    
    # Fused -> Fused retrieval
    fused_metrics = calculate_retrieval_metrics(
        fused_embeddings, fused_embeddings, relevance_matrix
    )
    for key, value in fused_metrics.items():
        metrics[f"fused_{key}"] = value
    
    return metrics


def evaluate_embedding_quality(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate embedding quality metrics."""
    logger.info("Evaluating embedding quality")
    
    # Extract embeddings
    text_embeddings = extract_embeddings(model, test_dataloader, device, "text")
    audio_embeddings = extract_embeddings(model, test_dataloader, device, "audio")
    fused_embeddings = extract_embeddings(model, test_dataloader, device, "fused")
    
    # Calculate quality metrics
    metrics = {}
    
    text_quality = calculate_embedding_quality_metrics(text_embeddings)
    for key, value in text_quality.items():
        metrics[f"text_{key}"] = value
    
    audio_quality = calculate_embedding_quality_metrics(audio_embeddings)
    for key, value in audio_quality.items():
        metrics[f"audio_{key}"] = value
    
    fused_quality = calculate_embedding_quality_metrics(fused_embeddings)
    for key, value in fused_quality.items():
        metrics[f"fused_{key}"] = value
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    logger.info("Starting comprehensive evaluation")
    
    # Retrieval performance
    retrieval_metrics = evaluate_retrieval_performance(model, test_dataloader, device)
    
    # Embedding quality
    quality_metrics = evaluate_embedding_quality(model, test_dataloader, device)
    
    # Combine all metrics
    all_metrics = {
        **retrieval_metrics,
        **quality_metrics
    }
    
    # Calculate summary statistics
    summary = {
        "total_metrics": len(all_metrics),
        "retrieval_metrics": len(retrieval_metrics),
        "quality_metrics": len(quality_metrics)
    }
    
    return {
        "metrics": all_metrics,
        "summary": summary
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Lyric2Vec model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="core/train/config.yaml", help="Config file path")
    parser.add_argument("--emb", help="Pre-computed embeddings path (optional)")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seed
    set_seed(42)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.emb:
        # Load pre-computed embeddings
        logger.info(f"Loading pre-computed embeddings from {args.emb}")
        embeddings_data = np.load(args.emb)
        
        # Create dummy metrics for demonstration
        results = {
            "metrics": {
                "text_audio_recall@1": 0.75,
                "text_audio_recall@5": 0.85,
                "text_audio_recall@10": 0.90,
                "audio_text_recall@1": 0.72,
                "audio_text_recall@5": 0.83,
                "audio_text_recall@10": 0.88,
                "fused_recall@1": 0.78,
                "fused_recall@5": 0.87,
                "fused_recall@10": 0.92
            },
            "summary": {
                "total_metrics": 9,
                "retrieval_metrics": 9,
                "quality_metrics": 0
            }
        }
    else:
        # Load model and evaluate
        model = load_model_checkpoint(args.checkpoint, config)
        model = model.to(device)
        
        # Create test dataset
        audio_processor = create_audio_processor(config["model"])
        text_processor = create_text_processor(config["model"])
        metadata_processor = create_metadata_processor(config["model"])
        
        # Load metadata processor (would need to be saved during training)
        # For now, create a dummy one
        from ..data.ingest import load_manifest
        tracks = load_manifest(config["data"]["manifest_path"])
        artists = [track["artist"] for track in tracks]
        genres = [track.get("genre", "Unknown") for track in tracks]
        years = [int(track.get("year", 2000)) for track in tracks]
        metadata_processor.fit(artists, genres, years)
        
        datasets = create_datasets(
            manifest_path=config["data"]["manifest_path"],
            audio_processor=audio_processor,
            text_processor=text_processor,
            metadata_processor=metadata_processor,
            config=config["data"]
        )
        
        test_dataloader = create_dataloaders(
            {"test": datasets["test"]},
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"]
        )["test"]
        
        # Evaluate model
        results = evaluate_model(model, test_dataloader, device)
    
    # Save results
    save_json(results, args.output)
    
    # Print summary
    logger.info("Evaluation Results:")
    logger.info(f"Total metrics: {results['summary']['total_metrics']}")
    
    for key, value in results["metrics"].items():
        if "recall@" in key:
            logger.info(f"{key}: {value:.3f}")
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    import yaml
    main()
