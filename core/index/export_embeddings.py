import os
import argparse
import logging
import numpy as np
import torch
from typing import Dict, Any, List
import json

from ..config import load_config
from ..utils.seed import set_seed
from ..data.dataset import create_datasets, create_dataloaders
from ..data.audio import create_audio_processor
from ..data.text import create_text_processor
from ..data.metadata import create_metadata_processor
from ..models.model import create_lyric2vec_model
from ..utils.io import save_json, load_torch, ensure_dir

logger = logging.getLogger(__name__)


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


def export_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str
) -> Dict[str, str]:
    """Export all embeddings to files."""
    logger.info("Extracting embeddings for all modalities")
    
    # Extract embeddings
    text_embeddings = extract_embeddings(model, dataloader, device, "text")
    audio_embeddings = extract_embeddings(model, dataloader, device, "audio")
    meta_embeddings = extract_embeddings(model, dataloader, device, "meta")
    fused_embeddings = extract_embeddings(model, dataloader, device, "fused")
    
    # Get track IDs
    track_ids = []
    track_metadata = {}
    
    for batch in dataloader:
        track_ids.extend(batch["track_ids"])
        
        # Collect metadata
        for i, track_id in enumerate(batch["track_ids"]):
            track_metadata[track_id] = {
                "artist_id": batch["artist_id"][i].item() if "artist_id" in batch else 0,
                "genre_id": batch["genre_id"][i].item() if "genre_id" in batch else 0,
                "year_norm": batch["year_norm"][i].item() if "year_norm" in batch else 0.0
            }
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Save embeddings
    file_paths = {}
    
    # Text embeddings
    text_path = os.path.join(output_dir, "emb_text.npy")
    np.save(text_path, text_embeddings)
    file_paths["text"] = text_path
    logger.info(f"Saved text embeddings: {text_path} ({text_embeddings.shape})")
    
    # Audio embeddings
    audio_path = os.path.join(output_dir, "emb_audio.npy")
    np.save(audio_path, audio_embeddings)
    file_paths["audio"] = audio_path
    logger.info(f"Saved audio embeddings: {audio_path} ({audio_embeddings.shape})")
    
    # Metadata embeddings
    meta_path = os.path.join(output_dir, "emb_meta.npy")
    np.save(meta_path, meta_embeddings)
    file_paths["meta"] = meta_path
    logger.info(f"Saved metadata embeddings: {meta_path} ({meta_embeddings.shape})")
    
    # Fused embeddings
    fused_path = os.path.join(output_dir, "fused.npy")
    np.save(fused_path, fused_embeddings)
    file_paths["fused"] = fused_path
    logger.info(f"Saved fused embeddings: {fused_path} ({fused_embeddings.shape})")
    
    # Track IDs
    track_ids_path = os.path.join(output_dir, "track_ids.npy")
    np.save(track_ids_path, np.array(track_ids))
    file_paths["track_ids"] = track_ids_path
    logger.info(f"Saved track IDs: {track_ids_path} ({len(track_ids)} tracks)")
    
    # Track metadata
    metadata_path = os.path.join(output_dir, "track_metadata.npy")
    np.save(metadata_path, track_metadata)
    file_paths["metadata"] = metadata_path
    logger.info(f"Saved track metadata: {metadata_path}")
    
    # Save combined embeddings file
    combined_path = os.path.join(output_dir, "embeddings.npz")
    np.savez(
        combined_path,
        text=text_embeddings,
        audio=audio_embeddings,
        meta=meta_embeddings,
        fused=fused_embeddings,
        track_ids=np.array(track_ids)
    )
    file_paths["combined"] = combined_path
    logger.info(f"Saved combined embeddings: {combined_path}")
    
    # Save metadata summary
    summary = {
        "num_tracks": len(track_ids),
        "embedding_dimensions": {
            "text": text_embeddings.shape[1],
            "audio": audio_embeddings.shape[1],
            "meta": meta_embeddings.shape[1],
            "fused": fused_embeddings.shape[1]
        },
        "file_paths": file_paths
    }
    
    summary_path = os.path.join(output_dir, "embedding_summary.json")
    save_json(summary, summary_path)
    logger.info(f"Saved embedding summary: {summary_path}")
    
    return file_paths


def main():
    """Main function for exporting embeddings."""
    parser = argparse.ArgumentParser(description="Export embeddings from trained model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="core/train/config.yaml", help="Config file path")
    parser.add_argument("--output", default="data/artifacts", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--modal", choices=["text", "audio", "meta", "fused", "all"], 
                       default="all", help="Modality to export")
    
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
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = load_torch(args.checkpoint)
    
    model = create_lyric2vec_model(
        config["model"],
        artist_vocab_size=1000,  # This should match training
        genre_vocab_size=100,
        use_metadata=True
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Create dataset
    audio_processor = create_audio_processor(config["model"])
    text_processor = create_text_processor(config["model"])
    metadata_processor = create_metadata_processor(config["model"])
    
    # Load metadata processor (would need to be saved during training)
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
    
    # Use all data for embedding export
    all_dataloader = create_dataloaders(
        {"all": datasets["train"]},  # Use train set for now
        batch_size=args.batch_size,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"]
    )["all"]
    
    # Export embeddings
    file_paths = export_embeddings(model, all_dataloader, device, args.output)
    
    logger.info("Embedding export completed successfully")
    logger.info(f"Files saved to: {args.output}")
    
    for modal, path in file_paths.items():
        logger.info(f"  {modal}: {path}")


if __name__ == "__main__":
    import yaml
    main()
