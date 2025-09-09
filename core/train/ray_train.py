import os
import yaml
import logging
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ray
from ray import train
from ray.train import Trainer, ScalingConfig
from ray.train.torch import TorchTrainer
import numpy as np

from ..config import load_config
from ..utils.seed import set_seed
from ..data.dataset import create_datasets, create_dataloaders
from ..data.audio import create_audio_processor
from ..data.text import create_text_processor
from ..data.metadata import create_metadata_processor
from ..models.model import create_lyric2vec_model
from ..utils.io import save_json, ensure_dir

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, device, fp16=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        if fp16 and scaler:
            with torch.cuda.amp.autocast():
                losses = model.compute_losses(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["mel_spec"],
                    batch.get("artist_id"),
                    batch.get("genre_id"),
                    batch.get("year_norm")
                )
                loss = losses["total"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model.compute_losses(
                batch["input_ids"],
                batch["attention_mask"],
                batch["mel_spec"],
                batch.get("artist_id"),
                batch.get("genre_id"),
                batch.get("year_norm")
            )
            loss = losses["total"]
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress
        if num_batches % 10 == 0:
            logger.info(f"Batch {num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            losses = model.compute_losses(
                batch["input_ids"],
                batch["attention_mask"],
                batch["mel_spec"],
                batch.get("artist_id"),
                batch.get("genre_id"),
                batch.get("year_norm")
            )
            loss = losses["total"]
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_worker(config: Dict[str, Any]):
    """Training worker function for Ray."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    
    # Create processors
    audio_processor = create_audio_processor(model_config)
    text_processor = create_text_processor(model_config)
    
    # Create metadata processor (fit on training data)
    metadata_processor = create_metadata_processor(model_config)
    
    # Load and fit metadata processor
    from ..data.ingest import load_manifest
    tracks = load_manifest(data_config["manifest_path"])
    
    artists = [track["artist"] for track in tracks]
    genres = [track.get("genre", "Unknown") for track in tracks]
    years = [int(track.get("year", 2000)) for track in tracks]
    
    metadata_processor.fit(artists, genres, years)
    
    # Create datasets
    datasets = create_datasets(
        manifest_path=data_config["manifest_path"],
        audio_processor=audio_processor,
        text_processor=text_processor,
        metadata_processor=metadata_processor,
        config=data_config
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        datasets,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"]
    )
    
    # Create model
    model = create_lyric2vec_model(
        model_config,
        artist_vocab_size=metadata_processor.artist_vocab_size,
        genre_vocab_size=metadata_processor.genre_vocab_size,
        use_metadata=True
    )
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(training_config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, 
            dataloaders["train"], 
            optimizer, 
            device, 
            fp16=training_config["fp16"]
        )
        
        # Validate
        val_loss = validate_epoch(model, dataloaders["val"], device)
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Report metrics to Ray
        train.report({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model checkpoint
            checkpoint_dir = os.path.join(data_config["artifacts_path"], "checkpoints")
            ensure_dir(checkpoint_dir)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}.pt")
            model.save_checkpoint(checkpoint_path, {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": config
            })
            
            logger.info(f"Saved best model checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(data_config["artifacts_path"], "final_model.pt")
    model.save_checkpoint(final_path, {
        "final_epoch": training_config["num_epochs"],
        "final_val_loss": val_loss,
        "config": config
    })
    
    logger.info(f"Training completed. Final model saved: {final_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Lyric2Vec model with Ray")
    parser.add_argument("--config", default="core/train/config.yaml", help="Config file path")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of Ray workers")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override Ray config with command line args
    config["ray"]["num_workers"] = args.num_workers
    config["ray"]["use_gpu"] = args.use_gpu
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create scaling config
    scaling_config = ScalingConfig(
        num_workers=config["ray"]["num_workers"],
        use_gpu=config["ray"]["use_gpu"],
        resources_per_worker=config["ray"]["resources_per_worker"]
    )
    
    # Create trainer
    trainer = TorchTrainer(
        train_worker,
        train_loop_config=config,
        scaling_config=scaling_config
    )
    
    # Train
    logger.info("Starting Ray training")
    result = trainer.fit()
    
    logger.info("Training completed successfully")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
