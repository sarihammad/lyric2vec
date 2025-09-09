import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import os

from .ingest import load_manifest, validate_manifest
from .audio import AudioProcessor
from .text import TextProcessor
from .metadata import MetadataProcessor

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """Multimodal dataset for lyrics, audio, and metadata."""
    
    def __init__(
        self,
        manifest_path: str,
        audio_processor: AudioProcessor,
        text_processor: TextProcessor,
        metadata_processor: Optional[MetadataProcessor] = None,
        split: str = "train",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None
    ):
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.split = split
        
        # Load and validate manifest
        tracks = load_manifest(manifest_path)
        tracks = validate_manifest(tracks)
        
        # Limit samples if specified
        if max_samples:
            tracks = tracks[:max_samples]
        
        # Split data
        self.tracks = self._split_data(tracks, train_split, val_split, test_split)
        
        logger.info(f"Loaded {len(self.tracks)} tracks for {split} split")
    
    def _split_data(
        self, 
        tracks: List[Dict], 
        train_split: float, 
        val_split: float, 
        test_split: float
    ) -> List[Dict]:
        """Split data into train/val/test."""
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(tracks))
        
        train_end = int(len(tracks) * train_split)
        val_end = train_end + int(len(tracks) * val_split)
        
        if self.split == "train":
            split_indices = indices[:train_end]
        elif self.split == "val":
            split_indices = indices[train_end:val_end]
        elif self.split == "test":
            split_indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        return [tracks[i] for i in split_indices]
    
    def __len__(self) -> int:
        return len(self.tracks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        track = self.tracks[idx]
        
        # Load and process text
        try:
            with open(track["lyrics_path"], 'r', encoding='utf-8') as f:
                lyrics = f.read()
            text_data = self.text_processor.tokenize(lyrics)
        except Exception as e:
            logger.warning(f"Error loading lyrics for {track['track_id']}: {e}")
            # Use empty text as fallback
            text_data = self.text_processor.tokenize("")
        
        # Load and process audio
        try:
            audio_data = self.audio_processor.process_audio_file(
                track["audio_path"], 
                apply_augmentation=(self.split == "train")
            )
        except Exception as e:
            logger.warning(f"Error loading audio for {track['track_id']}: {e}")
            # Use zero tensor as fallback
            audio_data = torch.zeros(128, 1000)  # Default mel spec shape
        
        # Process metadata
        if self.metadata_processor:
            try:
                artist_id, genre_id, year_norm = self.metadata_processor.process_metadata(
                    track["artist"],
                    track.get("genre", "Unknown"),
                    int(track.get("year", 2000))
                )
                metadata = {
                    "artist_id": torch.tensor(artist_id, dtype=torch.long),
                    "genre_id": torch.tensor(genre_id, dtype=torch.long),
                    "year_norm": torch.tensor(year_norm, dtype=torch.float)
                }
            except Exception as e:
                logger.warning(f"Error processing metadata for {track['track_id']}: {e}")
                metadata = {
                    "artist_id": torch.tensor(0, dtype=torch.long),
                    "genre_id": torch.tensor(0, dtype=torch.long),
                    "year_norm": torch.tensor(0.0, dtype=torch.float)
                }
        else:
            metadata = {}
        
        return {
            "track_id": track["track_id"],
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
            "mel_spec": audio_data,
            **metadata
        }


def create_dataloader(
    dataset: MultimodalDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader for dataset."""
    
    def collate_fn(batch):
        """Custom collate function for multimodal data."""
        # Separate different data types
        track_ids = [item["track_id"] for item in batch]
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_masks = torch.stack([item["attention_mask"] for item in batch])
        mel_specs = torch.stack([item["mel_spec"] for item in batch])
        
        # Handle metadata if present
        if "artist_id" in batch[0]:
            artist_ids = torch.stack([item["artist_id"] for item in batch])
            genre_ids = torch.stack([item["genre_id"] for item in batch])
            year_norms = torch.stack([item["year_norm"] for item in batch])
            
            return {
                "track_ids": track_ids,
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "mel_spec": mel_specs,
                "artist_id": artist_ids,
                "genre_id": genre_ids,
                "year_norm": year_norms
            }
        else:
            return {
                "track_ids": track_ids,
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "mel_spec": mel_specs
            }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def create_datasets(
    manifest_path: str,
    audio_processor: AudioProcessor,
    text_processor: TextProcessor,
    metadata_processor: Optional[MetadataProcessor] = None,
    config: Optional[Any] = None
) -> Dict[str, MultimodalDataset]:
    """Create train/val/test datasets."""
    datasets = {}
    
    for split in ["train", "val", "test"]:
        datasets[split] = MultimodalDataset(
            manifest_path=manifest_path,
            audio_processor=audio_processor,
            text_processor=text_processor,
            metadata_processor=metadata_processor,
            split=split,
            train_split=config.train_split if config else 0.8,
            val_split=config.val_split if config else 0.1,
            test_split=config.test_split if config else 0.1,
            max_samples=config.max_samples if config else None
        )
    
    return datasets


def create_dataloaders(
    datasets: Dict[str, MultimodalDataset],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """Create DataLoaders for all datasets."""
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == "train")
        dataloaders[split] = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return dataloaders
