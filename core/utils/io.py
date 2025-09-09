import os
import json
import pickle
import logging
from typing import Any, Dict, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str):
    """Save data as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Dict[str, Any]:
    """Load data from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {path}")
    return data


def save_pickle(data: Any, path: str):
    """Save data as pickle."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {path}")


def load_pickle(path: str) -> Any:
    """Load data from pickle."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded pickle from {path}")
    return data


def save_numpy(data: np.ndarray, path: str):
    """Save numpy array."""
    ensure_dir(os.path.dirname(path))
    np.save(path, data)
    logger.info(f"Saved numpy array to {path}")


def load_numpy(path: str) -> np.ndarray:
    """Load numpy array."""
    data = np.load(path)
    logger.info(f"Loaded numpy array from {path}")
    return data


def save_torch(model: torch.nn.Module, path: str, metadata: Optional[Dict] = None):
    """Save PyTorch model."""
    ensure_dir(os.path.dirname(path))
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    
    if metadata:
        save_dict["metadata"] = metadata
    
    torch.save(save_dict, path)
    logger.info(f"Saved PyTorch model to {path}")


def load_torch(path: str) -> Dict[str, Any]:
    """Load PyTorch model."""
    data = torch.load(path, map_location="cpu")
    logger.info(f"Loaded PyTorch model from {path}")
    return data


def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(path)


def list_files(directory: str, extension: Optional[str] = None) -> list:
    """List files in directory with optional extension filter."""
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if extension is None or filename.endswith(extension):
                files.append(filepath)
    
    return sorted(files)
