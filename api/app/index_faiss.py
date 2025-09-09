import time
import logging
from typing import List, Tuple, Optional
import numpy as np
import faiss
from .metrics import record_faiss_latency

logger = logging.getLogger(__name__)

# Global FAISS index and metadata
_faiss_index = None
_track_ids = None
_track_metadata = None


def load_faiss_index(index_path: str = "data/artifacts/fused.index", 
                    ids_path: str = "data/artifacts/track_ids.npy",
                    metadata_path: Optional[str] = None):
    """Load FAISS index and track metadata."""
    global _faiss_index, _track_ids, _track_metadata
    
    try:
        logger.info(f"Loading FAISS index from {index_path}")
        _faiss_index = faiss.read_index(index_path)
        
        logger.info(f"Loading track IDs from {ids_path}")
        _track_ids = np.load(ids_path)
        
        if metadata_path:
            logger.info(f"Loading track metadata from {metadata_path}")
            _track_metadata = np.load(metadata_path, allow_pickle=True).item()
        else:
            _track_metadata = {}
        
        logger.info(f"FAISS index loaded: {_faiss_index.ntotal} vectors")
        
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise


def search_fused(query_vector: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
    """Search fused embeddings."""
    if _faiss_index is None:
        raise RuntimeError("FAISS index not loaded")
    
    start_time = time.time()
    
    try:
        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        scores, indices = _faiss_index.search(query_vector, k)
        
        # Convert to results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(_track_ids):  # Valid index
                track_id = str(_track_ids[idx])
                results.append((track_id, float(score)))
        
        duration_ms = (time.time() - start_time) * 1000
        record_faiss_latency("fused", duration_ms)
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching fused embeddings: {e}")
        raise


def search_text(query_vector: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
    """Search text embeddings."""
    # For now, use the same fused search
    # In a real implementation, you'd have separate indices
    return search_fused(query_vector, k)


def search_audio(query_vector: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
    """Search audio embeddings."""
    # For now, use the same fused search
    # In a real implementation, you'd have separate indices
    return search_fused(query_vector, k)


def get_track_metadata(track_id: str) -> dict:
    """Get metadata for a track."""
    if _track_metadata and track_id in _track_metadata:
        return _track_metadata[track_id]
    return {}


def warmup_search():
    """Perform a warmup search to initialize the index."""
    if _faiss_index is None:
        logger.warning("FAISS index not loaded, skipping warmup")
        return
    
    try:
        logger.info("Performing warmup search")
        dummy_query = np.random.randn(1, _faiss_index.d).astype(np.float32)
        search_fused(dummy_query, k=1)
        logger.info("Warmup search completed")
    except Exception as e:
        logger.error(f"Warmup search failed: {e}")


def get_index_stats() -> dict:
    """Get FAISS index statistics."""
    if _faiss_index is None:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "total_vectors": _faiss_index.ntotal,
        "dimension": _faiss_index.d,
        "is_trained": _faiss_index.is_trained,
        "track_ids_loaded": len(_track_ids) if _track_ids is not None else 0
    }
