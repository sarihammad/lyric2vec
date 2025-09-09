from fastapi import APIRouter, HTTPException
from typing import List
import time
import logging

from .schemas import SearchRequest, SearchResponse, SearchItem
from .encode import encode_text, encode_audio, get_fused_by_id
from .index_faiss import search_fused, search_text, search_audio, get_track_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search for similar tracks using text, audio, or fused embeddings."""
    start_time = time.time()
    
    try:
        if req.modal == "text":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for text modal")
            
            # Encode text query
            query_vector = encode_text(req.query)
            
            # Search text embeddings
            hits = search_text(query_vector, k=req.k)
            
        elif req.modal == "audio":
            if not req.audio_b64:
                raise HTTPException(status_code=400, detail="audio_b64 required for audio modal")
            
            # Encode audio query
            query_vector = encode_audio(req.audio_b64)
            
            # Search audio embeddings
            hits = search_audio(query_vector, k=req.k)
            
        elif req.modal == "fused":
            if req.query:
                # Use text query for fused search
                query_vector = encode_text(req.query)
            elif req.track_id:
                # Use existing track embedding
                query_vector = get_fused_by_id(req.track_id)
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="provide query or track_id for fused modal"
                )
            
            # Search fused embeddings
            hits = search_fused(query_vector, k=req.k)
            
        else:
            raise HTTPException(status_code=400, detail="invalid modal")
        
        # Convert hits to SearchItem objects
        items = []
        for track_id, score in hits:
            metadata = get_track_metadata(track_id)
            item = SearchItem(
                track_id=track_id,
                score=float(score),
                artist=metadata.get("artist"),
                title=metadata.get("title"),
                genre=metadata.get("genre")
            )
            items.append(item)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            items=items,
            query_time_ms=query_time_ms,
            modal=req.modal
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/similar/{track_id}")
def get_similar_tracks(track_id: str, k: int = 50):
    """Get similar tracks by track ID using fused embeddings."""
    if k < 1 or k > 1000:
        raise HTTPException(status_code=400, detail="k must be between 1 and 1000")
    
    start_time = time.time()
    
    try:
        # Get fused embedding for the track
        query_vector = get_fused_by_id(track_id)
        
        # Search for similar tracks
        hits = search_fused(query_vector, k=k + 1)  # +1 to exclude self
        
        # Filter out the query track itself
        filtered_hits = [(tid, score) for tid, score in hits if tid != track_id][:k]
        
        # Convert to SearchItem objects
        items = []
        for tid, score in filtered_hits:
            metadata = get_track_metadata(tid)
            item = SearchItem(
                track_id=tid,
                score=float(score),
                artist=metadata.get("artist"),
                title=metadata.get("title"),
                genre=metadata.get("genre")
            )
            items.append(item)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            items=items,
            query_time_ms=query_time_ms,
            modal="fused"
        )
        
    except Exception as e:
        logger.error(f"Similar tracks error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
