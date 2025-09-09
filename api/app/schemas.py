from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    modal: Literal["text", "audio", "fused"] = Field(..., description="Search modality")
    query: Optional[str] = Field(None, description="Text query for text or fused search")
    audio_b64: Optional[str] = Field(None, description="Base64 encoded audio for audio search")
    track_id: Optional[str] = Field(None, description="Track ID for fused search by reference")
    k: int = Field(50, ge=1, le=1000, description="Number of results to return")


class SearchItem(BaseModel):
    track_id: str = Field(..., description="Track identifier")
    score: float = Field(..., description="Similarity score")
    artist: Optional[str] = Field(None, description="Artist name")
    title: Optional[str] = Field(None, description="Track title")
    genre: Optional[str] = Field(None, description="Genre")


class SearchResponse(BaseModel):
    items: List[SearchItem] = Field(..., description="Search results")
    query_time_ms: float = Field(..., description="Total query time in milliseconds")
    modal: str = Field(..., description="Search modality used")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")
