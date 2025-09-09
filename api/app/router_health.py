from fastapi import APIRouter
import time
import logging

from .schemas import HealthResponse
from .metrics import get_uptime_seconds, get_metrics
from .index_faiss import get_index_stats

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    try:
        # Check if index is loaded
        index_stats = get_index_stats()
        
        if not index_stats.get("loaded", False):
            logger.warning("FAISS index not loaded")
            return HealthResponse(
                status="degraded",
                version="1.0.0",
                uptime_seconds=get_uptime_seconds()
            )
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=get_uptime_seconds()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            uptime_seconds=get_uptime_seconds()
        )


@router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return get_metrics()


@router.get("/status")
def status():
    """Detailed status information."""
    try:
        index_stats = get_index_stats()
        
        return {
            "status": "healthy" if index_stats.get("loaded", False) else "degraded",
            "version": "1.0.0",
            "uptime_seconds": get_uptime_seconds(),
            "index": index_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "unhealthy",
            "version": "1.0.0",
            "uptime_seconds": get_uptime_seconds(),
            "error": str(e),
            "timestamp": time.time()
        }
