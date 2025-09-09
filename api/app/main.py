import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .router_search import router as search_router
from .router_health import router as health_router
from .metrics import PrometheusMiddleware, set_index_loaded
from .index_faiss import load_faiss_index, warmup_search
from .deps import get_api_config, get_data_paths, validate_environment, get_log_level

# Configure logging
logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Lyric2Vec API")
    
    try:
        # Validate environment
        validate_environment()
        
        # Get data paths
        data_paths = get_data_paths()
        artifacts_path = data_paths["artifacts_path"]
        
        # Load FAISS index
        index_path = os.path.join(artifacts_path, "fused.index")
        ids_path = os.path.join(artifacts_path, "track_ids.npy")
        metadata_path = os.path.join(artifacts_path, "track_metadata.npy")
        
        if os.path.exists(index_path) and os.path.exists(ids_path):
            load_faiss_index(index_path, ids_path, metadata_path)
            set_index_loaded(True)
            
            # Warmup search
            warmup_search()
            
            logger.info("FAISS index loaded successfully")
        else:
            logger.warning(f"FAISS index not found at {index_path}")
            set_index_loaded(False)
        
        logger.info("API startup completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        set_index_loaded(False)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Lyric2Vec API")


# Create FastAPI app
app = FastAPI(
    title="Lyric2Vec API",
    description="Multimodal music embedding search API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # React/Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(search_router)
app.include_router(health_router)

# Mount metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    from .metrics import get_metrics
    return get_metrics()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Lyric2Vec API",
        "version": "1.0.0",
        "description": "Multimodal music embedding search API",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics"
    }


if __name__ == "__main__":
    import uvicorn
    
    config = get_api_config()
    uvicorn.run(
        "main:app",
        host=config["host"],
        port=config["port"],
        reload=True
    )
