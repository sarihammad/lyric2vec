import os
import logging
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Get environment variable with optional default and required validation."""
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not set")
    
    return value


def get_model_config() -> dict:
    """Get model configuration from environment variables."""
    return {
        "hf_text_model": get_env_var("HF_TEXT_MODEL", "distilbert-base-uncased"),
        "audio_sr": int(get_env_var("AUDIO_SR", "16000")),
        "emb_d": int(get_env_var("EMB_D", "256")),
        "loss_weights": [float(x) for x in get_env_var("LOSS_WEIGHTS", "1.0,0.5,0.2").split(",")],
    }


def get_faiss_config() -> dict:
    """Get FAISS configuration from environment variables."""
    return {
        "faiss_type": get_env_var("FAISS_TYPE", "HNSW"),
        "faiss_m": int(get_env_var("FAISS_M", "32")),
        "faiss_ef_search": int(get_env_var("FAISS_EF_SEARCH", "128")),
    }


def get_api_config() -> dict:
    """Get API configuration from environment variables."""
    return {
        "port": int(get_env_var("API_PORT", "8080")),
        "host": get_env_var("API_HOST", "0.0.0.0"),
        "prom_path": get_env_var("PROM_PATH", "/metrics"),
    }


def get_data_paths() -> dict:
    """Get data paths from environment variables."""
    return {
        "raw_path": get_env_var("DATA_RAW_PATH", "data/raw"),
        "processed_path": get_env_var("DATA_PROCESSED_PATH", "data/processed"),
        "artifacts_path": get_env_var("DATA_ARTIFACTS_PATH", "data/artifacts"),
    }


def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "HF_TEXT_MODEL",
        "AUDIO_SR",
        "EMB_D",
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("Environment validation passed")


def get_log_level() -> str:
    """Get logging level from environment."""
    return get_env_var("LOG_LEVEL", "INFO").upper()
