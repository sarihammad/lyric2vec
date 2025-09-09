import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'route', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'route', 'status_code'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ENCODE_LATENCY = Histogram(
    'encode_latency_ms',
    'Encoding latency in milliseconds',
    ['modal'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

FAISS_LATENCY = Histogram(
    'faiss_latency_ms',
    'FAISS search latency in milliseconds',
    ['modal'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

INDEX_LOADED = Gauge(
    'index_loaded',
    'Whether the FAISS index is loaded (1) or not (0)'
)

# Global state
start_time = time.time()


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get route template for consistent labeling
        route = request.url.path
        if hasattr(request, 'route') and request.route:
            route = request.route.path
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        status_code = str(response.status_code)
        method = request.method
        
        REQUEST_COUNT.labels(
            method=method,
            route=route,
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            route=route,
            status_code=status_code
        ).observe(duration)
        
        return response


def record_encode_latency(modal: str, duration_ms: float):
    """Record encoding latency for a specific modality."""
    ENCODE_LATENCY.labels(modal=modal).observe(duration_ms)


def record_faiss_latency(modal: str, duration_ms: float):
    """Record FAISS search latency for a specific modality."""
    FAISS_LATENCY.labels(modal=modal).observe(duration_ms)


def set_index_loaded(loaded: bool):
    """Set the index loaded status."""
    INDEX_LOADED.set(1 if loaded else 0)


def get_uptime_seconds() -> float:
    """Get service uptime in seconds."""
    return time.time() - start_time


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    return generate_latest().decode('utf-8')
