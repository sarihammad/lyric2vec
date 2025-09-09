import os
import argparse
import logging
import numpy as np
import faiss
from typing import Optional

from ..utils.io import ensure_dir

logger = logging.getLogger(__name__)


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "HNSW",
    m: int = 32,
    ef_search: int = 128,
    ef_construction: int = 200
) -> faiss.Index:
    """Build FAISS index from embeddings."""
    logger.info(f"Building FAISS index: {index_type}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Ensure embeddings are float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create index based on type
    if index_type == "HNSW":
        # Hierarchical Navigable Small World
        index = faiss.IndexHNSWFlat(embeddings.shape[1], m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        
    elif index_type == "IVF":
        # Inverted File
        nlist = min(4096, embeddings.shape[0] // 100)  # Number of clusters
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product quantizer
        index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
        
    elif index_type == "Flat":
        # Flat index (exact search)
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Train index if needed
    if hasattr(index, 'is_trained') and not index.is_trained:
        logger.info("Training FAISS index")
        index.train(embeddings)
    
    # Add embeddings to index
    logger.info("Adding embeddings to index")
    index.add(embeddings)
    
    logger.info(f"FAISS index built successfully: {index.ntotal} vectors")
    return index


def save_faiss_index(index: faiss.Index, output_path: str):
    """Save FAISS index to file."""
    ensure_dir(os.path.dirname(output_path))
    faiss.write_index(index, output_path)
    logger.info(f"FAISS index saved to: {output_path}")


def load_faiss_index(index_path: str) -> faiss.Index:
    """Load FAISS index from file."""
    index = faiss.read_index(index_path)
    logger.info(f"FAISS index loaded from: {index_path} ({index.ntotal} vectors)")
    return index


def search_index(
    index: faiss.Index,
    query: np.ndarray,
    k: int = 10
) -> tuple:
    """Search FAISS index."""
    # Ensure query is float32 and normalized
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    
    # Normalize query
    faiss.normalize_L2(query.reshape(1, -1))
    
    # Search
    scores, indices = index.search(query.reshape(1, -1), k)
    
    return scores[0], indices[0]


def benchmark_index(
    index: faiss.Index,
    test_queries: np.ndarray,
    k: int = 10,
    num_queries: int = 100
) -> dict:
    """Benchmark FAISS index performance."""
    import time
    
    # Sample test queries
    if len(test_queries) > num_queries:
        indices = np.random.choice(len(test_queries), num_queries, replace=False)
        test_queries = test_queries[indices]
    
    # Benchmark search
    start_time = time.time()
    
    for query in test_queries:
        search_index(index, query, k)
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_query = total_time / len(test_queries)
    queries_per_second = len(test_queries) / total_time
    
    results = {
        "total_queries": len(test_queries),
        "total_time_seconds": total_time,
        "avg_time_per_query_ms": avg_time_per_query * 1000,
        "queries_per_second": queries_per_second,
        "index_size": index.ntotal,
        "index_type": type(index).__name__
    }
    
    logger.info(f"Benchmark results: {results}")
    return results


def main():
    """Main function for building FAISS index."""
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--input", required=True, help="Input embeddings file (.npy)")
    parser.add_argument("--output", required=True, help="Output index file (.index)")
    parser.add_argument("--type", choices=["HNSW", "IVF", "Flat"], default="HNSW", 
                       help="FAISS index type")
    parser.add_argument("--m", type=int, default=32, help="HNSW parameter M")
    parser.add_argument("--ef-search", type=int, default=128, help="HNSW efSearch parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction parameter")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after building")
    parser.add_argument("--benchmark-queries", type=int, default=100, help="Number of benchmark queries")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.input}")
    embeddings = np.load(args.input)
    logger.info(f"Loaded embeddings: {embeddings.shape}")
    
    # Build index
    index = build_faiss_index(
        embeddings,
        index_type=args.type,
        m=args.m,
        ef_search=args.ef_search,
        ef_construction=args.ef_construction
    )
    
    # Save index
    save_faiss_index(index, args.output)
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("Running benchmark")
        benchmark_results = benchmark_index(
            index,
            embeddings,
            k=10,
            num_queries=args.benchmark_queries
        )
        
        # Save benchmark results
        benchmark_path = args.output.replace('.index', '_benchmark.json')
        import json
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Benchmark results saved to: {benchmark_path}")
    
    logger.info("FAISS index building completed successfully")


if __name__ == "__main__":
    main()
