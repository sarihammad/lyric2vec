import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 10) -> float:
    """Calculate Recall@K metric."""
    if len(y_true) == 0:
        return 0.0
    
    # Get top-k predictions
    top_k_indices = np.argsort(y_scores)[-k:]
    
    # Check if any of the top-k are relevant
    relevant_in_top_k = np.any(y_true[top_k_indices])
    
    return float(relevant_in_top_k)


def mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    if len(y_true) == 0:
        return 0.0
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Find rank of first relevant item
    for rank, idx in enumerate(sorted_indices, 1):
        if y_true[idx]:
            return 1.0 / rank
    
    return 0.0


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate Average Precision (AP)."""
    if len(y_true) == 0:
        return 0.0
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Calculate precision at each relevant position
    precision_sum = 0.0
    relevant_count = 0
    
    for rank, idx in enumerate(sorted_indices, 1):
        if y_true[idx]:
            relevant_count += 1
            precision_sum += relevant_count / rank
    
    return precision_sum / np.sum(y_true) if np.sum(y_true) > 0 else 0.0


def calculate_retrieval_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    relevance_matrix: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """Calculate retrieval metrics for cross-modal search."""
    metrics = {}
    
    # Calculate similarities
    similarities = np.dot(query_embeddings, candidate_embeddings.T)
    
    # For each query
    recalls = {f"recall@{k}": [] for k in k_values}
    mrrs = []
    aps = []
    
    for i in range(len(query_embeddings)):
        query_similarities = similarities[i]
        query_relevance = relevance_matrix[i]
        
        # Calculate metrics for this query
        for k in k_values:
            recalls[f"recall@{k}"].append(recall_at_k(query_relevance, query_similarities, k))
        
        mrrs.append(mean_reciprocal_rank(query_relevance, query_similarities))
        aps.append(average_precision(query_relevance, query_similarities))
    
    # Average across queries
    for k in k_values:
        metrics[f"recall@{k}"] = np.mean(recalls[f"recall@{k}"])
    
    metrics["mrr"] = np.mean(mrrs)
    metrics["map"] = np.mean(aps)
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
) -> Dict[str, Any]:
    """Calculate classification metrics."""
    metrics = {}
    
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels
    )
    
    if labels:
        for i, label in enumerate(labels):
            metrics[f"{label}_precision"] = precision[i]
            metrics[f"{label}_recall"] = recall[i]
            metrics[f"{label}_f1"] = f1[i]
            metrics[f"{label}_support"] = support[i]
    
    # Macro averages
    metrics["macro_precision"] = np.mean(precision)
    metrics["macro_recall"] = np.mean(recall)
    metrics["macro_f1"] = np.mean(f1)
    
    return metrics


def calculate_embedding_quality_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """Calculate embedding quality metrics."""
    metrics = {}
    
    # L2 norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    metrics["mean_norm"] = float(np.mean(norms))
    metrics["std_norm"] = float(np.std(norms))
    metrics["min_norm"] = float(np.min(norms))
    metrics["max_norm"] = float(np.max(norms))
    
    # Centroid distance
    centroid = np.mean(embeddings, axis=0)
    centroid_distances = np.linalg.norm(embeddings - centroid, axis=1)
    metrics["mean_centroid_distance"] = float(np.mean(centroid_distances))
    metrics["std_centroid_distance"] = float(np.std(centroid_distances))
    
    # Pairwise cosine similarities
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine_similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Remove diagonal (self-similarities)
    mask = ~np.eye(cosine_similarities.shape[0], dtype=bool)
    off_diagonal_similarities = cosine_similarities[mask]
    
    metrics["mean_cosine_similarity"] = float(np.mean(off_diagonal_similarities))
    metrics["std_cosine_similarity"] = float(np.std(off_diagonal_similarities))
    
    return metrics
