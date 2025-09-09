import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import pandas as pd

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

from ..utils.io import ensure_dir

logger = logging.getLogger(__name__)


def reduce_dimensions_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42
) -> np.ndarray:
    """Reduce embedding dimensions using UMAP."""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    logger.info(f"Reducing dimensions with UMAP: {embeddings.shape} -> {n_components}D")
    
    # Create UMAP reducer
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    # Fit and transform
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    logger.info(f"UMAP reduction completed: {reduced_embeddings.shape}")
    return reduced_embeddings


def reduce_dimensions_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """Reduce embedding dimensions using PCA."""
    from sklearn.decomposition import PCA
    
    logger.info(f"Reducing dimensions with PCA: {embeddings.shape} -> {n_components}D")
    
    # Create PCA reducer
    reducer = PCA(n_components=n_components, random_state=random_state)
    
    # Fit and transform
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    logger.info(f"PCA reduction completed: {reduced_embeddings.shape}")
    logger.info(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
    
    return reduced_embeddings


def create_umap_plot(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP Visualization",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    use_umap: bool = True
) -> plt.Figure:
    """Create UMAP plot of embeddings."""
    # Reduce dimensions
    if use_umap and UMAP_AVAILABLE:
        reduced_embeddings = reduce_dimensions_umap(embeddings)
        method = "UMAP"
    else:
        reduced_embeddings = reduce_dimensions_pca(embeddings)
        method = "PCA"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    if labels is not None:
        # Colored by labels
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[i]],
                label=str(label),
                alpha=0.7,
                s=20
            )
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Single color
        ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            alpha=0.7,
            s=20
        )
    
    # Customize plot
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Save if path provided
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {output_path}")
    
    return fig


def create_multimodal_plot(
    text_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    fused_embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_dir: str = "plots",
    use_umap: bool = True
):
    """Create UMAP plots for all modalities."""
    ensure_dir(output_dir)
    
    # Create plots for each modality
    modalities = {
        "Text": text_embeddings,
        "Audio": audio_embeddings,
        "Fused": fused_embeddings
    }
    
    for modality_name, embeddings in modalities.items():
        logger.info(f"Creating {modality_name} plot")
        
        output_path = os.path.join(output_dir, f"{modality_name.lower()}_umap.png")
        
        fig = create_umap_plot(
            embeddings,
            labels=labels,
            title=f"{modality_name} Embeddings - {method} Visualization",
            output_path=output_path,
            use_umap=use_umap
        )
        
        plt.close(fig)
    
    # Create combined plot
    logger.info("Creating combined plot")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (modality_name, embeddings) in enumerate(modalities.items()):
        # Reduce dimensions
        if use_umap and UMAP_AVAILABLE:
            reduced_embeddings = reduce_dimensions_umap(embeddings)
            method = "UMAP"
        else:
            reduced_embeddings = reduce_dimensions_pca(embeddings)
            method = "PCA"
        
        # Plot
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for j, label in enumerate(unique_labels):
                mask = labels == label
                axes[i].scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=[colors[j]],
                    label=str(label),
                    alpha=0.7,
                    s=20
                )
        else:
            axes[i].scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=0.7,
                s=20
            )
        
        axes[i].set_title(f"{modality_name} Embeddings")
        axes[i].set_xlabel(f"{method} Component 1")
        axes[i].set_ylabel(f"{method} Component 2")
        axes[i].grid(True, alpha=0.3)
    
    # Save combined plot
    combined_path = os.path.join(output_dir, "combined_umap.png")
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Combined plot saved to: {combined_path}")


def save_umap_data(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    track_ids: Optional[np.ndarray] = None,
    output_path: str = "umap_data.csv"
):
    """Save UMAP data to CSV for further analysis."""
    # Reduce dimensions
    if UMAP_AVAILABLE:
        reduced_embeddings = reduce_dimensions_umap(embeddings)
        method = "UMAP"
    else:
        reduced_embeddings = reduce_dimensions_pca(embeddings)
        method = "PCA"
    
    # Create DataFrame
    data = {
        f"{method}_1": reduced_embeddings[:, 0],
        f"{method}_2": reduced_embeddings[:, 1]
    }
    
    if labels is not None:
        data["label"] = labels
    
    if track_ids is not None:
        data["track_id"] = track_ids
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    logger.info(f"UMAP data saved to: {output_path}")
    
    return df


def main():
    """Main function for UMAP visualization."""
    parser = argparse.ArgumentParser(description="Create UMAP visualization of embeddings")
    parser.add_argument("--embeddings", required=True, help="Embeddings file (.npy or .npz)")
    parser.add_argument("--labels", help="Labels file (.npy) for coloring")
    parser.add_argument("--track-ids", help="Track IDs file (.npy)")
    parser.add_argument("--output-dir", default="plots", help="Output directory")
    parser.add_argument("--title", default="Embedding Visualization", help="Plot title")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA instead of UMAP")
    parser.add_argument("--save-data", action="store_true", help="Save UMAP data to CSV")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings}")
    if args.embeddings.endswith('.npz'):
        data = np.load(args.embeddings)
        if 'fused' in data:
            embeddings = data['fused']
        elif 'text' in data:
            embeddings = data['text']
        else:
            # Use first array
            embeddings = data[data.files[0]]
    else:
        embeddings = np.load(args.embeddings)
    
    logger.info(f"Loaded embeddings: {embeddings.shape}")
    
    # Load labels if provided
    labels = None
    if args.labels:
        labels = np.load(args.labels)
        logger.info(f"Loaded labels: {labels.shape}")
    
    # Load track IDs if provided
    track_ids = None
    if args.track_ids:
        track_ids = np.load(args.track_ids)
        logger.info(f"Loaded track IDs: {track_ids.shape}")
    
    # Create plots
    use_umap = not args.use_pca
    
    if embeddings.ndim == 3 and embeddings.shape[0] == 3:
        # Multi-modal embeddings (text, audio, fused)
        logger.info("Detected multi-modal embeddings")
        create_multimodal_plot(
            embeddings[0],  # text
            embeddings[1],  # audio
            embeddings[2],  # fused
            labels=labels,
            output_dir=args.output_dir,
            use_umap=use_umap
        )
    else:
        # Single modality
        logger.info("Creating single modality plot")
        output_path = os.path.join(args.output_dir, "umap_plot.png")
        
        fig = create_umap_plot(
            embeddings,
            labels=labels,
            title=args.title,
            output_path=output_path,
            use_umap=use_umap
        )
        
        plt.close(fig)
    
    # Save data if requested
    if args.save_data:
        csv_path = os.path.join(args.output_dir, "umap_data.csv")
        save_umap_data(embeddings, labels, track_ids, csv_path)
    
    logger.info("UMAP visualization completed")


if __name__ == "__main__":
    main()
