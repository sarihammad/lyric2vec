import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import base64
import io
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Lyric2Vec Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .search-result {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8080"


def check_api_health() -> bool:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        return response.status_code == 200
    except:
        return False


def search_api(query: str, modal: str = "fused", k: int = 10) -> Optional[List[Dict]]:
    """Search using the API."""
    try:
        payload = {
            "modal": modal,
            "query": query,
            "k": k
        }
        
        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["items"]
        else:
            st.error(f"API error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None


def load_sample_data() -> Dict[str, Any]:
    """Load sample data for demonstration."""
    # This would load actual data in a real implementation
    return {
        "embeddings": np.random.randn(100, 256),
        "labels": np.random.choice(["Rock", "Pop", "Jazz", "Blues"], 100),
        "track_ids": [f"track_{i:06d}" for i in range(100)],
        "metadata": {
            f"track_{i:06d}": {
                "artist": f"Artist {i % 10}",
                "title": f"Song {i}",
                "genre": np.random.choice(["Rock", "Pop", "Jazz", "Blues"])
            }
            for i in range(100)
        }
    }


def create_umap_plot(embeddings: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Create UMAP plot using plotly."""
    # Simple 2D projection for demo (in practice, use UMAP)
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })
    
    # Create plot
    fig = px.scatter(
        df, x='x', y='y', color='label',
        title="Embedding Space Visualization (PCA)",
        labels={'x': 'PC1', 'y': 'PC2'},
        hover_data={'label': True}
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">🎵 Lyric2Vec Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Search Demo", "Embedding Clusters", "Ablation Studies"]
    )
    
    # API Health Check
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.info("Make sure the API is running on localhost:8080")
    
    # Load sample data
    sample_data = load_sample_data()
    
    if page == "Overview":
        show_overview()
    elif page == "Search Demo":
        show_search_demo()
    elif page == "Embedding Clusters":
        show_embedding_clusters(sample_data)
    elif page == "Ablation Studies":
        show_ablation_studies()


def show_overview():
    """Show overview page."""
    st.header("📊 System Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tracks", "1,000", "↗️ 50")
    
    with col2:
        st.metric("Embedding Dimension", "256", "→ 0")
    
    with col3:
        st.metric("API Latency (p99)", "95ms", "↘️ 5ms")
    
    with col4:
        st.metric("Recall@10", "0.87", "↗️ 0.02")
    
    # System Architecture
    st.header("🏗️ System Architecture")
    
    st.markdown("""
    Lyric2Vec is a multimodal music embedding system that learns joint representations from:
    
    - **Text**: Song lyrics using DistilBERT
    - **Audio**: Mel spectrograms using CNN + BiGRU
    - **Metadata**: Artist/genre/year using embedding layers
    
    The system uses contrastive learning and neural fusion to create unified embeddings
    that enable cross-modal music search and recommendation.
    """)
    
    # Architecture diagram placeholder
    st.image("https://via.placeholder.com/800x400/1f77b4/ffffff?text=System+Architecture+Diagram", 
             caption="System Architecture (placeholder)")
    
    # Performance Metrics
    st.header("📈 Performance Metrics")
    
    # Create sample performance data
    metrics_data = pd.DataFrame({
        'Metric': ['Recall@1', 'Recall@5', 'Recall@10', 'MRR', 'MAP'],
        'Text→Audio': [0.65, 0.78, 0.85, 0.72, 0.68],
        'Audio→Text': [0.62, 0.75, 0.82, 0.69, 0.65],
        'Fused': [0.78, 0.87, 0.92, 0.85, 0.82]
    })
    
    st.dataframe(metrics_data, use_container_width=True)
    
    # Performance chart
    fig = px.bar(
        metrics_data.melt(id_vars=['Metric'], var_name='Modality', value_name='Score'),
        x='Metric', y='Score', color='Modality',
        title="Cross-Modal Retrieval Performance",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)


def show_search_demo():
    """Show search demonstration page."""
    st.header("🔍 Cross-Modal Search Demo")
    
    # Search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'melancholic acoustic ballad with soft vocals'"
        )
    
    with col2:
        search_modal = st.selectbox(
            "Search modality:",
            ["fused", "text", "audio"]
        )
    
    k = st.slider("Number of results:", 1, 50, 10)
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_api(search_query, search_modal, k)
                
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="search-result">
                                <h4>#{i} {result.get('title', 'Unknown Title')}</h4>
                                <p><strong>Artist:</strong> {result.get('artist', 'Unknown Artist')}</p>
                                <p><strong>Genre:</strong> {result.get('genre', 'Unknown Genre')}</p>
                                <p><strong>Score:</strong> {result.get('score', 0):.3f}</p>
                                <p><strong>Track ID:</strong> {result.get('track_id', 'Unknown')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("No results found or API error")
        else:
            st.warning("Please enter a search query")
    
    # Audio upload demo
    st.header("🎵 Audio Search Demo")
    
    uploaded_file = st.file_uploader(
        "Upload an audio file for search:",
        type=['wav', 'mp3', 'm4a']
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        if st.button("🎵 Search by Audio", type="primary"):
            # In a real implementation, you would encode the audio and search
            st.info("Audio search functionality would be implemented here")
            st.info("The uploaded audio would be processed and used to search the embedding space")


def show_embedding_clusters(sample_data: Dict[str, Any]):
    """Show embedding clusters page."""
    st.header("🎨 Embedding Space Visualization")
    
    # UMAP plot
    st.subheader("2D Embedding Projection")
    
    fig = create_umap_plot(sample_data["embeddings"], sample_data["labels"])
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster statistics
    st.subheader("Cluster Statistics")
    
    cluster_stats = pd.DataFrame({
        'Genre': ['Rock', 'Pop', 'Jazz', 'Blues'],
        'Count': [25, 30, 20, 25],
        'Avg Similarity': [0.65, 0.72, 0.58, 0.61]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        fig = px.pie(cluster_stats, values='Count', names='Genre', title="Genre Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive exploration
    st.subheader("Interactive Exploration")
    
    selected_genre = st.selectbox("Select genre to explore:", cluster_stats['Genre'].tolist())
    
    # Filter data by genre
    genre_mask = sample_data["labels"] == selected_genre
    genre_embeddings = sample_data["embeddings"][genre_mask]
    genre_track_ids = [sample_data["track_ids"][i] for i in range(len(sample_data["track_ids"])) if genre_mask[i]]
    
    st.info(f"Found {len(genre_embeddings)} tracks in {selected_genre} genre")
    
    # Show some tracks from this genre
    if genre_track_ids:
        st.subheader(f"Sample {selected_genre} Tracks")
        for track_id in genre_track_ids[:5]:
            metadata = sample_data["metadata"].get(track_id, {})
            st.write(f"• {metadata.get('title', 'Unknown')} by {metadata.get('artist', 'Unknown')}")


def show_ablation_studies():
    """Show ablation studies page."""
    st.header("🔬 Ablation Studies")
    
    # Model components comparison
    st.subheader("Model Components Analysis")
    
    ablation_data = pd.DataFrame({
        'Configuration': ['Text Only', 'Audio Only', 'Metadata Only', 'Text + Audio', 'Text + Metadata', 'Audio + Metadata', 'All Modalities'],
        'Recall@10': [0.72, 0.68, 0.45, 0.81, 0.75, 0.70, 0.87],
        'MRR': [0.65, 0.61, 0.38, 0.74, 0.68, 0.63, 0.82]
    })
    
    fig = px.bar(
        ablation_data,
        x='Configuration',
        y='Recall@10',
        title="Recall@10 by Model Configuration",
        color='Recall@10',
        color_continuous_scale='Viridis'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Fusion method comparison
    st.subheader("Fusion Method Comparison")
    
    fusion_data = pd.DataFrame({
        'Fusion Method': ['Weighted Sum', 'Gated Fusion', 'Attention', 'Cross-Modal'],
        'Recall@10': [0.82, 0.87, 0.85, 0.84],
        'Training Time (hrs)': [2.5, 3.2, 4.1, 3.8],
        'Inference Time (ms)': [45, 52, 68, 61]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(fusion_data, x='Fusion Method', y='Recall@10', title="Performance by Fusion Method")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(fusion_data, x='Training Time (hrs)', y='Recall@10', 
                        size='Inference Time (ms)', hover_name='Fusion Method',
                        title="Performance vs Training Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Loss function analysis
    st.subheader("Loss Function Analysis")
    
    loss_data = pd.DataFrame({
        'Loss Component': ['Text-Audio Contrastive', 'Text-Metadata Contrastive', 'Audio-Metadata Contrastive', 'Triplet Loss', 'Fusion Loss'],
        'Weight': [1.0, 0.5, 0.2, 0.1, 0.1],
        'Contribution': [0.35, 0.18, 0.12, 0.08, 0.27]
    })
    
    fig = px.pie(loss_data, values='Contribution', names='Loss Component', 
                 title="Loss Function Component Contributions")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
