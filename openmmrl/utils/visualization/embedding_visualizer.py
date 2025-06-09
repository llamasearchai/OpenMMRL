import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from openmmrl.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingVisualizer:
    """
    Visualize embeddings using dimensionality reduction techniques.
    
    Supports:
    - PCA
    - t-SNE
    - UMAP
    """
    
    def __init__(
        self,
        output_dir: str = "visualizations",
        n_components: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize the embedding visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            n_components: Number of components for dimensionality reduction
            random_state: Random state for reproducibility
        """
        self.output_dir = output_dir
        self.n_components = n_components
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_embeddings(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        metadata: Optional[Dict] = None,
        method: str = "umap",
        title: str = "Embedding Visualization",
        filename: str = "embeddings",
        cmap: str = "viridis",
        alpha: float = 0.7,
        figsize: Tuple[int, int] = (12, 10),
    ) -> np.ndarray:
        """
        Visualize embeddings using a dimensionality reduction technique.
        
        Args:
            embeddings: Embeddings to visualize
            labels: Labels for coloring the points
            metadata: Additional metadata for the visualization
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            title: Title for the visualization
            filename: Filename for saving the visualization
            cmap: Colormap for the visualization
            alpha: Alpha value for the points
            figsize: Figure size
        
        Returns:
            Reduced embeddings
        """
        # Convert embeddings to numpy array
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Convert labels to numpy array if provided
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            elif isinstance(labels, list):
                labels = np.array(labels)
        
        # Perform dimensionality reduction
        reduced_embeddings = self._reduce_dimensions(embeddings, method)
        
        # Plot embeddings
        self._plot_embeddings(
            reduced_embeddings,
            labels,
            metadata,
            method,
            title,
            filename,
            cmap,
            alpha,
            figsize,
        )
        
        return reduced_embeddings
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
    ) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings.
        
        Args:
            embeddings: Embeddings to reduce
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        
        Returns:
            Reduced embeddings
        """
        if method.lower() == "pca":
            reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        elif method.lower() == "tsne":
            reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                n_jobs=-1,
                perplexity=min(30, embeddings.shape[0] - 1),
            )
        elif method.lower() == "umap":
            reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.1,
            )
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        # Perform dimensionality reduction
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        return reduced_embeddings
    
    def _plot_embeddings(
        self,
        reduced_embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        method: str = "umap",
        title: str = "Embedding Visualization",
        filename: str = "embeddings",
        cmap: str = "viridis",
        alpha: float = 0.7,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """
        Plot reduced embeddings.
        
        Args:
            reduced_embeddings: Reduced embeddings to plot
            labels: Labels for coloring the points
            metadata: Additional metadata for the visualization
            method: Dimensionality reduction method used
            title: Title for the visualization
            filename: Filename for saving the visualization
            cmap: Colormap for the visualization
            alpha: Alpha value for the points
            figsize: Figure size
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot embeddings
        if labels is not None:
            # Convert categorical labels to numeric
            if not np.issubdtype(labels.dtype, np.number):
                unique_labels = np.unique(labels)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_map[label] for label in labels])
            else:
                numeric_labels = labels
                unique_labels = np.unique(numeric_labels)
            
            # Create scatter plot with labels
            scatter = plt.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=numeric_labels,
                cmap=cmap,
                alpha=alpha,
                s=100,
            )
            
            # Add legend
            if len(unique_labels) <= 20:  # Only add legend if not too many labels
                plt.colorbar(scatter, label="Label")
                
                # Add legend if labels are categorical
                if not np.issubdtype(labels.dtype, np.number):
                    from matplotlib.lines import Line2D
                    
                    # Create legend elements
                    legend_elements = [
                        Line2D(
                            [0], [0],
                            marker="o",
                            color="w",
                            markerfacecolor=plt.get_cmap(cmap)(i / len(unique_labels)),
                            markersize=10,
                            label=label,
                        )
                        for i, label in enumerate(unique_labels)
                    ]
                    
                    # Add legend
                    plt.legend(handles=legend_elements, title="Labels", loc="best")
        else:
            # Create scatter plot without labels
            plt.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=alpha,
                s=100,
            )
        
        # Add annotations if metadata is provided
        if metadata is not None and "annotations" in metadata:
            for i, annotation in enumerate(metadata["annotations"]):
                plt.annotate(
                    annotation,
                    (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                    fontsize=8,
                    alpha=0.7,
                )
        
        # Add title and labels
        plt.title(f"{title} ({method.upper()})")
        plt.xlabel(f"Component 1")
        plt.ylabel(f"Component 2")
        
        # Add grid
        plt.grid(alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}_{method}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}_{method}.pdf"),
            bbox_inches="tight",
        )
        
        plt.close()
    
    def visualize_attention(
        self,
        attention_weights: Union[np.ndarray, torch.Tensor],
        tokens: Optional[List[str]] = None,
        layer: int = 0,
        head: int = 0,
        title: str = "Attention Visualization",
        filename: str = "attention",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """
        Visualize attention weights.
        
        Args:
            attention_weights: Attention weights to visualize
            tokens: Tokens for axis labels
            layer: Layer index
            head: Attention head index
            title: Title for the visualization
            filename: Filename for saving the visualization
            cmap: Colormap for the visualization
            figsize: Figure size
        """
        # Convert attention weights to numpy array
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Extract attention weights for the specified layer and head
        if attention_weights.ndim == 4:  # [batch, layer, head, seq_len, seq_len]
            attention_weights = attention_weights[0, layer, head]
        elif attention_weights.ndim == 5:  # [batch, layer, head, seq_len, seq_len]
            attention_weights = attention_weights[0, layer, head]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot attention heatmap
        plt.imshow(attention_weights, cmap=cmap)
        
        # Add colorbar
        plt.colorbar(label="Attention Weight")
        
        # Add tokens if provided
        if tokens is not None:
            # Limit the number of tokens to display
            max_tokens = 50
            if len(tokens) > max_tokens:
                # Subsample tokens
                indices = np.linspace(0, len(tokens) - 1, max_tokens, dtype=int)
                tokens = [tokens[i] for i in indices]
                attention_weights = attention_weights[indices, :][:, indices]
            
            plt.xticks(
                range(len(tokens)),
                tokens,
                rotation=90,
                fontsize=8,
            )
            plt.yticks(
                range(len(tokens)),
                tokens,
                fontsize=8,
            )
        
        # Add title
        plt.title(f"{title} (Layer {layer}, Head {head})")
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}_layer{layer}_head{head}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}_layer{layer}_head{head}.pdf"),
            bbox_inches="tight",
        )
        
        plt.close()
    
    def visualize_modality_interactions(
        self,
        cross_attention_weights: Dict[str, Union[np.ndarray, torch.Tensor]],
        modalities: List[str],
        layer: int = 0,
        title: str = "Modality Interactions",
        filename: str = "modality_interactions",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """
        Visualize cross-attention between modalities.
        
        Args:
            cross_attention_weights: Dictionary of cross-attention weights
            modalities: List of modalities
            layer: Layer index
            title: Title for the visualization
            filename: Filename for saving the visualization
            cmap: Colormap for the visualization
            figsize: Figure size
        """
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create modality interaction matrix
        num_modalities = len(modalities)
        interaction_matrix = np.zeros((num_modalities, num_modalities))
        
        # Fill interaction matrix
        for i, target in enumerate(modalities):
            for j, source in enumerate(modalities):
                key = f"{target}_to_{source}"
                if key in cross_attention_weights:
                    # Convert to numpy array
                    weights = cross_attention_weights[key]
                    if isinstance(weights, torch.Tensor):
                        weights = weights.detach().cpu().numpy()
                    
                    # Extract weights for the specified layer
                    if weights.ndim > 2:
                        weights = weights[0, layer] if weights.ndim == 3 else weights[0, layer, 0]
                    
                    # Compute average attention
                    interaction_matrix[i, j] = np.mean(weights)
        
        # Plot interaction heatmap
        plt.imshow(interaction_matrix, cmap=cmap)
        
        # Add colorbar
        plt.colorbar(label="Average Attention")
        
        # Add modality labels
        plt.xticks(range(num_modalities), modalities, rotation=45)
        plt.yticks(range(num_modalities), modalities)
        
        # Add title
        plt.title(f"{title} (Layer {layer})")
        
        # Add axis labels
        plt.xlabel("Source Modality")
        plt.ylabel("Target Modality")
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}_layer{layer}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}_layer{layer}.pdf"),
            bbox_inches="tight",
        )
        
        plt.close() 