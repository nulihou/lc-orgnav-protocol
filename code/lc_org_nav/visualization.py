"""
Visualization tools for LC-ORGNav.

Includes:
1. Training curves plotting
2. Attention weight visualization
3. Scene graph visualization
4. Confusion matrix plotting
5. Error distribution analysis
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Graph visualization will be limited.")


class TrainingVisualizer:
    """Visualize training progress and metrics."""
    
    def __init__(self, save_dir: str = "outputs/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_metrics: Dict[str, List[float]],
        save_name: str = "training_curves.png"
    ):
        """Plot training loss and validation metrics over epochs.
        
        Args:
            train_losses: List of training losses per epoch
            val_metrics: Dict of metric_name -> list of values per epoch
            save_name: Output filename
        """
        num_metrics = len(val_metrics) + 1
        fig, axes = plt.subplots(1, min(num_metrics, 3), figsize=(15, 4))
        
        if num_metrics == 1:
            axes = [axes]
        
        # Plot training loss
        ax = axes[0]
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot validation metrics
        for idx, (metric_name, values) in enumerate(list(val_metrics.items())[:2]):
            if idx + 1 >= len(axes):
                break
            ax = axes[idx + 1]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'r-', linewidth=2, label=metric_name)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Validation {metric_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = "confusion_matrix.png"
    ):
        """Plot confusion matrix for relation classification.
        
        Args:
            confusion_matrix: (num_classes, num_classes) confusion matrix
            class_names: Optional list of class names
            save_name: Output filename
        """
        num_classes = confusion_matrix.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Normalize by row (true labels)
        cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Raw counts
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax2, cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_error_distributions(
        self,
        dist_errors: List[float],
        angle_errors: List[float],
        save_name: str = "error_distributions.png"
    ):
        """Plot error distributions for geometric predictions.
        
        Args:
            dist_errors: List of distance errors
            angle_errors: List of angle errors (in radians)
            save_name: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distance error histogram
        ax = axes[0, 0]
        ax.hist(dist_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Distance Error (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distance Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distance error box plot
        ax = axes[0, 1]
        bp = ax.boxplot([dist_errors], vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Distance Error (m)', fontsize=12)
        ax.set_title('Distance Error Box Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Angle error histogram (convert to degrees)
        ax = axes[1, 0]
        angle_errors_deg = [e * 180 / np.pi for e in angle_errors]
        ax.hist(angle_errors_deg, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Angle Error (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Angle Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Angle error box plot
        ax = axes[1, 1]
        bp = ax.boxplot([angle_errors_deg], vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightyellow')
        ax.set_ylabel('Angle Error (degrees)', fontsize=12)
        ax.set_title('Angle Error Box Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error distributions saved to {save_path}")
    
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        node_labels: Optional[List[str]] = None,
        save_name: str = "attention_weights.png"
    ):
        """Plot attention weight heatmap.
        
        Args:
            attention_weights: (N, N) or (num_heads, N, N) attention weights
            node_labels: Optional list of node labels
            save_name: Output filename
        """
        if attention_weights.ndim == 3:
            # Average over heads
            attention_weights = attention_weights.mean(axis=0)
        
        N = attention_weights.shape[0]
        if node_labels is None:
            node_labels = [f"N{i}" for i in range(N)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(attention_weights, annot=False, cmap='viridis',
                   xticklabels=node_labels, yticklabels=node_labels,
                   ax=ax, cbar_kws={'label': 'Attention Weight'})
        
        ax.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Nodes', fontsize=12)
        ax.set_ylabel('Query Nodes', fontsize=12)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention weights saved to {save_path}")


class SceneGraphVisualizer:
    """Visualize scene graphs with predictions."""
    
    def __init__(self, save_dir: str = "outputs/visualizations/graphs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_NETWORKX:
            print("Warning: networkx not available. Graph visualization disabled.")
    
    def visualize_scene_graph(
        self,
        node_feats: np.ndarray,
        edge_index: np.ndarray,
        candidate_mask: np.ndarray,
        target_index: int,
        predicted_index: Optional[int] = None,
        goal_scores: Optional[np.ndarray] = None,
        save_name: str = "scene_graph.png"
    ):
        """Visualize a scene graph with predictions.
        
        Args:
            node_feats: (N, F) node features
            edge_index: (2, E) edge indices
            candidate_mask: (N,) boolean mask for candidates
            target_index: ground truth target index
            predicted_index: predicted target index
            goal_scores: (N,) goal selection scores
            save_name: Output filename
        """
        if not HAS_NETWORKX:
            print("Skipping graph visualization (networkx not available)")
            return
        
        N = node_feats.shape[0]
        E = edge_index.shape[1]
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        
        # Add edges
        for i in range(E):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Node colors based on status
        node_colors = []
        node_sizes = []
        for i in range(N):
            if i == target_index:
                node_colors.append('green')  # Ground truth
                node_sizes.append(1000)
            elif predicted_index is not None and i == predicted_index:
                if i == target_index:
                    node_colors.append('green')
                else:
                    node_colors.append('red')  # Wrong prediction
                node_sizes.append(1000)
            elif candidate_mask[i]:
                node_colors.append('lightblue')  # Candidate
                node_sizes.append(600)
            else:
                node_colors.append('lightgray')  # Non-candidate
                node_sizes.append(400)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              alpha=0.3, arrows=True, arrowsize=10,
                              connectionstyle='arc3,rad=0.1')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8)
        
        # Draw labels
        labels = {i: f"{i}" for i in range(N)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')
        
        # Add goal scores as text if available
        if goal_scores is not None:
            for i, (x, y) in pos.items():
                if candidate_mask[i]:
                    ax.text(x, y - 0.15, f"{goal_scores[i]:.2f}", 
                           ha='center', va='top', fontsize=8, color='blue')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='green', label='Ground Truth Target'),
            mpatches.Patch(color='lightblue', label='Candidate'),
            mpatches.Patch(color='lightgray', label='Non-Candidate'),
        ]
        if predicted_index is not None and predicted_index != target_index:
            legend_elements.insert(1, mpatches.Patch(color='red', label='Wrong Prediction'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_title('Scene Graph Visualization', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scene graph saved to {save_path}")
    
    def visualize_top_k_predictions(
        self,
        node_feats: np.ndarray,
        candidate_mask: np.ndarray,
        target_index: int,
        goal_scores: np.ndarray,
        k: int = 5,
        save_name: str = "top_k_predictions.png"
    ):
        """Visualize top-k predictions.
        
        Args:
            node_feats: (N, F) node features
            candidate_mask: (N,) boolean mask
            target_index: ground truth index
            goal_scores: (N,) goal scores
            k: number of top predictions to show
            save_name: Output filename
        """
        # Get candidate indices and scores
        cand_indices = np.where(candidate_mask)[0]
        cand_scores = goal_scores[cand_indices]
        
        # Sort by score
        sorted_idx = np.argsort(cand_scores)[::-1]
        top_k_local = sorted_idx[:k]
        top_k_global = cand_indices[top_k_local]
        top_k_scores = cand_scores[top_k_local]
        
        # Find rank of target
        target_local = np.where(cand_indices == target_index)[0]
        if len(target_local) > 0:
            target_rank = np.where(sorted_idx == target_local[0])[0][0] + 1
        else:
            target_rank = -1
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if idx == target_index else 'skyblue' for idx in top_k_global]
        bars = ax.barh(range(k), top_k_scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Labels
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"Node {idx} (Rank {i+1})" for i, idx in enumerate(top_k_global)])
        ax.set_xlabel('Goal Score', fontsize=12)
        ax.set_title(f'Top-{k} Predictions (Target: Node {target_index}, Rank: {target_rank})', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, top_k_scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top-k predictions saved to {save_path}")

