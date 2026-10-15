"""
Data augmentation strategies for LC-ORGNav.

Includes:
1. Node feature perturbation
2. Edge dropout and addition
3. Graph structure augmentation
4. Hard negative mining for training
5. Synthetic scene generation
"""

import random
import math
from typing import Dict, Any, Tuple, List
import numpy as np
import torch


class GraphAugmentation:
    """Graph-level data augmentation."""
    
    def __init__(
        self,
        node_feat_noise: float = 0.1,
        edge_dropout_prob: float = 0.1,
        edge_add_prob: float = 0.05,
        node_dropout_prob: float = 0.0,
        enable: bool = True,
    ):
        self.node_feat_noise = node_feat_noise
        self.edge_dropout_prob = edge_dropout_prob
        self.edge_add_prob = edge_add_prob
        self.node_dropout_prob = node_dropout_prob
        self.enable = enable
    
    def augment_node_features(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to node features.
        
        Args:
            node_feats: (N, F) node features
            
        Returns:
            Augmented node features
        """
        if not self.enable or self.node_feat_noise == 0:
            return node_feats
        
        noise = torch.randn_like(node_feats) * self.node_feat_noise
        return node_feats + noise
    
    def augment_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment graph edges by dropout and random addition.
        
        Args:
            edge_index: (2, E) edge indices
            edge_attr: (E, F_edge) edge attributes
            num_nodes: number of nodes
            
        Returns:
            Augmented edge_index and edge_attr
        """
        if not self.enable:
            return edge_index, edge_attr
        
        E = edge_index.size(1)
        device = edge_index.device
        
        # Edge dropout
        if self.edge_dropout_prob > 0 and E > 0:
            keep_mask = torch.rand(E, device=device) > self.edge_dropout_prob
            edge_index = edge_index[:, keep_mask]
            edge_attr = edge_attr[keep_mask]
        
        # Edge addition (random edges with mean features)
        if self.edge_add_prob > 0 and num_nodes > 1:
            num_add = int(E * self.edge_add_prob)
            if num_add > 0:
                # Sample random node pairs
                src_nodes = torch.randint(0, num_nodes, (num_add,), device=device)
                dst_nodes = torch.randint(0, num_nodes, (num_add,), device=device)
                
                # Avoid self-loops
                valid_mask = src_nodes != dst_nodes
                src_nodes = src_nodes[valid_mask]
                dst_nodes = dst_nodes[valid_mask]
                
                if len(src_nodes) > 0:
                    new_edges = torch.stack([src_nodes, dst_nodes], dim=0)
                    
                    # Create random edge features (or use mean of existing)
                    if edge_attr.size(0) > 0:
                        mean_edge_attr = edge_attr.mean(dim=0, keepdim=True)
                        new_edge_attr = mean_edge_attr.repeat(len(src_nodes), 1)
                        new_edge_attr += torch.randn_like(new_edge_attr) * 0.1
                    else:
                        new_edge_attr = torch.randn(len(src_nodes), edge_attr.size(1), device=device)
                    
                    edge_index = torch.cat([edge_index, new_edges], dim=1)
                    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
        
        return edge_index, edge_attr
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation to a sample.
        
        Args:
            sample: Data sample dict
            
        Returns:
            Augmented sample
        """
        if not self.enable:
            return sample
        
        augmented = sample.copy()
        
        # Augment node features
        augmented["node_feats"] = self.augment_node_features(sample["node_feats"])
        
        # Augment edges
        num_nodes = sample["node_feats"].size(0)
        augmented["edge_index"], augmented["edge_attr"] = self.augment_edges(
            sample["edge_index"],
            sample["edge_attr"],
            num_nodes,
        )
        
        return augmented


class HardNegativeSampler:
    """Sample hard negative examples for contrastive learning."""
    
    def __init__(self, num_hard_negatives: int = 3):
        self.num_hard_negatives = num_hard_negatives
    
    def sample_hard_negatives(
        self,
        node_feats: torch.Tensor,
        candidate_mask: torch.Tensor,
        target_index: int,
        goal_logits: torch.Tensor,
    ) -> List[int]:
        """Sample hard negative nodes (high scoring but incorrect).
        
        Args:
            node_feats: (N, F) node features
            candidate_mask: (N,) boolean mask
            target_index: ground truth target
            goal_logits: (N,) current model predictions
            
        Returns:
            List of hard negative indices
        """
        # Get candidate indices
        cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        
        # Remove target from candidates
        neg_mask = candidate_mask.clone()
        neg_mask[target_index] = False
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        
        if len(neg_indices) == 0:
            return []
        
        # Get scores for negatives
        neg_logits = goal_logits[neg_indices]
        
        # Select top-k as hard negatives
        k = min(self.num_hard_negatives, len(neg_indices))
        _, top_k_local = torch.topk(neg_logits, k)
        hard_neg_indices = neg_indices[top_k_local].cpu().tolist()
        
        return hard_neg_indices


class SyntheticSceneGenerator:
    """Generate synthetic challenging scenes for data augmentation."""
    
    def __init__(
        self,
        node_feat_dim: int = 32,
        edge_feat_dim: int = 8,
        lang_feat_dim: int = 32,
        num_rel_types: int = 5,
    ):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.num_rel_types = num_rel_types
    
    def generate_challenging_scene(
        self,
        num_nodes: int = 15,
        num_candidates: int = 5,
        difficulty: str = "medium",
    ) -> Dict[str, Any]:
        """Generate a synthetic challenging scene.
        
        Args:
            num_nodes: number of nodes in the graph
            num_candidates: number of candidate nodes
            difficulty: "easy", "medium", or "hard"
            
        Returns:
            Synthetic sample dict
        """
        # Generate node features
        node_feats = np.random.randn(num_nodes, self.node_feat_dim).astype(np.float32)
        
        # Generate edges (fully connected or sparse based on difficulty)
        if difficulty == "easy":
            # Fully connected
            edge_list = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        elif difficulty == "medium":
            # Random sparse graph
            edge_list = []
            for i in range(num_nodes):
                num_neighbors = random.randint(2, min(5, num_nodes - 1))
                neighbors = random.sample([j for j in range(num_nodes) if j != i], num_neighbors)
                edge_list.extend([(i, j) for j in neighbors])
        else:  # hard
            # Very sparse graph
            edge_list = []
            for i in range(num_nodes):
                num_neighbors = random.randint(1, min(3, num_nodes - 1))
                neighbors = random.sample([j for j in range(num_nodes) if j != i], num_neighbors)
                edge_list.extend([(i, j) for j in neighbors])
        
        edge_index = np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        E = edge_index.shape[1]
        
        # Generate edge features
        edge_attr = np.random.randn(E, self.edge_feat_dim).astype(np.float32)
        
        # Generate language features
        rel_type = random.randrange(self.num_rel_types)
        lang_feat = np.zeros(self.lang_feat_dim, dtype=np.float32)
        lang_feat[rel_type] = 1.0  # One-hot encoding
        
        # Add some noise to make it more realistic
        lang_feat += np.random.randn(self.lang_feat_dim).astype(np.float32) * 0.1
        
        # Candidate mask
        candidate_mask = np.zeros(num_nodes, dtype=bool)
        cand_indices = random.sample(range(num_nodes), min(num_candidates, num_nodes))
        candidate_mask[cand_indices] = True
        
        # Target index
        target_index = random.choice(cand_indices)
        
        # Generate geometric label
        distance = random.uniform(5.0, 50.0)
        theta = random.uniform(-math.pi, math.pi)
        geom = np.array([distance, theta], dtype=np.float32)
        
        # For hard difficulty, make similar nodes confusing
        if difficulty == "hard":
            # Make some candidates very similar to target
            target_feat = node_feats[target_index].copy()
            for idx in cand_indices:
                if idx != target_index and random.random() < 0.3:
                    # Add similar features
                    noise = np.random.randn(self.node_feat_dim).astype(np.float32) * 0.2
                    node_feats[idx] = target_feat + noise
        
        sample = {
            "node_feats": torch.from_numpy(node_feats),
            "edge_index": torch.from_numpy(edge_index),
            "edge_attr": torch.from_numpy(edge_attr),
            "lang_feat": torch.from_numpy(lang_feat),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "target_index": target_index,
            "rel_type": rel_type,
            "geom": torch.from_numpy(geom),
        }
        
        return sample
    
    def generate_batch(
        self,
        batch_size: int = 10,
        difficulty: str = "medium",
    ) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic scenes.
        
        Args:
            batch_size: number of scenes to generate
            difficulty: difficulty level
            
        Returns:
            List of synthetic samples
        """
        samples = []
        for _ in range(batch_size):
            num_nodes = random.randint(8, 20)
            num_candidates = random.randint(3, min(10, num_nodes))
            sample = self.generate_challenging_scene(num_nodes, num_candidates, difficulty)
            samples.append(sample)
        
        return samples


class MixUpAugmentation:
    """MixUp augmentation for graph data."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def mixup(
        self,
        sample1: Dict[str, Any],
        sample2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply MixUp between two samples.
        
        Note: This is a simplified version that only mixes node features and language.
        Graph structure mixing is complex and left for future work.
        
        Args:
            sample1: First sample
            sample2: Second sample
            
        Returns:
            Mixed sample (uses structure from sample1)
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix node features (only if same number of nodes)
        if sample1["node_feats"].size(0) == sample2["node_feats"].size(0):
            mixed_node_feats = lam * sample1["node_feats"] + (1 - lam) * sample2["node_feats"]
        else:
            mixed_node_feats = sample1["node_feats"]  # Fallback to sample1
        
        # Mix language features
        mixed_lang_feat = lam * sample1["lang_feat"] + (1 - lam) * sample2["lang_feat"]
        
        # Use structure and labels from sample1 (primary sample)
        mixed_sample = {
            "node_feats": mixed_node_feats,
            "edge_index": sample1["edge_index"],
            "edge_attr": sample1["edge_attr"],
            "lang_feat": mixed_lang_feat,
            "candidate_mask": sample1["candidate_mask"],
            "target_index": sample1["target_index"],
            "rel_type": sample1["rel_type"],
            "geom": sample1["geom"],
        }
        
        return mixed_sample

