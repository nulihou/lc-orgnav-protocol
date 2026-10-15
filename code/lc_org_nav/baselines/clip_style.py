"""
CLIP-style baseline for goal selection.

Directly matches language embeddings with node embeddings without graph structure.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPStyleNet(nn.Module):
    """CLIP-style baseline using direct language-node matching."""
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,  # Ignored
        lang_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,  # Depth of encoders
        num_rel_types: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rel_types = num_rel_types
        
        # Node encoder (deep)
        layers = []
        in_dim = node_feat_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.node_encoder = nn.Sequential(*layers)
        
        # Language encoder (deep)
        layers = []
        in_dim = lang_feat_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.lang_encoder = nn.Sequential(*layers)
        
        # Projection heads for contrastive matching
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.lang_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Temperature parameter for similarity scaling
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Auxiliary heads
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rel_types),
        )
        
        self.geom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,  # Ignored
        edge_attr: torch.Tensor,   # Ignored
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass using CLIP-style matching.
        
        Args:
            node_feats: (N, F_node)
            edge_index: (2, E) - ignored
            edge_attr: (E, F_edge) - ignored
            lang_feat: (F_lang,)
            candidate_mask: (N,)
            
        Returns:
            dict with predictions
        """
        N = node_feats.size(0)
        device = node_feats.device
        
        # Encode
        h_nodes = self.node_encoder(node_feats)  # (N, H)
        h_lang = self.lang_encoder(lang_feat)     # (H,)
        
        # Project to matching space
        node_embeddings = self.node_proj(h_nodes)  # (N, H)
        lang_embedding = self.lang_proj(h_lang)    # (H,)
        
        # Normalize for cosine similarity
        node_embeddings = F.normalize(node_embeddings, dim=-1)
        lang_embedding = F.normalize(lang_embedding, dim=-1)
        
        # Compute similarities (CLIP-style)
        similarities = torch.matmul(node_embeddings, lang_embedding) / self.temperature
        
        # Use similarities as goal logits
        goal_logits = similarities
        
        # Relation type prediction
        rel_type_logits = self.rel_head(h_lang)
        
        # Geometry prediction (use candidate-weighted node features)
        cand_mask = candidate_mask.to(torch.float32)
        if cand_mask.sum() > 0:
            # Use similarity-weighted combination
            sim_weights = F.softmax(goal_logits * cand_mask + (1 - cand_mask) * (-1e9), dim=0)
            geom_repr = (sim_weights.unsqueeze(-1) * h_nodes).sum(dim=0)
        else:
            geom_repr = torch.max(h_nodes, dim=0)[0]
        
        geom_pred = self.geom_head(geom_repr)
        
        return {
            "goal_logits": goal_logits,
            "rel_type_logits": rel_type_logits,
            "geom_pred": geom_pred,
        }

