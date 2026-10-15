"""
End-to-end baseline that directly processes node features without explicit graph structure.

Uses simple MLPs with global pooling, no message passing.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class EndToEndNet(nn.Module):
    """End-to-end baseline without explicit graph message passing."""
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,  # Ignored
        lang_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rel_types: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rel_types = num_rel_types
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Deep processing layers (no graph structure)
        self.process_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.process_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
        
        # Goal selection head
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Relation head
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rel_types),
        )
        
        # Geometry head
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
        Forward pass without using graph structure.
        
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
        
        # Expand language to all nodes
        h_lang_expanded = h_lang.unsqueeze(0).expand(N, -1)  # (N, H)
        
        # Process with language conditioning (but no graph structure)
        for layer in self.process_layers:
            combined = torch.cat([h_nodes, h_lang_expanded], dim=-1)  # (N, 2H)
            h_nodes = h_nodes + layer(combined)  # Residual
        
        # Goal selection
        goal_input = torch.cat([h_nodes, h_lang_expanded], dim=-1)
        goal_logits = self.goal_head(goal_input).squeeze(-1)
        
        # Relation type (global pooling + language)
        pooled = torch.max(h_nodes, dim=0)[0]
        rel_input = torch.cat([pooled, h_lang], dim=-1)
        rel_type_logits = self.rel_head(rel_input)
        
        # Geometry (candidate-weighted)
        cand_mask = candidate_mask.to(torch.float32)
        if cand_mask.sum() > 0:
            weights = cand_mask / (cand_mask.sum() + 1e-6)
            geom_repr = (weights.unsqueeze(-1) * h_nodes).sum(dim=0)
        else:
            geom_repr = pooled
        geom_pred = self.geom_head(geom_repr)
        
        return {
            "goal_logits": goal_logits,
            "rel_type_logits": rel_type_logits,
            "geom_pred": geom_pred,
        }

