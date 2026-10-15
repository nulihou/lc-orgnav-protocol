"""
GNN-only baseline without language conditioning.

Uses standard GNN (GAT-style) for goal selection without language input.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNOnlyNet(nn.Module):
    """GNN baseline without language conditioning."""
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        lang_feat_dim: int,  # Ignored, for interface compatibility
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rel_types: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rel_types = num_rel_types
        
        # Encoders (no language encoder)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # GAT-style message passing
        self.msg_mlps = nn.ModuleList()
        self.attn_mlps = nn.ModuleList()
        self.self_mlps = nn.ModuleList()
        
        for _ in range(num_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.attn_mlps.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, 1),
                    nn.LeakyReLU(0.2),
                )
            )
            self.self_mlps.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Prediction heads
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
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
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,  # Ignored
        candidate_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Forward pass without language conditioning."""
        device = node_feats.device
        N = node_feats.size(0)
        
        h_nodes = self.node_encoder(node_feats)
        h_edges = self.edge_encoder(edge_attr)
        
        # Message passing (GAT-style)
        for layer_idx in range(self.num_layers):
            src = edge_index[0]
            dst = edge_index[1]
            
            h_src = h_nodes[src]
            h_dst = h_nodes[dst]
            
            # Attention weights
            attn_input = torch.cat([h_src, h_dst, h_edges], dim=-1)
            attn_logits = self.attn_mlps[layer_idx](attn_input)
            
            # Normalize per destination
            attn_weights = torch.zeros(N, device=device)
            attn_weights.index_add_(0, dst, attn_logits.squeeze(-1).exp())
            attn_weights = attn_weights[dst] + 1e-8
            attn_normalized = attn_logits.squeeze(-1).exp() / attn_weights
            
            # Messages
            msg_input = torch.cat([h_src, h_edges], dim=-1)
            msgs = self.msg_mlps[layer_idx](msg_input)
            msgs = msgs * attn_normalized.unsqueeze(-1)
            
            # Aggregate
            agg = torch.zeros(N, self.hidden_dim, device=device)
            agg.index_add_(0, dst, msgs)
            
            # Update
            self_update = self.self_mlps[layer_idx](h_nodes)
            h_nodes = F.relu(h_nodes + self_update + agg)
        
        # Goal selection (no language conditioning)
        goal_logits = self.goal_head(h_nodes).squeeze(-1)
        
        # Relation type (use global pooling)
        pooled = torch.max(h_nodes, dim=0)[0]
        rel_type_logits = self.rel_head(pooled)
        
        # Geometry (use candidate-weighted representation)
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

