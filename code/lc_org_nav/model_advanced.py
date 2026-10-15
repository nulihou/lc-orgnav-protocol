"""
Advanced Language-Conditioned Object-Relation Graph Network with multiple innovations:

1. Multi-head cross-attention between language and graph nodes
2. Graph normalization (LayerNorm + edge dropout)
3. Residual connections in message passing
4. Relation-aware attention mechanism
5. Contrastive learning support
6. Hierarchical pooling for relation reasoning
7. Uncertainty estimation for geometric predictions
"""

from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention between language and graph nodes."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (N, H) or (B, N, H)
            key: (H,) or (B, H) or (B, M, H)
            value: (H,) or (B, H) or (B, M, H)
            mask: (N,) or (B, N) optional attention mask
        Returns:
            attended: (N, H) or (B, N, H)
        """
        # Handle different input shapes
        if query.dim() == 2:  # (N, H)
            N, H = query.shape
            batch_mode = False
            query = query.unsqueeze(0)  # (1, N, H)
            if key.dim() == 1:
                key = key.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
                value = value.unsqueeze(0).unsqueeze(0)
            elif key.dim() == 2:
                key = key.unsqueeze(1)  # (B, 1, H)
                value = value.unsqueeze(1)
        else:
            batch_mode = True
            
        B, N, H = query.shape
        _, M, _ = key.shape
        
        # Project and reshape
        Q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, N, hd)
        K = self.k_proj(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)    # (B, nh, M, hd)
        V = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, M, hd)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, N, M)
        
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, N, 1)
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)  # (B, nh, N, hd)
        out = out.transpose(1, 2).contiguous().view(B, N, H)  # (B, N, H)
        out = self.out_proj(out)
        
        if not batch_mode:
            out = out.squeeze(0)  # (N, H)
            
        return out


class RelationAwareMessagePassing(nn.Module):
    """Enhanced message passing with relation-aware attention and residual connections."""
    
    def __init__(self, hidden_dim: int, num_rel_types: int = 5, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message computation
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Self-update
        self.self_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Edge attention
        self.edge_attn = nn.Sequential(
            nn.Linear(3 * hidden_dim, 1),  # src + dst + edge
            nn.LeakyReLU(0.2),
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_nodes: (N, H)
            h_edges: (E, H)
            edge_index: (2, E)
        Returns:
            h_nodes_new: (N, H)
        """
        N = h_nodes.size(0)
        device = h_nodes.device
        
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        h_src = h_nodes[src]  # (E, H)
        h_dst = h_nodes[dst]  # (E, H)
        
        # Compute attention weights
        attn_input = torch.cat([h_src, h_dst, h_edges], dim=-1)  # (E, 3H)
        attn_logits = self.edge_attn(attn_input)  # (E, 1)
        
        # Normalize attention per destination node
        attn_weights = torch.zeros(N, device=device)
        attn_weights.index_add_(0, dst, attn_logits.squeeze(-1).exp())
        attn_weights = attn_weights[dst] + 1e-8  # (E,)
        attn_normalized = attn_logits.squeeze(-1).exp() / attn_weights  # (E,)
        
        # Compute messages
        msg_input = torch.cat([h_src, h_edges], dim=-1)  # (E, 2H)
        msgs = self.msg_mlp(msg_input)  # (E, H)
        msgs = msgs * attn_normalized.unsqueeze(-1)  # Weighted messages
        
        # Aggregate messages
        agg = torch.zeros(N, self.hidden_dim, device=device)
        agg.index_add_(0, dst, msgs)
        
        # Self-update with residual
        self_update = self.self_mlp(h_nodes)
        residual = self.residual_proj(h_nodes)
        
        h_nodes_new = self.layer_norm(residual + self_update + agg)
        
        return h_nodes_new


class HierarchicalPooling(nn.Module):
    """Hierarchical pooling for multi-scale graph representation."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.max_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mean_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion = nn.Linear(3 * hidden_dim, hidden_dim)
        
    def forward(self, h_nodes: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h_nodes: (N, H)
            mask: (N,) optional mask
        Returns:
            pooled: (H,)
        """
        if mask is not None:
            mask = mask.to(torch.float32).unsqueeze(-1)  # (N, 1)
            h_masked = h_nodes * mask
            sum_mask = mask.sum() + 1e-8
        else:
            h_masked = h_nodes
            sum_mask = h_nodes.size(0)
            
        # Max pooling
        max_pool = torch.max(h_masked, dim=0)[0]  # (H,)
        max_pool = self.max_pool_proj(max_pool)
        
        # Mean pooling
        mean_pool = h_masked.sum(dim=0) / sum_mask  # (H,)
        mean_pool = self.mean_pool_proj(mean_pool)
        
        # Attention pooling
        attn_scores = self.attn_pool(h_nodes)  # (N, 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=0)  # (N, 1)
        attn_pool = (h_nodes * attn_weights).sum(dim=0)  # (H,)
        
        # Fusion
        pooled = self.fusion(torch.cat([max_pool, mean_pool, attn_pool], dim=-1))
        
        return pooled


class LCORGNetAdvanced(nn.Module):
    """Advanced Language-Conditioned Object-Relation Graph Network.
    
    Key innovations:
    1. Multi-head cross-attention for language-graph interaction
    2. Relation-aware message passing with edge attention
    3. Hierarchical pooling for multi-scale reasoning
    4. Residual connections and layer normalization
    5. Uncertainty estimation for geometric predictions
    6. Support for contrastive learning
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        lang_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rel_types: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_dropout: bool = True,
        edge_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_rel_types = num_rel_types
        self.use_edge_dropout = use_edge_dropout
        self.edge_dropout_rate = edge_dropout_rate
        
        # Initial encoders with layer norm
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Cross-attention layers
        self.cross_attns = nn.ModuleList([
            MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # FiLM layers
        self.film_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 4 * hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Relation-aware message passing layers
        self.mp_layers = nn.ModuleList([
            RelationAwareMessagePassing(hidden_dim, num_rel_types, dropout)
            for _ in range(num_layers)
        ])
        
        # Hierarchical pooling
        self.hierarchical_pool = HierarchicalPooling(hidden_dim)
        
        # Goal selection head with multi-scale features
        self.goal_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Relation type head with hierarchical features
        self.rel_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_rel_types),
        )
        
        # Geometric regression head with uncertainty
        self.geom_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [distance, theta]
        )
        
        # Uncertainty estimation (log variance)
        self.geom_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [log_var_dist, log_var_theta]
        )
        
        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
    def edge_dropout(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                     p: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply edge dropout during training."""
        if not self.training or p == 0:
            return edge_index, edge_attr
            
        E = edge_index.size(1)
        mask = torch.rand(E, device=edge_index.device) > p
        
        return edge_index[:, mask], edge_attr[mask]
        
    def forward_single(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass for a single scene.
        
        Args:
            node_feats: (N, F_node)
            edge_index: (2, E)
            edge_attr: (E, F_edge)
            lang_feat: (F_lang,)
            candidate_mask: (N,)
            return_embeddings: whether to return intermediate embeddings for contrastive learning
            
        Returns:
            dict with predictions and optionally embeddings
        """
        device = node_feats.device
        N = node_feats.size(0)
        
        # Apply edge dropout
        if self.use_edge_dropout and self.training:
            edge_index, edge_attr = self.edge_dropout(edge_index, edge_attr, self.edge_dropout_rate)
        
        # Initial encoding
        h_nodes = self.node_encoder(node_feats)  # (N, H)
        h_edges = self.edge_encoder(edge_attr)   # (E, H)
        h_lang = self.lang_encoder(lang_feat)    # (H,)
        
        # Store initial embeddings
        h_nodes_init = h_nodes.clone()
        
        # Multi-layer message passing with cross-attention and FiLM
        for layer_idx in range(self.num_layers):
            # Cross-attention: nodes attend to language
            h_nodes_attn = self.cross_attns[layer_idx](h_nodes, h_lang, h_lang)
            h_nodes = h_nodes + h_nodes_attn  # Residual
            
            # FiLM modulation
            film = self.film_layers[layer_idx](h_lang)  # (4H,)
            gamma_node, beta_node, gamma_edge, beta_edge = torch.chunk(film, 4, dim=-1)
            
            h_nodes_f = h_nodes * gamma_node + beta_node
            h_edges_f = h_edges * gamma_edge + beta_edge
            
            # Relation-aware message passing
            h_nodes = self.mp_layers[layer_idx](h_nodes_f, h_edges_f, edge_index)
        
        # Hierarchical pooling for global representation
        pooled = self.hierarchical_pool(h_nodes, candidate_mask)  # (H,)
        
        # Goal selection: per-node scoring
        lang_expanded = h_lang.unsqueeze(0).expand(N, -1)  # (N, H)
        goal_input = torch.cat([h_nodes, lang_expanded], dim=-1)  # (N, 2H)
        goal_logits = self.goal_head(goal_input).squeeze(-1)  # (N,)
        
        # Relation type prediction
        rel_input = torch.cat([pooled, h_lang], dim=-1)  # (2H,)
        rel_type_logits = self.rel_head(rel_input)  # (num_rel_types,)
        
        # Geometric prediction with uncertainty
        # Use attention-weighted candidate representation
        cand_mask = candidate_mask.to(torch.float32)  # (N,)
        if cand_mask.sum() > 0:
            # Soft attention over candidates based on goal logits
            cand_logits = goal_logits * cand_mask + (1 - cand_mask) * (-1e9)
            cand_weights = F.softmax(cand_logits, dim=0).unsqueeze(-1)  # (N, 1)
            geom_repr = (h_nodes * cand_weights).sum(dim=0)  # (H,)
        else:
            geom_repr = pooled
            
        geom_mean = self.geom_mean_head(geom_repr)  # (2,)
        geom_logvar = self.geom_logvar_head(geom_repr)  # (2,)
        
        outputs = {
            "goal_logits": goal_logits,
            "rel_type_logits": rel_type_logits,
            "geom_pred": geom_mean,
            "geom_logvar": geom_logvar,
            "pooled_repr": pooled,
        }
        
        if return_embeddings:
            # For contrastive learning
            node_embeddings = self.contrastive_proj(h_nodes)  # (N, H/2)
            lang_embedding = self.contrastive_proj(h_lang)    # (H/2,)
            outputs["node_embeddings"] = node_embeddings
            outputs["lang_embedding"] = lang_embedding
            outputs["h_nodes"] = h_nodes
            
        return outputs
    
    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Batched forward (currently processes scenes individually)."""
        if node_feats.dim() == 2:
            return self.forward_single(node_feats, edge_index, edge_attr, 
                                      lang_feat, candidate_mask, return_embeddings)
        
        # Batch processing
        B = node_feats.size(0)
        outputs_list = []
        
        for b in range(B):
            out_b = self.forward_single(
                node_feats[b],
                edge_index[b],
                edge_attr[b],
                lang_feat[b],
                candidate_mask[b],
                return_embeddings,
            )
            outputs_list.append(out_b)
        
        # Stack outputs
        outputs = {
            "goal_logits": torch.stack([o["goal_logits"] for o in outputs_list], dim=0),
            "rel_type_logits": torch.stack([o["rel_type_logits"] for o in outputs_list], dim=0),
            "geom_pred": torch.stack([o["geom_pred"] for o in outputs_list], dim=0),
            "geom_logvar": torch.stack([o["geom_logvar"] for o in outputs_list], dim=0),
            "pooled_repr": torch.stack([o["pooled_repr"] for o in outputs_list], dim=0),
        }
        
        if return_embeddings:
            outputs["node_embeddings"] = [o["node_embeddings"] for o in outputs_list]
            outputs["lang_embedding"] = torch.stack([o["lang_embedding"] for o in outputs_list], dim=0)
            outputs["h_nodes"] = [o["h_nodes"] for o in outputs_list]
            
        return outputs

