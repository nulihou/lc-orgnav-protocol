"""Query-token Transformer baseline for goal selection.

This is a non-graph baseline: it uses node features and the query feature, but
ignores edge attributes and graph message passing. A learned query token attends
over the node tokens through a small Transformer encoder, then each candidate is
scored from its contextualized node state and the contextualized query state.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTokenTransformerNet(nn.Module):
    """Query-aware attention baseline without explicit graph structure."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,  # Ignored by design.
        lang_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rel_types: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        del edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_rel_types = num_rel_types

        if hidden_dim % num_heads != 0:
            num_heads = 1

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rel_types),
        )
        self.geom_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,  # Ignored by design.
        edge_attr: torch.Tensor,  # Ignored by design.
        lang_feat: torch.Tensor,
        candidate_mask: torch.Tensor,
        node_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        del edge_index, edge_attr
        squeeze_batch = False
        if node_feats.dim() == 2:
            node_feats = node_feats.unsqueeze(0)
            lang_feat = lang_feat.unsqueeze(0)
            candidate_mask = candidate_mask.unsqueeze(0)
            if node_valid_mask is not None:
                node_valid_mask = node_valid_mask.unsqueeze(0)
            squeeze_batch = True

        h_nodes = self.node_encoder(node_feats)
        h_query = self.lang_encoder(lang_feat)

        batch_size, n_nodes, _ = h_nodes.shape
        if node_valid_mask is None:
            node_valid_mask = torch.ones(
                batch_size,
                n_nodes,
                dtype=torch.bool,
                device=node_feats.device,
            )
        else:
            node_valid_mask = node_valid_mask.to(torch.bool)

        tokens = torch.cat([h_query.unsqueeze(1), h_nodes], dim=1)
        query_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=node_feats.device)
        key_padding_mask = ~torch.cat([query_valid, node_valid_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        query_ctx = encoded[:, 0, :]
        node_ctx = encoded[:, 1:, :]

        query_expanded = query_ctx.unsqueeze(1).expand(-1, n_nodes, -1)
        score_input = torch.cat(
            [node_ctx, query_expanded, node_ctx * query_expanded, torch.abs(node_ctx - query_expanded)],
            dim=-1,
        )
        goal_logits = self.goal_head(score_input).squeeze(-1)

        rel_type_logits = self.rel_head(query_ctx)

        cand_mask = candidate_mask.to(torch.bool) & node_valid_mask
        masked_logits = goal_logits.masked_fill(~cand_mask, -1e9)
        has_candidate = cand_mask.any(dim=1)
        weights = F.softmax(masked_logits, dim=1).masked_fill(~has_candidate.unsqueeze(1), 0.0)
        geom_repr = (weights.unsqueeze(-1) * node_ctx).sum(dim=1)
        fallback_repr = node_ctx.masked_fill(~node_valid_mask.unsqueeze(-1), -1e9).max(dim=1)[0]
        geom_repr = torch.where(has_candidate.unsqueeze(1), geom_repr, fallback_repr)
        geom_pred = self.geom_head(torch.cat([geom_repr, query_ctx], dim=-1))

        if squeeze_batch:
            goal_logits = goal_logits.squeeze(0)
            rel_type_logits = rel_type_logits.squeeze(0)
            geom_pred = geom_pred.squeeze(0)

        return {
            "goal_logits": goal_logits,
            "rel_type_logits": rel_type_logits,
            "geom_pred": geom_pred,
        }
