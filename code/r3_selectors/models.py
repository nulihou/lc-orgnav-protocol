"""Matched geometry/query selectors; DriveTransformer adaptation is explicit.

The DriveTransformer adapter reimplements the official task-attention primitives
in plain PyTorch: independent pre-normalized Q/K/V, positional additions to Q/K,
residual attention and pre-normalized SwiGLU. It does not reproduce sensor or
temporal attention, detection, map prediction or trajectory planning.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from lc_org_nav.model import LCORGNet


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, out))


def masked_pool(x, valid):
    count = valid.sum(1, keepdim=True).clamp_min(1)
    mean = (x * valid.unsqueeze(-1)).sum(1) / count
    maximum = x.masked_fill(~valid.unsqueeze(-1), -1e9).max(1).values
    return torch.cat([mean, maximum], -1)


def edge_summary(edges, edge_mask):
    # Dense edges are indexed [batch, source, destination, feature].
    mask = edge_mask.unsqueeze(-1)
    incoming_mean = (edges * mask).sum(1) / mask.sum(1).clamp_min(1)
    outgoing_mean = (edges * mask).sum(2) / mask.sum(2).clamp_min(1)
    incoming_max = edges.masked_fill(~mask, -1e9).max(1).values
    outgoing_max = edges.masked_fill(~mask, -1e9).max(2).values
    incoming_max = torch.where(edge_mask.any(1).unsqueeze(-1), incoming_max, 0.0)
    outgoing_max = torch.where(edge_mask.any(2).unsqueeze(-1), outgoing_max, 0.0)
    return torch.cat([incoming_mean, outgoing_mean, incoming_max, outgoing_max], -1)


class BatchedOriginalGraph(nn.Module):
    """Original LCORGNet modules and arithmetic, vectorized across graphs.

    No new skip, normalization or auxiliary score path is added. Padding is
    excluded in each aggregation. Input normalization is shared with all rows.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = LCORGNet(32, 8, 32, cfg.hidden_dim, cfg.num_layers, 5,
                                use_rel_gate=True, use_edge_context_head=False)

    def forward(self, nodes, edges, query, candidates, valid, edge_mask):
        m = self.backbone
        h = m.node_encoder(nodes)
        e = m.edge_encoder(edges)
        q = m.lang_proj(query)
        for layer in range(m.num_layers):
            gn, bn, ge, be = m.film_layers[layer](q).chunk(4, -1)
            hn = h * gn[:, None] + bn[:, None]
            he = e * ge[:, None, None] + be[:, None, None]
            src = hn[:, :, None].expand(-1, -1, h.shape[1], -1)
            messages = m.msg_mlps[layer](torch.cat([src, he], -1))
            agg = (messages * edge_mask[..., None]).sum(1)
            h = F.relu(m.self_mlps[layer](hn) + agg) * valid[..., None]
        r = m.rel_proj(query)[:, None].expand(-1, h.shape[1], -1)
        gate = m.gate_mlp(torch.cat([h, r], -1))
        h = gate * h + (1 - gate) * r
        return m.goal_head(torch.cat([h, q[:, None].expand_as(h)], -1)).squeeze(-1)


class MatchedGeometryMLP(nn.Module):
    """Candidate-wise MLP with pooled scene AND all pairwise edge geometry."""
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.node = mlp(32, h, h)
        self.edge = mlp(32, h, h)
        self.query = mlp(32, h, h)
        self.score = nn.Sequential(nn.Linear(h * 5, h * 2), nn.ReLU(),
                                   nn.Dropout(cfg.dropout), nn.Linear(h * 2, h * 2),
                                   nn.ReLU(), nn.Linear(h * 2, 1))

    def forward(self, nodes, edges, query, candidates, valid, edge_mask):
        h = self.node(nodes)
        es = self.edge(edge_summary(edges, edge_mask))
        q = self.query(query)[:, None].expand_as(h)
        pooled = masked_pool(h, valid)[:, None].expand(-1, h.shape[1], -1)
        return self.score(torch.cat([h, es, q, pooled], -1)).squeeze(-1)


class GeometryAttention(nn.Module):
    """Multihead attention with learned pairwise edge bias and query tokens."""
    def __init__(self, h, heads, dropout):
        super().__init__()
        self.h, self.heads, self.d = h, heads, h // heads
        self.norm = nn.LayerNorm(h)
        self.qkv = nn.Linear(h, h * 3)
        self.edge_bias = mlp(8, 32, heads)
        self.out = nn.Linear(h, h)
        self.dropout = dropout
        self.ffn = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(h * 2, h))
        self.qread = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.qnorm = nn.LayerNorm(h)
        self.qffn = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h * 2), nn.GELU(),
                                  nn.Linear(h * 2, h))

    def forward(self, h, q, edges, valid, edge_mask):
        b, n, _ = h.shape
        qkv = self.qkv(self.norm(h)).view(b, n, 3, self.heads, self.d)
        qq, kk, vv = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        # Bias for [query destination, key source] follows the graph edge direction.
        bias = self.edge_bias(edges).permute(0, 3, 2, 1)
        bias = bias * edge_mask.transpose(1, 2)[:, None]
        bias = bias.masked_fill(~valid[:, None, None, :], float('-inf'))
        out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.0)
        h = h + self.out(out.transpose(1, 2).reshape(b, n, self.h))
        h = h + self.ffn(h)
        qnorm = self.qnorm(q)
        read, _ = self.qread(qnorm, self.qnorm(h), self.qnorm(h),
                            key_padding_mask=~valid, need_weights=False)
        q = q + read
        q = q + self.qffn(q)
        return h * valid[..., None], q


class QueryGeometryAttention(nn.Module):
    """Purpose-built strong reference, not attributed to a published SOTA."""
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.node = mlp(32, h, h)
        self.edge = mlp(32, h, h)
        self.query = mlp(32, h, h)
        self.condition = nn.ModuleList([mlp(h * 2, h, h) for _ in range(cfg.num_layers)])
        self.layers = nn.ModuleList([GeometryAttention(h, cfg.num_heads, cfg.dropout)
                                     for _ in range(cfg.num_layers)])
        self.score = mlp(h * 3, h, 1)

    def forward(self, nodes, edges, query, candidates, valid, edge_mask):
        h = self.node(nodes) + self.edge(edge_summary(edges, edge_mask))
        q = self.query(query)[:, None]
        for condition, layer in zip(self.condition, self.layers):
            h = h + condition(torch.cat([h, q.expand_as(h)], -1))
            h, q = layer(h, q, edges, valid, edge_mask)
        qe = q.expand_as(h)
        return self.score(torch.cat([h, qe, h * qe], -1)).squeeze(-1)


class DriveTaskAttention(nn.Module):
    """Plain-PyTorch adaptation of official AttentionLayer, not renamed MHA."""
    def __init__(self, h, heads, dropout):
        super().__init__()
        self.heads, self.d = heads, h // heads
        self.wq = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.wk = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.wv = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        self.out = nn.Linear(h, h)
        self.dropout = dropout

    def forward(self, tokens, position, valid):
        b, n, h = tokens.shape
        q = self.wq(tokens + position).reshape(b, n, self.heads, self.d).transpose(1, 2)
        k = self.wk(tokens + position).reshape(b, n, self.heads, self.d).transpose(1, 2)
        v = self.wv(tokens).reshape(b, n, self.heads, self.d).transpose(1, 2)
        mask = torch.zeros(b, 1, n, n, device=tokens.device, dtype=tokens.dtype)
        mask = mask.masked_fill(~valid[:, None, None], float('-inf'))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                       dropout_p=self.dropout if self.training else 0.0)
        return tokens + self.out(out.transpose(1, 2).reshape(b, n, h))


class DriveSwiGLU(nn.Module):
    def __init__(self, h, dropout):
        super().__init__()
        # Same 2/3 hidden-size adjustment and 8-channel rounding as official code.
        middle = (int(h * 4 * 2 / 3) + 7) // 8 * 8
        self.norm = nn.LayerNorm(h)
        self.w12 = nn.Linear(h, middle * 2)
        self.w3 = nn.Linear(middle, h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(self.norm(x)).chunk(2, -1)
        return x + self.drop(self.w3(F.silu(a) * b))


class DriveTransformerTaskAdapter(nn.Module):
    """ICLR2025 DriveTransformer task-block adapted to candidate-ID selection.

    Retained: pre-normalized independent Q/K/V, Q/K positional encoding,
    residual attention, SwiGLU, direct interaction among agent and task queries.
    Removed: absent sensor and temporal streams, map/detection/motion task heads.
    Added adapter: common NPZ geometry encoder and semantic candidate score head.
    """
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.node = mlp(32, h, h)
        self.edge = mlp(32, h, h)
        self.query = mlp(32, h, h)
        self.position = mlp(32, h, h)
        self.types = nn.Parameter(torch.zeros(2, h))
        self.attn = nn.ModuleList([DriveTaskAttention(h, cfg.num_heads, cfg.dropout)
                                  for _ in range(cfg.num_layers)])
        self.ffn = nn.ModuleList([DriveSwiGLU(h, cfg.dropout) for _ in range(cfg.num_layers)])
        self.score = mlp(h * 3, h, 1)
        self.apply(self.initialize)

    @staticmethod
    def initialize(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, nodes, edges, query, candidates, valid, edge_mask):
        h = self.node(nodes) + self.edge(edge_summary(edges, edge_mask)) + self.types[0]
        q = self.query(query)[:, None] + self.types[1]
        tokens = torch.cat([h, q], 1)
        position = torch.cat([self.position(nodes), torch.zeros_like(q)], 1)
        token_valid = torch.cat([valid, torch.ones_like(valid[:, :1])], 1)
        for attn, ffn in zip(self.attn, self.ffn):
            tokens = ffn(attn(tokens, position, token_valid)) * token_valid[..., None]
        h, q = tokens[:, :-1], tokens[:, -1:]
        qe = q.expand_as(h)
        return self.score(torch.cat([h, qe, h * qe], -1)).squeeze(-1)


MODEL_REGISTRY = {
    'graph_relation_gate': BatchedOriginalGraph,
    'matched_geometry_mlp': MatchedGeometryMLP,
    'query_geometry_attention': QueryGeometryAttention,
    'drivetransformer_task_adapted': DriveTransformerTaskAdapter,
}


def build_model(cfg):
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return MODEL_REGISTRY[cfg.name](cfg)
