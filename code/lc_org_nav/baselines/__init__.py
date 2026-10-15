"""
Baseline models for comparison with LC-ORGNav.

Includes:
1. Rule-based baseline (geometric heuristics)
2. GNN-only baseline (no language conditioning)
3. End-to-end baseline (direct feature to prediction)
4. CLIP-style baseline (language-feature matching)
5. Query-token Transformer baseline (attention without graph edges)
"""

from .rule_based import RuleBasedSelector
from .gnn_only import GNNOnlyNet
from .end_to_end import EndToEndNet
from .clip_style import CLIPStyleNet
from .attention_pool import QueryTokenTransformerNet

__all__ = [
    "RuleBasedSelector",
    "GNNOnlyNet",
    "EndToEndNet",
    "CLIPStyleNet",
    "QueryTokenTransformerNet",
]

