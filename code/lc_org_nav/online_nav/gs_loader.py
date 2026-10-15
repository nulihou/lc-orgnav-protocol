"""
GS模型加载模块

从checkpoint加载LCORGNet模型用于在线推理。
"""

import time
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lc_org_nav.model import LCORGNet

from lc_org_nav.model import LCORGNet
from lc_org_nav.config import GoalOnlyConfig


class _R3CheckpointAdapter:
    """Expose the frozen R3 selector through the legacy online evaluator API."""
    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle['model']
        self.device = bundle['device']

    def eval(self):
        self.model.eval()
        return self

    def forward_single(self, node_feats, edge_index, edge_attr, lang_feat, candidate_mask):
        import numpy as np
        from r3_selectors.pipeline import assemble_batch, forward_logits
        candidate_np = candidate_mask.detach().cpu().numpy().astype(bool)
        sample = {
            'nodes': node_feats.detach().cpu().numpy().astype(np.float32),
            'edge_index': edge_index.detach().cpu().numpy().astype(np.int64),
            'edge_attr': edge_attr.detach().cpu().numpy().astype(np.float32),
            'query': lang_feat.detach().cpu().numpy().astype(np.float32),
            'candidates': candidate_np,
            'target': int(np.flatnonzero(candidate_np)[0]),
        }
        batch = assemble_batch([sample], self.bundle['normalization'], self.device)
        logits = forward_logits(self.model, batch)[0]
        return {'goal_logits': logits}

# 从 carla_collect_data 导入常量
import os
import sys
# 添加 scripts 目录到路径
_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from scripts.data.carla_collect_data import (
        NODE_FEAT_DIM,
        EDGE_FEAT_DIM,
        LANG_FEAT_DIM,
        NUM_REL_TYPES,
    )
except ModuleNotFoundError:
    NODE_FEAT_DIM = 32
    EDGE_FEAT_DIM = 8
    LANG_FEAT_DIM = 32
    NUM_REL_TYPES = 5


def load_gs_model(checkpoint_path: str, device: torch.device) -> LCORGNet:
    """加载GS模型"""
    print(f"[Model] Loading from {checkpoint_path}...")
    start_time = time.time()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[Model] Checkpoint loaded in {time.time() - start_time:.2f}s")

    # Current R3 checkpoints are structured bundles.  Use the native loader and
    # adapt only the forward_single interface expected by the legacy CARLA loop.
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint and 'state_dict' in checkpoint:
        from r3_selectors.pipeline import load_checkpoint
        bundle = load_checkpoint(checkpoint_path, device=str(device))
        model = _R3CheckpointAdapter(bundle)
        model.eval()
        print(f"[Model] Loaded native R3 checkpoint ({bundle['model_config']['name']})")
        return model
    use_rel_gate = False
    for key in checkpoint.keys():
        if "rel_proj" in key or "gate_mlp" in key:
            use_rel_gate = True
            break
    use_edge_context_head = any(
        (
            "raw_node_proj" in key
            or "edge_context_proj" in key
            or "graph_pool_proj" in key
            or "edge_goal_head" in key
        )
        for key in checkpoint.keys()
    )
    
    # 创建模型
    print(f"[Model] Creating model (device={device})...")
    config = GoalOnlyConfig()
    model = LCORGNet(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        lang_feat_dim=LANG_FEAT_DIM,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_rel_types=NUM_REL_TYPES,
        use_rel_gate=use_rel_gate,
        use_edge_context_head=use_edge_context_head,
    )
    print(f"[Model] Model created, moving to device {device}...")
    model = model.to(device)
    print(f"[Model] Loading state dict...")
    model.load_state_dict(checkpoint)
    print(f"[Model] Setting to eval mode...")
    model.eval()
    print(f"[Model] Loaded successfully (use_rel_gate={use_rel_gate}, use_edge_context_head={use_edge_context_head})")
    return model

