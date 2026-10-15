"""
目标选择模块

提供 Oracle、GS、Random 三种目标选择策略。
"""

import random
import torch
from typing import Dict, Any, Optional, Tuple

from lc_org_nav.model import LCORGNet
from lc_org_nav.gs_utils import select_goal_from_logits


def select_target_oracle(graph: Dict[str, Any]) -> Tuple[int, Optional[bool]]:
    """Oracle模式：使用GT目标"""
    chosen_idx = graph["target_idx_gt"]
    return chosen_idx, None  # Oracle不算GS指标


def select_target_gs(
    graph: Dict[str, Any],
    model: LCORGNet,
    device: torch.device,
    debug: bool = True,  # 添加debug参数
) -> Tuple[int, bool, Dict[str, Any]]:
    """GS模式：使用模型预测
    
    Returns:
        chosen_idx: 选择的候选索引
        gs_correct: GS@1是否正确
        gs_metrics: 包含GS@1、GS@3和topk_indices的字典
    """
    target_idx_gt = graph["target_idx_gt"]
    candidate_mask = graph["candidate_mask"]
    
    node_feats = graph["node_feats"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    lang_feat = graph["lang_feat"].to(device)
    candidate_mask_t = candidate_mask.to(device)
    
    # Add batch dimension if needed
    if node_feats.dim() == 2:
        node_feats = node_feats.unsqueeze(0)
    if edge_attr.dim() == 2:
        edge_attr = edge_attr.unsqueeze(0)
    if lang_feat.dim() == 1:
        lang_feat = lang_feat.unsqueeze(0)
    if candidate_mask_t.dim() == 1:
        candidate_mask_t = candidate_mask_t.unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        outputs = model.forward_single(
            node_feats=node_feats[0],
            edge_index=edge_index,
            edge_attr=edge_attr[0],
            lang_feat=lang_feat[0],
            candidate_mask=candidate_mask_t[0],
        )
        goal_logits = outputs["goal_logits"]  # (N,)
    
    # ===== Step B: 添加详细的GS debug打印 =====
    if debug:
        cand_indices = torch.nonzero(candidate_mask_t[0], as_tuple=False).squeeze(-1)
        logits_cand = goal_logits[cand_indices]
        print(f"  [GS-Debug] candidate_mask: {candidate_mask_t[0].cpu().numpy()}")
        print(f"  [GS-Debug] cand_indices (global): {cand_indices.cpu().tolist()}")
        print(f"  [GS-Debug] logits[cand] = {logits_cand.cpu().numpy()}")
        print(f"  [GS-Debug] GT target_idx_gt (global): {target_idx_gt}")
    
    # 使用公共函数选择目标（确保与离线评估一致）
    try:
        chosen_idx, topk_indices = select_goal_from_logits(
            goal_logits, candidate_mask_t[0], topk=3
        )
    except ValueError as e:
        # 如果没有候选节点，返回错误
        return -1, False, {"gs_at_1": False, "gs_at_3": False, "topk_indices": []}
    
    # ===== Step B: 继续debug打印 =====
    if debug:
        print(f"  [GS-Debug] top1_idx (global) = {chosen_idx}, GT_idx (global) = {target_idx_gt}")
        print(f"  [GS-Debug] topk_indices (global) = {topk_indices}")
        print(f"  [GS-Debug] top1 == GT? {chosen_idx == target_idx_gt} → {'CORRECT' if chosen_idx == target_idx_gt else 'WRONG'}")
    
    # 计算GS@1和GS@3（使用与offline完全一致的逻辑）
    # Step C: 重用offline的GS@1计算逻辑
    gs_at_1 = (chosen_idx == target_idx_gt)
    gs_at_3 = (target_idx_gt in topk_indices)
    gs_correct = gs_at_1
    
    return chosen_idx, gs_correct, {
        "gs_at_1": gs_at_1,
        "gs_at_3": gs_at_3,
        "topk_indices": topk_indices  # 用于调试日志
    }


def select_target_random(graph: Dict[str, Any]) -> Tuple[int, bool]:
    """Random模式：随机选择"""
    target_idx_gt = graph["target_idx_gt"]
    candidate_mask = graph["candidate_mask"]
    
    cand_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    if cand_indices.numel() == 0:
        return -1, False
    
    chosen_local = random.randint(0, len(cand_indices) - 1)
    chosen_idx = int(cand_indices[chosen_local])
    gs_correct = (chosen_idx == target_idx_gt)
    
    return chosen_idx, gs_correct

