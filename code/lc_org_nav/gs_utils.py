"""
Goal Selection工具函数

提供统一的从logits到chosen_idx的映射逻辑，确保离线和在线评估的一致性。
"""

import torch
from typing import Tuple, List, Optional


def select_goal_from_logits(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    topk: int = 3,
) -> Tuple[int, List[int]]:
    """
    给定所有节点的 logits 和 candidate_mask，返回：
    - chosen_idx: Top-1 目标在全局节点中的索引
    - topk_indices: Top-k 目标在全局节点中的索引列表
    
    Args:
        logits: (N,) 所有节点的logits
        candidate_mask: (N,) 0/1或bool，标记哪些节点是候选
        topk: 返回Top-k索引（默认3，用于GS@3计算）
    
    Returns:
        chosen_idx: Top-1目标在全局节点中的索引（int）
        topk_indices: Top-k目标在全局节点中的索引列表（List[int]）
    
    Raises:
        ValueError: 如果没有候选节点
    """
    # 确保logits是1D
    if logits.dim() > 1:
        logits = logits.squeeze()
    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    
    # 统一处理candidate_mask（支持0/1或bool）
    if isinstance(candidate_mask, torch.Tensor):
        if candidate_mask.dim() > 1:
            candidate_mask = candidate_mask.squeeze()
        cand_mask = candidate_mask > 0
    else:
        import numpy as np
        cand_mask = torch.as_tensor(candidate_mask, dtype=torch.bool)
        if cand_mask.dim() > 1:
            cand_mask = cand_mask.squeeze()
    
    # 获取候选节点的全局索引
    candidate_indices = torch.nonzero(cand_mask, as_tuple=False).squeeze(-1)
    
    # 处理边界情况
    if candidate_indices.numel() == 0:
        raise ValueError("No candidate nodes found in candidate_mask")
    
    # 如果只有一个候选，确保返回正确的形状
    if candidate_indices.dim() == 0:
        candidate_indices = candidate_indices.unsqueeze(0)
    
    # 取候选节点的 logits
    cand_logits = logits[candidate_indices]  # [K]
    
    # Top-1: 找到相对索引（0..K-1）
    best_rel = int(torch.argmax(cand_logits))  # 0..K-1
    chosen_idx = int(candidate_indices[best_rel])  # 映射回全局节点索引
    
    # Top-k（用于统计 GS@3）
    k = min(topk, cand_logits.shape[0])
    topk_rel = torch.topk(cand_logits, k=k, dim=0).indices.tolist()  # 相对索引列表
    topk_indices = [int(candidate_indices[i]) for i in topk_rel]  # 映射回全局索引
    
    return chosen_idx, topk_indices

