#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一载入协议图的工具

封装一个函数：
def load_protocol_graph(scen_id: int, root_dir: str):
    # 返回 dict：node_feats, edge_index, edge_attr,
    #           candidate_mask, target_idx_gt, etc.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def load_protocol_graph(
    scen_id: int,
    root_dir: str,
    return_torch: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    加载预计算的协议图
    
    Args:
        scen_id: 场景ID
        root_dir: 协议图根目录（通常是 scripts/online_multi/protocol_graphs）
        return_torch: 是否返回torch tensor（默认False，返回numpy）
    
    Returns:
        graph字典，包含：
        - node_feats: (N, NODE_FEAT_DIM) array
        - edge_index: (2, E) array
        - edge_attr: (E, EDGE_FEAT_DIM) array
        - lang_feat: (LANG_FEAT_DIM,) array
        - candidate_mask: (N,) bool array
        - target_idx_gt: int
        - candidate_indices: list of int
        - num_candidates: int
        - scen_id: int
        - node_spawn_indices: (可选) list of int
        - canonical_candidate_indices: (可选) list of int
        - canonical_target_index_gt: (可选) int
    """
    root_dir = Path(root_dir)
    graph_file = root_dir / f"scen_{scen_id:03d}.npz"
    
    if not graph_file.exists():
        print(f"[ERROR] Protocol graph not found: {graph_file}")
        return None
    
    try:
        data = np.load(str(graph_file), allow_pickle=False)
        
        graph = {}
        
        # 核心字段
        if "node_feats" in data:
            graph["node_feats"] = data["node_feats"]
        else:
            print(f"[ERROR] Missing node_feats in {graph_file}")
            return None
        
        if "edge_index" in data:
            graph["edge_index"] = data["edge_index"]
        else:
            print(f"[ERROR] Missing edge_index in {graph_file}")
            return None
        
        if "edge_attr" in data:
            graph["edge_attr"] = data["edge_attr"]
        else:
            print(f"[ERROR] Missing edge_attr in {graph_file}")
            return None
        
        if "lang_feat" in data:
            graph["lang_feat"] = data["lang_feat"]
        else:
            print(f"[ERROR] Missing lang_feat in {graph_file}")
            return None
        
        if "candidate_mask" in data:
            graph["candidate_mask"] = data["candidate_mask"]
        else:
            print(f"[ERROR] Missing candidate_mask in {graph_file}")
            return None
        
        if "target_idx_gt" in data:
            graph["target_idx_gt"] = int(data["target_idx_gt"])
        else:
            print(f"[ERROR] Missing target_idx_gt in {graph_file}")
            return None
        
        # 可选字段
        if "candidate_indices" in data:
            graph["candidate_indices"] = data["candidate_indices"].tolist()
        else:
            # 从candidate_mask计算
            graph["candidate_indices"] = np.nonzero(graph["candidate_mask"])[0].tolist()
        
        if "num_candidates" in data:
            graph["num_candidates"] = int(data["num_candidates"])
        else:
            graph["num_candidates"] = len(graph["candidate_indices"])
        
        if "scen_id" in data:
            graph["scen_id"] = int(data["scen_id"])
        else:
            graph["scen_id"] = scen_id
        
        # 协议信息（可选）
        if "node_spawn_indices" in data:
            graph["node_spawn_indices"] = data["node_spawn_indices"].tolist()
        
        if "canonical_candidate_indices" in data:
            graph["canonical_candidate_indices"] = data["canonical_candidate_indices"].tolist()
        
        if "canonical_target_index_gt" in data:
            graph["canonical_target_index_gt"] = int(data["canonical_target_index_gt"])
        
        # 转换为torch tensor（如果需要）
        if return_torch:
            graph["node_feats"] = torch.from_numpy(graph["node_feats"]).float()
            graph["edge_index"] = torch.from_numpy(graph["edge_index"]).long()
            graph["edge_attr"] = torch.from_numpy(graph["edge_attr"]).float()
            graph["lang_feat"] = torch.from_numpy(graph["lang_feat"]).float()
            graph["candidate_mask"] = torch.from_numpy(graph["candidate_mask"]).bool()
        
        return graph
        
    except Exception as e:
        print(f"[ERROR] Failed to load protocol graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_all_protocol_graphs(
    root_dir: str,
    max_scen_id: Optional[int] = None,
    return_torch: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    加载所有可用的协议图
    
    Args:
        root_dir: 协议图根目录
        max_scen_id: 最大场景ID（用于限制加载范围）
        return_torch: 是否返回torch tensor
    
    Returns:
        {scen_id: graph_dict} 字典
    """
    root_dir = Path(root_dir)
    graphs = {}
    
    if max_scen_id is None:
        # 自动检测所有可用的图
        for graph_file in root_dir.glob("scen_*.npz"):
            scen_id_str = graph_file.stem.replace("scen_", "")
            try:
                scen_id = int(scen_id_str)
                graph = load_protocol_graph(scen_id, root_dir, return_torch=return_torch)
                if graph is not None:
                    graphs[scen_id] = graph
            except ValueError:
                continue
    else:
        # 加载指定范围内的图
        for scen_id in range(max_scen_id + 1):
            graph = load_protocol_graph(scen_id, root_dir, return_torch=return_torch)
            if graph is not None:
                graphs[scen_id] = graph
    
    return graphs

