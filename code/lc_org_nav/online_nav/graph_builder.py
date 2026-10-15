"""
场景图构建模块

从 CARLA 场景构建场景图，用于目标选择。
"""

import math
import random
import numpy as np
import torch
import carla
from typing import Dict, Any, Optional, List

# 从 carla_collect_data 导入这些函数和常量
# 这些函数应该保留在 scripts/data/carla_collect_data.py 中
# 或者可以移动到 lc_org_nav/online_nav/ 中
import os
import sys
# 添加 scripts 目录到路径
_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from scripts.data.carla_collect_data import (
    build_node_features,
    build_edges,
    build_lang_feat_from_rel_type,
    classify_relation,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    LANG_FEAT_DIM,
    NUM_REL_TYPES,
)


def build_graph_from_scene(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    npc_vehicles: List[carla.Vehicle],
) -> Optional[Dict[str, Any]]:
    """
    从当前场景构建场景图。
    
    所有模式（Oracle/GS/Random）都使用相同的构建逻辑。
    """
    # 获取所有车辆（target + NPCs）
    alive_npcs = [npc for npc in npc_vehicles if npc.is_alive]
    if len(alive_npcs) < 1:
        return None
    
    # 选择anchor（从alive NPCs中随机选）
    anchor = random.choice(alive_npcs)
    target = target_vehicle
    
    # 获取transforms
    try:
        ego_transform = ego_vehicle.get_transform()
        anchor_tf = anchor.get_transform()
        target_tf = target.get_transform()
    except Exception as e:
        return None
    
    # 确定anchor和target之间的关系类型
    rel_type = None
    for rt in range(NUM_REL_TYPES):
        if classify_relation(anchor_tf, target_tf, rt):
            rel_type = rt
            break
    
    if rel_type is None:
        rel_type = 0  # 默认使用near
    
    # 构建节点特征（所有车辆除了ego）
    alive_vehicles_for_graph = [target_vehicle] + alive_npcs
    try:
        node_feats, candidate_mask = build_node_features(
            ego_transform, alive_vehicles_for_graph, anchor.id, target.id
        )
        
        # 构建边
        edge_index, edge_attr = build_edges(alive_vehicles_for_graph)
    except Exception as e:
        return None
    
    # 构建语言特征
    lang_feat = build_lang_feat_from_rel_type(rel_type)
    
    # 找到target在vehicles列表中的索引
    target_idx = None
    for i, v in enumerate(alive_vehicles_for_graph):
        if v.id == target.id:
            target_idx = i
            break
    
    if target_idx is None:
        return None
    
    return {
        "node_feats": torch.from_numpy(node_feats).float(),
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_attr": torch.from_numpy(edge_attr).float(),
        "lang_feat": torch.from_numpy(lang_feat).float(),
        "candidate_mask": torch.from_numpy(candidate_mask).bool(),
        "target_idx_gt": target_idx,
        "vehicles": alive_vehicles_for_graph,
        "rel_type": rel_type,
    }

