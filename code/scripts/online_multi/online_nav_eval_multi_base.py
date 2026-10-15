"""
在线多候选导航评估基础模块

专门为多候选场景设计，支持多个候选目标车辆。
与静态单目标版本的主要区别：
1. candidate_mask 包含多个候选（2-5个）
2. target 一定是候选之一
3. 基于距离阈值筛选候选，而不是只保留target
"""

import math
import random
import os
import numpy as np
import torch
import carla
from typing import Dict, Any, Optional, List
from collections import namedtuple

# 导入基础函数
import os
import sys
_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from scripts.data.carla_collect_data import (
    build_node_features,              # ✅ 新增：复用离线构图
    build_edges,
    build_lang_feat_from_rel_type,
    classify_relation,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    LANG_FEAT_DIM,
    NUM_REL_TYPES,
)

# 从 lc_org_nav.online_nav 导入控制器函数
from lc_org_nav.online_nav import (
    setup_world,
    run_behavior_nav,
    distance_between_locations,
    _import_behavior_agent,
    _find_carla_python_api,
)

# ===== 统一协议：Episode上下文结构 =====
# 用于在整个评估流程中传递场景信息，确保协议统一
EpisodeCtx = namedtuple("EpisodeCtx", [
    "scenario_id",
    "town",
    "ego_spawn_idx",
    "anchor_spawn_idx",
    "candidate_spawn_indices",
    "target_idx_gt_scene",  # 场景JSON中的target_idx_gt（不可变）
])


def load_protocol_graph(scenario: Dict[str, Any], protocol_dir: str, device: str = "cpu") -> tuple:
    """
    给定场景 dict（含 id），从预计算 npz 中加载协议图。
    
    Args:
        scenario: 场景字典，必须包含 "id" 字段
        protocol_dir: 协议图目录路径
        device: 设备（"cpu" 或 "cuda"）
    
    Returns:
        (graph, meta) 元组：
        - graph: 包含 node_feats, edge_index, edge_attr, lang_feat, candidate_mask, target_idx_gt, rel_type
        - meta: 包含 node_spawn_indices, candidate_indices, ego_index
    """
    scen_id = scenario["id"]
    path = os.path.join(protocol_dir, f"scen_{scen_id:03d}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Protocol graph not found: {path}")

    data = np.load(path)

    graph = {
        "node_feats": torch.from_numpy(data["node_feats"]).float().to(device),
        "edge_index": torch.from_numpy(data["edge_index"]).long().to(device),
        "edge_attr": torch.from_numpy(data["edge_attr"]).float().to(device),
        "lang_feat": torch.from_numpy(data["lang_feat"]).float().to(device),
        "candidate_mask": torch.from_numpy(data["candidate_mask"]).bool().to(device),
        "target_idx_gt": int(data["target_idx_gt"]),
        "rel_type": int(data["rel_type"]),
    }

    meta = {
        "node_spawn_indices": data["node_spawn_indices"].tolist(),
        "candidate_indices": data["candidate_indices"].tolist(),
        "ego_index": int(data["ego_index"]),
    }

    return graph, meta


def get_target_vehicle_from_spawn_idx(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    npc_vehicles: List[carla.Vehicle],
    spawn_points: List[carla.Transform],
    chosen_spawn_idx: int,
    max_distance: float = 5.0,
) -> Optional[carla.Vehicle]:
    """
    从spawn_idx获取对应的车辆。
    
    优先匹配：
    1. 如果chosen_spawn_idx是ego_spawn_idx，返回None（不应该发生，因为ego不在候选集合中）
    2. 如果chosen_spawn_idx是target_spawn_idx，返回target_vehicle
    3. 否则，在npc_vehicles中查找最接近该spawn点的车辆
    
    Args:
        world: CARLA world
        ego_vehicle: ego车辆
        target_vehicle: target车辆
        npc_vehicles: NPC车辆列表
        spawn_points: 所有spawn点列表
        chosen_spawn_idx: 选择的spawn点索引
        max_distance: 最大匹配距离（米）
    
    Returns:
        对应的车辆，如果找不到则返回None
    """
    if chosen_spawn_idx >= len(spawn_points):
        print(f"  [WARNING] chosen_spawn_idx={chosen_spawn_idx} out of range (len={len(spawn_points)})")
        return None
    
    target_spawn = spawn_points[chosen_spawn_idx]
    
    # 检查是否是target_vehicle（通过位置匹配）
    if target_vehicle.is_alive:
        target_tf = target_vehicle.get_transform()
        dx = target_tf.location.x - target_spawn.location.x
        dy = target_tf.location.y - target_spawn.location.y
        dz = target_tf.location.z - target_spawn.location.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < max_distance:
            return target_vehicle
    
    # 在NPC中查找最接近的车辆
    best_vehicle = None
    best_dist = float('inf')
    
    for npc in npc_vehicles:
        if not npc.is_alive:
            continue
        npc_tf = npc.get_transform()
        dx = npc_tf.location.x - target_spawn.location.x
        dy = npc_tf.location.y - target_spawn.location.y
        dz = npc_tf.location.z - target_spawn.location.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < best_dist and dist < max_distance:
            best_dist = dist
            best_vehicle = npc
    
    if best_vehicle is None:
        print(f"  [WARNING] No vehicle found near spawn_idx={chosen_spawn_idx} (max_dist={max_distance}m)")
    
    return best_vehicle


# ❌ 旧的在线专用 node 特征构建，不再使用
# 现在统一使用离线版本的 build_node_features，确保特征计算完全一致
# def build_node_features_for_online(ego_transform, vehicles, anchor_id, target_id):
#     """
#     构建节点特征（仅返回 node_feats，不返回 candidate_mask）。
#     
#     这个函数专门用于在线评估，不负责候选过滤。
#     候选过滤逻辑在 build_graph_from_scene 中单独处理。
#     
#     Args:
#         ego_transform: ego车辆的transform
#         vehicles: 车辆列表（包括target和NPCs）
#         anchor_id: anchor车辆的ID
#         target_id: target车辆的ID（用于标记，但不泄露到特征中）
#     
#     Returns:
#         node_feats: (N, NODE_FEAT_DIM) numpy array
#     """
#     ego_x = ego_transform.location.x
#     ego_y = ego_transform.location.y
#     ego_yaw = math.radians(ego_transform.rotation.yaw)
#     
#     # 找到anchor transform
#     anchor_tf = None
#     for v in vehicles:
#         if v.id == anchor_id:
#             anchor_tf = v.get_transform()
#             break
#     if anchor_tf is None:
#         raise RuntimeError("Anchor transform not found")
#     ax, ay = anchor_tf.location.x, anchor_tf.location.y
#     ayaw = math.radians(anchor_tf.rotation.yaw)
#
#     node_feats_list = []
#
#     for veh in vehicles:
#         loc = veh.get_transform().location
#         vel = veh.get_velocity()
#
#         # Ego-relative
#         dx_ego = loc.x - ego_x
#         dy_ego = loc.y - ego_y
#         d_ego = math.sqrt(dx_ego * dx_ego + dy_ego * dy_ego)
#         global_angle_ego = math.atan2(dy_ego, dx_ego)
#         theta_ego = global_angle_ego - ego_yaw
#         while theta_ego <= -math.pi:
#             theta_ego += 2 * math.pi
#         while theta_ego > math.pi:
#             theta_ego -= 2 * math.pi
#         
#         # Anchor-relative
#         dx_a = loc.x - ax
#         dy_a = loc.y - ay
#         d_a = math.sqrt(dx_a * dx_a + dy_a * dy_a)
#         global_angle_a = math.atan2(dy_a, dx_a)
#         theta_a = global_angle_a - ayaw
#         while theta_a <= -math.pi:
#             theta_a += 2 * math.pi
#         while theta_a > math.pi:
#             theta_a -= 2 * math.pi
#
#         speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
#
#         is_anchor = 1.0 if veh.id == anchor_id else 0.0
#         # 不泄露 target
#         is_target = 0.0
#
#         # Compose features: ego-relative + anchor-relative + speed/flags
#         base_feat = np.array(
#             [
#                 # ego-relative
#                 dx_ego, dy_ego, d_ego,
#                 math.cos(theta_ego), math.sin(theta_ego),
#                 # anchor-relative
#                 dx_a, dy_a, d_a,
#                 math.cos(theta_a), math.sin(theta_a),
#                 # others
#                 speed,
#                 is_anchor,
#                 is_target,
#             ],
#             dtype=np.float32,
#         )
#
#         # Pad to NODE_FEAT_DIM
#         if base_feat.shape[0] < NODE_FEAT_DIM:
#             pad = np.zeros((NODE_FEAT_DIM - base_feat.shape[0],), dtype=np.float32)
#             feat_vec = np.concatenate([base_feat, pad], axis=0)
#         else:
#             feat_vec = base_feat[:NODE_FEAT_DIM]
#
#         node_feats_list.append(feat_vec)
#
#     node_feats = np.stack(node_feats_list, axis=0)  # (N, NODE_FEAT_DIM)
#     return node_feats


def build_graph_from_scene(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    npc_vehicles: List[carla.Vehicle],
    max_cand_dist: float = 50.0,
    scenario: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    use_near_only: bool = False,
    max_candidates: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    从当前场景构建场景图（多候选版本）。
    
    与静态单目标版本的区别：
    - candidate_mask 包含多个候选（2-5个）
    - 基于距离阈值筛选候选
    - 确保target一定是候选
    - 确保至少2个候选（否则场景对GS没有区分力）
    
    Args:
        world: CARLA world
        ego_vehicle: ego车辆
        target_vehicle: 目标车辆（GT）
        npc_vehicles: NPC车辆列表
        max_cand_dist: 候选距离阈值（米），默认50.0
        scenario: 可选的场景字典（包含candidate_spawn_indices和target_idx_gt），用于确保车辆顺序一致
        debug: 是否打印调试信息
        use_near_only: 是否只保留关系类型为"near"的候选（默认False）
        max_candidates: 最大候选数量，超过此数量时截断到最近的N个（默认None，使用原有逻辑）
    
    Returns:
        graph字典，包含：
        - node_feats: (N, NODE_FEAT_DIM) tensor
        - edge_index: (2, E) tensor
        - edge_attr: (E, EDGE_FEAT_DIM) tensor
        - lang_feat: (LANG_FEAT_DIM,) tensor
        - candidate_mask: (N,) bool tensor，包含多个候选
        - target_idx_gt: target在vehicles列表中的索引（图索引）
        - vehicles: 车辆列表（顺序：target + NPCs，或按scenario顺序）
        - rel_type: 关系类型
    """
    # 获取scenario id用于日志
    scen_id = scenario.get("id", "unknown") if scenario is not None else "unknown"
    
    # 获取所有车辆（target + NPCs）
    alive_npcs = [npc for npc in npc_vehicles if npc.is_alive]
    if len(alive_npcs) < 1:
        print(f"[GRAPH] scen={scen_id}: no alive NPCs (need >=1)")
        return None
    
    # ===== 关键修复：使用规范信息构建图结构 =====
    # 优先使用规范信息（node_spawn_indices, candidate_indices, target_index_gt）
    # 如果规范信息不存在，回退到旧的逻辑
    alive_vehicles_for_graph = None
    target_idx_gt_from_scenario = None
    canonical_candidate_indices = None
    
    if scenario is not None:
        # 检查是否有规范信息
        node_spawn_indices = scenario.get("node_spawn_indices", None)
        canonical_candidate_indices = scenario.get("candidate_indices", None)
        # 注意：规范信息中使用的是target_index_gt（不是target_idx_gt）
        canonical_target_index_gt = scenario.get("target_index_gt", None)
        
        if debug:
            print(f"  [DEBUG] Checking canonical info:")
            print(f"    node_spawn_indices: {node_spawn_indices}")
            print(f"    candidate_indices: {canonical_candidate_indices}")
            print(f"    target_index_gt: {canonical_target_index_gt}")
        
        # 如果有规范信息，使用规范信息构建图
        if node_spawn_indices is not None and len(node_spawn_indices) > 0:
            # 使用规范信息：按照node_spawn_indices的顺序构建节点列表
            spawn_points = world.get_map().get_spawn_points()
            
            # 构建车辆ID到车辆的映射
            vehicle_map = {}
            vehicle_map[target_vehicle.id] = target_vehicle
            for npc in alive_npcs:
                vehicle_map[npc.id] = npc
            
            # 按照node_spawn_indices的顺序构建车辆列表
            ordered_vehicles = []
            spawn_to_vehicle = {}  # {spawn_idx: vehicle}
            
            # 先构建spawn_idx到车辆的映射
            # ✅ 改进：使用更宽松的匹配阈值，并确保每个车辆只被一个spawn点匹配
            used_vehicles_for_matching = set()  # 在匹配阶段就防止重复
            for spawn_idx in node_spawn_indices:
                if spawn_idx >= len(spawn_points):
                    if debug:
                        print(f"  [DEBUG] Spawn idx {spawn_idx} out of range (max: {len(spawn_points)})")
                    continue
                
                spawn_point = spawn_points[spawn_idx]
                best_vehicle = None
                best_dist = float('inf')
                
                for veh_id, veh in vehicle_map.items():
                    # ✅ 关键修复：跳过已经被匹配的车辆
                    if veh.id in used_vehicles_for_matching:
                        continue
                    
                    try:
                        veh_loc = veh.get_transform().location
                        dist = math.sqrt(
                            (veh_loc.x - spawn_point.location.x) ** 2 +
                            (veh_loc.y - spawn_point.location.y) ** 2
                        )
                        # ✅ 改进：使用更宽松的阈值（15米），确保能匹配到所有车辆
                        if dist < best_dist and dist < 15.0:  # 从10米增加到15米
                            best_dist = dist
                            best_vehicle = veh
                    except:
                        continue
                
                if best_vehicle is not None:
                    spawn_to_vehicle[spawn_idx] = best_vehicle
                    used_vehicles_for_matching.add(best_vehicle.id)  # 标记为已使用
                    if debug:
                        print(f"  [DEBUG] Matched spawn {spawn_idx} to vehicle {best_vehicle.id} (dist={best_dist:.2f}m)")
                else:
                    if debug:
                        print(f"  [DEBUG] WARNING: No vehicle found for spawn {spawn_idx} (within 15m)")
            
            # 按照node_spawn_indices的顺序构建ordered_vehicles
            used_vehicles = set()
            missing_spawns = []
            for spawn_idx in node_spawn_indices:
                if spawn_idx in spawn_to_vehicle:
                    veh = spawn_to_vehicle[spawn_idx]
                    if veh.id not in used_vehicles:
                        ordered_vehicles.append(veh)
                        used_vehicles.add(veh.id)
                    elif debug:
                        print(f"  [DEBUG] Vehicle {veh.id} already used for spawn {spawn_idx}")
                else:
                    missing_spawns.append(spawn_idx)
            
            if missing_spawns and debug:
                print(f"  [DEBUG] WARNING: Missing vehicles for spawns: {missing_spawns}")
                print(f"  [DEBUG] Expected {len(node_spawn_indices)} vehicles, got {len(ordered_vehicles)}")
            
            # 如果成功构建了有序列表，使用它
            if len(ordered_vehicles) > 0:
                alive_vehicles_for_graph = ordered_vehicles
                # 使用规范信息中的target_index_gt
                if canonical_target_index_gt is not None:
                    target_idx_gt_from_scenario = canonical_target_index_gt
                else:
                    # 回退：查找target在列表中的位置
                    for i, v in enumerate(ordered_vehicles):
                        if v.id == target_vehicle.id:
                            target_idx_gt_from_scenario = i
                            break
                
                if debug or True:  # 临时强制打印，用于调试
                    print(f"  [DEBUG] Using canonical node order: {len(alive_vehicles_for_graph)} vehicles")
                    print(f"  [DEBUG] Target idx (canonical): {target_idx_gt_from_scenario}")
                    print(f"  [DEBUG] Candidate indices (canonical): {canonical_candidate_indices}")
                    print(f"  [DEBUG] Vehicle IDs in order: {[v.id for v in alive_vehicles_for_graph]}")
        
        # 如果没有规范信息，回退到旧的逻辑（使用candidate_spawn_indices）
        if alive_vehicles_for_graph is None:
            candidate_spawn_indices = scenario.get("candidate_spawn_indices", [])
            target_idx_gt_scenario = scenario.get("target_idx_gt", 0)
            
            if candidate_spawn_indices and len(candidate_spawn_indices) > 0:
                # 获取spawn点
                spawn_points = world.get_map().get_spawn_points()
                
                # 构建车辆ID到车辆的映射
                vehicle_map = {}
                vehicle_map[target_vehicle.id] = target_vehicle
                for npc in alive_npcs:
                    vehicle_map[npc.id] = npc
                
                # 按照candidate_spawn_indices的顺序构建车辆列表（排除anchor）
                # ✅ 改进：使用更可靠的匹配方法
                ordered_vehicles = []
                target_spawn_idx = candidate_spawn_indices[target_idx_gt_scenario] if target_idx_gt_scenario < len(candidate_spawn_indices) else None
                ego_spawn_idx = scenario.get("ego_spawn_idx", -1)
                anchor_spawn_idx = scenario.get("anchor_spawn_idx", ego_spawn_idx)  # V3场景：anchor_spawn_idx = ego_spawn_idx
                
                # 计算target在排除anchor后的索引
                graph_idx_counter = 0
                
                # ✅ 改进：先构建spawn_idx到车辆的映射（更可靠）
                # 通过检查每个车辆的位置，找到最接近每个spawn点的车辆
                # ✅ 关键：明确排除anchor_spawn_idx，确保anchor不在候选集合C中
                spawn_to_vehicle = {}  # {spawn_idx: vehicle}
                for spawn_idx in candidate_spawn_indices:
                    if spawn_idx == anchor_spawn_idx:
                        continue  # ✅ 跳过anchor（V3协议：anchor不在候选集合C中）
                    if spawn_idx >= len(spawn_points):
                        continue
                    
                    spawn_point = spawn_points[spawn_idx]
                    best_vehicle = None
                    best_dist = float('inf')
                    
                    for veh_id, veh in vehicle_map.items():
                        try:
                            veh_loc = veh.get_transform().location
                            dist = math.sqrt(
                                (veh_loc.x - spawn_point.location.x) ** 2 +
                                (veh_loc.y - spawn_point.location.y) ** 2
                            )
                            # ✅ 改进：使用更宽松的阈值（10米），并选择最近的车辆
                            if dist < best_dist and dist < 10.0:  # 增加到10米，更宽松
                                best_dist = dist
                                best_vehicle = veh
                        except:
                            continue
                    
                    if best_vehicle is not None:
                        spawn_to_vehicle[spawn_idx] = best_vehicle
                
                # ✅ 改进：按照candidate_spawn_indices的顺序，使用映射构建ordered_vehicles
                # ✅ 统一协议：同时维护图索引到场景索引的映射
                graph_idx_to_scene_idx = {}  # {graph_idx: scene_idx}
                used_vehicles = set()  # 避免重复使用同一车辆
                for idx_in_candidates, spawn_idx in enumerate(candidate_spawn_indices):
                    # ✅ 跳过anchor的spawn点（V3协议：anchor不在候选集合C中）
                    if spawn_idx == anchor_spawn_idx:
                        continue
                    
                    # 从映射中获取车辆
                    if spawn_idx in spawn_to_vehicle:
                        veh = spawn_to_vehicle[spawn_idx]
                        # 确保车辆未被使用（避免重复）
                        if veh.id not in used_vehicles:
                            ordered_vehicles.append(veh)
                            graph_idx_to_scene_idx[graph_idx_counter] = idx_in_candidates  # 维护映射
                            used_vehicles.add(veh.id)
                            # 如果这是target（在candidate_spawn_indices中的索引是target_idx_gt_scenario），记录它在ordered_vehicles中的索引
                            if idx_in_candidates == target_idx_gt_scenario:
                                target_idx_gt_from_scenario = graph_idx_counter
                            graph_idx_counter += 1
                        elif debug:
                            print(f"  [DEBUG] Vehicle at spawn {spawn_idx} already used, skipping")
                    elif debug:
                        print(f"  [DEBUG] No vehicle found for spawn {spawn_idx}")
                
                # 如果成功构建了有序列表，使用它
                if len(ordered_vehicles) > 0 and target_idx_gt_from_scenario is not None:
                    alive_vehicles_for_graph = ordered_vehicles
                    if debug:
                        print(f"  [DEBUG] Using scenario-ordered vehicles: {len(alive_vehicles_for_graph)} vehicles")
                        print(f"  [DEBUG] Target idx in graph: {target_idx_gt_from_scenario}")
                    # ✅ 统一协议：保存映射供后续使用
                    if scenario is not None:
                        scenario["_graph_idx_to_scene_idx"] = graph_idx_to_scene_idx
    
    # 如果无法从scenario构建有序列表，使用默认顺序
    if alive_vehicles_for_graph is None:
        alive_vehicles_for_graph = [target_vehicle] + alive_npcs
        target_idx_gt_from_scenario = 0  # target总是第一个
        # ✅ 统一协议：为默认顺序也构建映射（如果可能）
        if scenario is not None:
            candidate_spawn_indices = scenario.get("candidate_spawn_indices", [])
            target_idx_gt_scenario = scenario.get("target_idx_gt", 0)
            graph_idx_to_scene_idx = {}
            # 尝试构建映射（如果target在candidate_spawn_indices中）
            if target_idx_gt_scenario < len(candidate_spawn_indices):
                graph_idx_to_scene_idx[0] = target_idx_gt_scenario
            scenario["_graph_idx_to_scene_idx"] = graph_idx_to_scene_idx
        if debug:
            print(f"  [DEBUG] Using default vehicle order: target first, then NPCs")
    
    # ===== 关键修改：anchor统一为ego =====
    # V3场景协议：anchor_spawn_idx = ego_spawn_idx，所以anchor就是ego
    anchor = ego_vehicle
    anchor_spawn_idx = scenario.get("anchor_spawn_idx", scenario.get("ego_spawn_idx", -1)) if scenario else -1
    
    if debug:
        print(f"  [DEBUG] Anchor set to ego (spawn_idx={anchor_spawn_idx})")
    
    target = target_vehicle
    
    # 获取transforms
    try:
        ego_transform = ego_vehicle.get_transform()
        anchor_tf = ego_transform  # anchor就是ego，所以使用ego的transform
        target_tf = target.get_transform()
    except Exception as e:
        print(f"[GRAPH] scen={scen_id}: exception getting transforms: {repr(e)}")
        return None
    
    # 确定anchor和target之间的关系类型
    rel_type = None
    for rt in range(NUM_REL_TYPES):
        if classify_relation(anchor_tf, target_tf, rt):
            rel_type = rt
            break
    
    if rel_type is None:
        rel_type = 0  # 默认使用near
    
    # ✅ 关键修复：将anchor（ego）添加到vehicles列表的开头，以便build_node_features能找到它
    # 但在构建candidate_mask时，我们会明确排除anchor
    vehicles_for_node_features = [anchor] + alive_vehicles_for_graph
    # 更新target在vehicles_for_node_features中的索引（因为前面加了anchor）
    target_idx_in_node_features = None
    for i, v in enumerate(vehicles_for_node_features):
        if v.id == target.id:
            target_idx_in_node_features = i
            break
    
    try:
        # ✅ 使用离线的标准 node 特征构建逻辑（与 multi_candidate 数据完全一致）
        # 注意：vehicles_for_node_features包含anchor，但candidate_mask会排除它
        node_feats_full, cand_mask_offline_full = build_node_features(
            ego_transform,
            vehicles_for_node_features,
            anchor.id,
            target.id,
        )
        
        # ✅ 提取alive_vehicles_for_graph对应的node_feats和candidate_mask
        # node_feats_full[0]是anchor，node_feats_full[1:]是alive_vehicles_for_graph
        node_feats = node_feats_full[1:]  # 排除anchor的node features
        # candidate_mask也需要排除anchor（索引0）
        if cand_mask_offline_full is not None and len(cand_mask_offline_full) > 1:
            cand_mask_offline = cand_mask_offline_full[1:]  # 排除anchor的candidate mask
        else:
            cand_mask_offline = None
        
        # ✅ 边的构建本来就复用的是离线版本，这里保持不变
        edge_index, edge_attr = build_edges(alive_vehicles_for_graph)
    except Exception as e:
        print(f"[GRAPH] scen={scen_id}: exception building node features or edges: {repr(e)}")
        return None
    
    # 构建语言特征
    lang_feat = build_lang_feat_from_rel_type(rel_type)
    
    # 找到target在vehicles列表中的索引（使用之前计算的索引）
    target_idx = target_idx_gt_from_scenario
    if target_idx is None:
        # 回退：通过ID查找
        for i, v in enumerate(alive_vehicles_for_graph):
            if v.id == target.id:
                target_idx = i
                break
    
    if target_idx is None:
        print(f"[GRAPH] scen={scen_id}: failed to find target in vehicles list (vehicles={[v.id for v in alive_vehicles_for_graph]}, target_id={target.id})")
        return None
    
    # ===== 关键改动：使用规范信息构建 candidate_mask =====
    # 确保N在所有分支中都有定义
    N = len(alive_vehicles_for_graph)
    
    # 如果提供了规范信息，直接使用；否则使用动态筛选
    if canonical_candidate_indices is not None:
        # 使用规范信息：直接根据candidate_indices构建candidate_mask
        candidate_mask = np.zeros(N, dtype=bool)
        valid_candidate_count = 0
        for ci in canonical_candidate_indices:
            if 0 <= ci < N:
                candidate_mask[ci] = True
                valid_candidate_count += 1
            elif debug:
                print(f"  [DEBUG] WARNING: Candidate index {ci} out of range (N={N})")
        
        if debug or True:  # 临时强制打印，用于调试
            print(f"  [DEBUG] Using canonical candidate_mask: {candidate_mask}")
            print(f"  [DEBUG] Num candidates (canonical): {candidate_mask.sum()} (expected {len(canonical_candidate_indices)}, valid {valid_candidate_count})")
            print(f"  [DEBUG] Candidate indices from scenario: {canonical_candidate_indices}")
            print(f"  [DEBUG] Actual number of vehicles: {N}")
    else:
        # ✅ 没有 canonical 信息时，优先使用离线的候选筛选逻辑（与训练数据一致）
        # 离线 build_node_features 已经返回了 candidate_mask，直接使用
        if cand_mask_offline is not None and len(cand_mask_offline) == N:
            candidate_mask = cand_mask_offline.copy()
            if debug:
                print(f"  [DEBUG] Using offline candidate_mask: {candidate_mask}")
                print(f"  [DEBUG] Num candidates (offline): {candidate_mask.sum()}")
        else:
            # 兜底：如果离线 mask 不可用，回退到旧的逻辑
            if debug:
                print(f"  [DEBUG] Offline candidate_mask not available, using fallback logic")
            ego_loc = ego_transform.location
            
            # 基于ego距离构造候选集合
            cand_mask_list = []
            for veh in alive_vehicles_for_graph:
                loc = veh.get_transform().location
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                d = math.sqrt(dx * dx + dy * dy)
                # 距离阈值内的车辆都算候选（排除anchor）
                is_candidate = (d < max_cand_dist) and (veh.id != anchor.id)
                cand_mask_list.append(is_candidate)
            
            candidate_mask = np.array(cand_mask_list, dtype=bool)
            
            # 确保target一定是候选
            if not candidate_mask[target_idx]:
                candidate_mask[target_idx] = True
                if debug:
                    print(f"  [DEBUG] Forced target to be candidate (was outside threshold)")
        
        # 确保至少2个候选（否则场景对GS没有区分力）
        # ✅ V3协议：anchor不在候选集合C中，所以如果候选不足，直接返回None
        if candidate_mask.sum() < 2:
            print(f"[GRAPH] scen={scen_id}: not enough candidates: {candidate_mask.sum()} (need >=2, anchor excluded per V3 protocol)")
            return None
    
    # ===== Step 1: 只保留关系类型为"near"的候选（如果启用）=====
    if use_near_only:
        # 找到anchor在图中的索引
        anchor_idx_in_graph = None
        for i, v in enumerate(alive_vehicles_for_graph):
            if v.id == anchor.id:
                anchor_idx_in_graph = i
                break
        
        if anchor_idx_in_graph is not None:
            anchor_tf = anchor.get_transform()
            near_candidate_mask = np.zeros(N, dtype=bool)
            
            # 遍历所有候选，检查它们与anchor的关系类型
            cand_indices_before = np.nonzero(candidate_mask)[0]
            for idx in cand_indices_before:
                if idx == anchor_idx_in_graph:
                    continue  # 跳过anchor本身
                
                veh = alive_vehicles_for_graph[idx]
                veh_tf = veh.get_transform()
                
                # 检查是否为"near"关系（rel_type=0）
                is_near = classify_relation(anchor_tf, veh_tf, 0)
                if is_near:
                    near_candidate_mask[idx] = True
            
            # ✅ 统一协议：如果near候选太少（<2），按场景索引顺序补充，而不是按距离排序
            if near_candidate_mask.sum() < 2:
                if debug:
                    print(f"  [Near-Filter] Near candidates insufficient ({near_candidate_mask.sum()}), adding candidates by scene index order")
                
                # 获取图索引到场景索引的映射
                graph_idx_to_scene_idx = scenario.get("_graph_idx_to_scene_idx", {}) if scenario else {}
                
                if graph_idx_to_scene_idx:
                    # 有映射：按场景索引顺序补充
                    cand_with_scene_idx = []
                    for idx in cand_indices_before:
                        if idx == anchor_idx_in_graph:
                            continue
                        if not near_candidate_mask[idx]:  # 只考虑非near候选
                            scene_idx = graph_idx_to_scene_idx.get(idx, float('inf'))
                            cand_with_scene_idx.append((idx, scene_idx))
                    
                    # 按场景索引排序（保持JSON顺序）
                    cand_with_scene_idx.sort(key=lambda x: x[1])
                    
                    # 按顺序补充到至少2个候选
                    for idx, _ in cand_with_scene_idx:
                        if near_candidate_mask.sum() >= 2:
                            break
                        near_candidate_mask[idx] = True
                else:
                    # 没有映射：按图索引顺序补充（保持alive_vehicles_for_graph的顺序）
                    for idx in cand_indices_before:
                        if idx == anchor_idx_in_graph:
                            continue
                        if near_candidate_mask.sum() >= 2:
                            break
                        if not near_candidate_mask[idx]:
                            near_candidate_mask[idx] = True
            
            # 确保target一定是候选
            if not near_candidate_mask[target_idx]:
                near_candidate_mask[target_idx] = True
                if debug:
                    print(f"  [Near-Filter] Forced target to be candidate (was not near)")
            
            candidate_mask = near_candidate_mask
            
            if debug:
                print(f"  [Near-Filter] Filtered to near candidates: {candidate_mask.sum()} candidates")
            
            # ✅ 统一协议：检查GT候选是否还在候选集合中
            if target_idx not in np.nonzero(near_candidate_mask)[0]:
                print(f"[GRAPH] scen={scen_id}: GT candidate dropped by near-filter (target_idx={target_idx}), marking episode invalid")
                return None
            
            # 如果仍然不足2个，返回None
            if candidate_mask.sum() < 2:
                print(f"[GRAPH] scen={scen_id}: not enough candidates after near-filter: {candidate_mask.sum()} (need >=2, use_near_only={use_near_only})")
                return None
    
    # ===== Step 2: 候选截断逻辑（统一协议：按场景索引顺序截断，不改顺序）=====
    cand_indices = np.nonzero(candidate_mask)[0]
    K = len(cand_indices)
    
    # 确定截断阈值
    truncate_threshold = max_candidates if max_candidates is not None else 3
    
    # 找到anchor在图中的索引（如果还没找到）
    if 'anchor_idx_in_graph' not in locals() or anchor_idx_in_graph is None:
        anchor_idx_in_graph = None
        for i, v in enumerate(alive_vehicles_for_graph):
            if v.id == anchor.id:
                anchor_idx_in_graph = i
                break
    
    if K > truncate_threshold:
        # ✅ 统一协议：按照场景索引顺序截断，而不是按距离排序
        # 获取图索引到场景索引的映射
        graph_idx_to_scene_idx = scenario.get("_graph_idx_to_scene_idx", {}) if scenario else {}
        
        if graph_idx_to_scene_idx:
            # 有映射：按场景索引顺序截断
            cand_with_scene_idx = []
            for graph_idx in cand_indices:
                if graph_idx == anchor_idx_in_graph:
                    continue  # 排除anchor
                scene_idx = graph_idx_to_scene_idx.get(graph_idx, float('inf'))  # 没有映射的排在最后
                cand_with_scene_idx.append((graph_idx, scene_idx))
        
            # 按场景索引排序（保持JSON顺序）
            cand_with_scene_idx.sort(key=lambda x: x[1])
            
            # 保留前N个（按场景索引顺序）
            keep_indices = [graph_idx for graph_idx, _ in cand_with_scene_idx[:truncate_threshold]]
            
            if debug:
                print(f"  [Truncate] Using scene index order (protocol-compliant)")
                print(f"  [Truncate] Candidates with scene_idx: {[(g, s) for g, s in cand_with_scene_idx[:K]]}")
        else:
            # 没有映射：回退到按图索引顺序（保持alive_vehicles_for_graph的顺序）
            keep_indices = [idx for idx in cand_indices if idx != anchor_idx_in_graph][:truncate_threshold]
            
            if debug:
                print(f"  [Truncate] No scene index mapping, using graph index order (fallback)")
        
        # ✅ 统一协议：检查GT候选是否在保留的候选内
        if target_idx not in keep_indices and target_idx != anchor_idx_in_graph:
            # GT候选被截断掉，标记episode无效
            print(f"[GRAPH] scen={scen_id}: GT candidate dropped by truncation (target_idx={target_idx}, kept={keep_indices}), marking episode invalid")
            return None
        
        # 构建新的candidate_mask（明确排除anchor）
        new_candidate_mask = np.zeros(N, dtype=bool)
        for idx in keep_indices:
            if idx != anchor_idx_in_graph:  # 确保不包含anchor
                new_candidate_mask[idx] = True
        
        candidate_mask = new_candidate_mask
        
        if debug:
            print(f"  [Truncate] Candidates truncated: {K} → {candidate_mask.sum()} (threshold={truncate_threshold})")
            print(f"  [Truncate] Kept indices (excl anchor): {keep_indices}")
            print(f"  [Truncate] Anchor idx: {anchor_idx_in_graph} (excluded)")
    
    # ===== Step 1: 打印候选关系类型表（用于调试和验证）=====
    # 按照用户checklist，统计候选集合C的关系类型，而不是整图所有边
    if debug or True:  # 临时强制打印，用于排查
        print(f"  [Candidate-RelType] Episode candidate relation types:")
        print(f"    Anchor ID: {anchor.id}")
        anchor_tf = anchor.get_transform()
        anchor_loc = anchor_tf.location
        
        final_cand_indices = np.nonzero(candidate_mask)[0]
        near_count = 0
        rel_type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # near, front, behind, left, right
        
        for cand_idx in final_cand_indices:
            if cand_idx == anchor_idx_in_graph:
                continue  # 跳过anchor本身
            
            veh = alive_vehicles_for_graph[cand_idx]
            veh_tf = veh.get_transform()
            
            # 计算到anchor的距离
            veh_loc = veh_tf.location
            dx = veh_loc.x - anchor_loc.x
            dy = veh_loc.y - anchor_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            
            # 检查关系类型（按优先级：near > front > behind > left > right）
            is_near = classify_relation(anchor_tf, veh_tf, 0)
            is_front = classify_relation(anchor_tf, veh_tf, 1)
            is_behind = classify_relation(anchor_tf, veh_tf, 2)
            is_left = classify_relation(anchor_tf, veh_tf, 3)
            is_right = classify_relation(anchor_tf, veh_tf, 4)
            
            # 确定关系类型（near优先级最高）
            if is_near:
                rel_type = 0
                rel_name = "near"
                near_count += 1
                rel_type_counts[0] += 1
            elif is_front:
                rel_type = 1
                rel_name = "front"
                rel_type_counts[1] += 1
            elif is_behind:
                rel_type = 2
                rel_name = "behind"
                rel_type_counts[2] += 1
            elif is_left:
                rel_type = 3
                rel_name = "left"
                rel_type_counts[3] += 1
            elif is_right:
                rel_type = 4
                rel_name = "right"
                rel_type_counts[4] += 1
            else:
                rel_type = -1
                rel_name = "unknown"
            
            is_gt = (cand_idx == target_idx)
            print(f"    cand={cand_idx:2d} | rel={rel_name:>6} (type={rel_type}) | d_a={dist:5.2f}m | GT={is_gt}")
        
        total_cands = len(final_cand_indices)
        near_pct = near_count / total_cands * 100 if total_cands > 0 else 0.0
        print(f"  [Candidate-RelType] Near candidates: {near_count}/{total_cands} ({near_pct:.1f}%)")
        print(f"  [Candidate-RelType] Relation type distribution: {rel_type_counts}")
        print("")
    
    # ===== 调试信息 =====
    if debug:
        print(f"  [DEBUG] Graph building summary:")
        print(f"    Vehicles in graph: {len(alive_vehicles_for_graph)}")
        print(f"    Target idx (graph): {target_idx}")
        print(f"    Candidate mask: {candidate_mask}")
        print(f"    Num candidates: {candidate_mask.sum()}")
        print(f"    Vehicle IDs: {[v.id for v in alive_vehicles_for_graph]}")
        print(f"    Anchor ID: {anchor.id}")
        print(f"    Target ID: {target.id}")
    
    # ===== 统一协议：固定使用场景JSON中的target_idx_gt，不再动态重算 =====
    # ❌ 删除：不再根据运行时距离动态计算target_idx_gt
    # ✅ 固定：直接使用场景JSON中的target_idx_gt（已在target_idx中）
    
    # ✅ 统一协议：最终验证target_idx是否在候选集合中（如果不在，标记episode无效）
    final_cand_indices = np.nonzero(candidate_mask)[0]
    if target_idx not in final_cand_indices:
        print(f"[GRAPH] scen={scen_id}: GT candidate not in final candidate_mask (target_idx={target_idx}, candidates={final_cand_indices.tolist()}), marking episode invalid")
        return None
    
    if debug:
        print(f"  [GT-Fixed] Using scenario JSON target_idx_gt: {target_idx} (from scene index)")
        print(f"  [GT-Fixed] Final candidate indices: {final_cand_indices.tolist()}")
    
    # ===== 多候选构造完成 =====
    
    return {
        "node_feats": torch.from_numpy(node_feats).float(),
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_attr": torch.from_numpy(edge_attr).float(),
        "lang_feat": torch.from_numpy(lang_feat).float(),
        "candidate_mask": torch.from_numpy(candidate_mask).bool(),
        "target_idx_gt": target_idx,  # ✅ 统一协议：固定使用场景JSON中的target_idx_gt
        "vehicles": alive_vehicles_for_graph,
        "rel_type": rel_type,
    }


# 导入目标选择函数（这些函数应该可以直接复用）
from lc_org_nav.online_nav.target_select import (
    select_target_oracle,
    select_target_gs,
    select_target_random,
)

# 导入模型加载函数
from lc_org_nav.online_nav.gs_loader import load_gs_model

def reposition_npcs_for_scenario(
    world: carla.World,
    npc_vehicles: List[carla.Vehicle],
    scenario: Dict[str, Any],
    spawn_points: List[carla.Transform],
    ego_spawn_idx: int,
    target_spawn_idx: int,
) -> None:
    """
    根据场景的candidate_spawn_indices重新定位NPC到正确位置。
    
    Args:
        world: CARLA world
        npc_vehicles: NPC车辆列表
        scenario: 场景字典（包含candidate_spawn_indices）
        spawn_points: 所有spawn点列表
        ego_spawn_idx: ego的spawn点索引
        target_spawn_idx: target的spawn点索引
    """
    import random
    
    candidate_spawn_indices = scenario.get("candidate_spawn_indices", [])
    seed = scenario.get("seed", 42)
    
    zero_control = carla.VehicleControl(
        throttle=0.0, steer=0.0, brake=1.0,
        hand_brake=False, reverse=False, manual_gear_shift=False
    )
    
    if candidate_spawn_indices:
        # 从候选spawn点中排除ego和target
        npc_spawn_indices = [idx for idx in candidate_spawn_indices 
                             if idx != ego_spawn_idx and idx != target_spawn_idx]
        # 确保至少有2个NPC spawn点（如果不足，从其他spawn点补充）
        if len(npc_spawn_indices) < 2:
            available_indices = [i for i in range(len(spawn_points)) 
                               if i != ego_spawn_idx and i != target_spawn_idx 
                               and i not in npc_spawn_indices]
            random.seed(seed)
            while len(npc_spawn_indices) < 2 and available_indices:
                npc_spawn_indices.append(available_indices.pop(random.randint(0, len(available_indices)-1)))
        
        # 限制NPC数量（最多5个，避免过多）
        npc_spawn_indices = npc_spawn_indices[:min(5, len(npc_spawn_indices))]
        
        # 确保有足够的NPC车辆
        # ✅ 使用场景seed确保一致性
        random.seed(seed)
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")
        while len(npc_vehicles) < len(npc_spawn_indices):
            if vehicle_bps:
                npc_bp = random.choice(vehicle_bps)
                if npc_bp.has_attribute("color"):
                    color = random.choice(npc_bp.get_attribute("color").recommended_values)
                    npc_bp.set_attribute("color", color)
                # 在第一个可用的spawn点spawn
                available_spawn = None
                for idx in range(len(spawn_points)):
                    if idx not in [ego_spawn_idx, target_spawn_idx]:
                        available_spawn = spawn_points[idx]
                        break
                if available_spawn:
                    npc = world.try_spawn_actor(npc_bp, available_spawn)
                    if npc is not None:
                        npc_vehicles.append(npc)
        
        # 重新定位NPC到候选spawn点
        for i, npc_spawn_idx in enumerate(npc_spawn_indices):
            if npc_spawn_idx >= len(spawn_points):
                continue
            if i < len(npc_vehicles) and npc_vehicles[i].is_alive:
                try:
                    npc_vehicles[i].set_autopilot(False)
                    npc_vehicles[i].set_transform(spawn_points[npc_spawn_idx])
                    npc_vehicles[i].apply_control(zero_control)
                except Exception as e:
                    print(f"  Warning: Failed to reposition NPC {i}: {e}")
        # 停止多余的NPC（如果NPC数量多于需要的）
        for i in range(len(npc_spawn_indices), len(npc_vehicles)):
            if npc_vehicles[i].is_alive:
                try:
                    npc_vehicles[i].set_autopilot(False)
                    npc_vehicles[i].apply_control(zero_control)
                except:
                    pass
    else:
        # 如果没有candidate_spawn_indices，使用原来的逻辑
        for npc in npc_vehicles:
            if npc.is_alive:
                try:
                    npc.set_autopilot(False)
                    npc.apply_control(zero_control)
                except:
                    pass


# 导出所有需要的函数
__all__ = [
    # Controller
    "setup_world",
    "reposition_npcs_for_scenario",
    "run_behavior_nav",
    "distance_between_locations",
    "_import_behavior_agent",
    "_find_carla_python_api",
    # Graph builder (multi-candidate version)
    "build_graph_from_scene",
    # "build_node_features_for_online",  # ❌ 已废弃，现在使用离线的 build_node_features
    # Target selection
    "select_target_oracle",
    "select_target_gs",
    "select_target_random",
    # Model loader
    "load_gs_model",
]

