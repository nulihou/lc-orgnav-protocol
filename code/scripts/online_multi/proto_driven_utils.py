#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协议驱动在线评估工具函数

提供从协议图索引映射到CARLA车辆的函数。
"""

import numpy as np
import carla
from typing import Dict, Any, List, Optional
import math


def graph_index_to_actor(
    chosen_idx_graph: int,
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    npc_vehicles: List[carla.Vehicle],
    protocol_graph: Dict[str, Any],
    scenario: Dict[str, Any],
    max_distance: float = 10.0,  # 增加到10米，更宽松的匹配
) -> Optional[carla.Vehicle]:
    """
    将协议图中的chosen_idx_graph映射到CARLA中的具体车辆actor
    
    Args:
        chosen_idx_graph: 协议图中选择的节点索引
        world: CARLA world
        ego_vehicle: ego车辆
        target_vehicle: target车辆
        npc_vehicles: NPC车辆列表
        protocol_graph: 协议图字典（包含node_spawn_indices等）
        scenario: 场景字典
        max_distance: 最大匹配距离（米）
    
    Returns:
        对应的车辆actor，如果找不到则返回None
    """
    node_spawn_indices = protocol_graph.get("node_spawn_indices", [])
    
    if chosen_idx_graph < 0 or chosen_idx_graph >= len(node_spawn_indices):
        print(f"  [ERROR] chosen_idx_graph={chosen_idx_graph} out of range (len={len(node_spawn_indices)})")
        return None
    
    chosen_spawn_idx = node_spawn_indices[chosen_idx_graph]
    spawn_points = world.get_map().get_spawn_points()
    
    if chosen_spawn_idx >= len(spawn_points):
        print(f"  [ERROR] chosen_spawn_idx={chosen_spawn_idx} out of range (len={len(spawn_points)})")
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


def setup_world_for_protocol_scenario(
    world: carla.World,
    scenario: Dict[str, Any],
    protocol_graph: Dict[str, Any],
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    npc_vehicles: List[carla.Vehicle],
) -> None:
    """
    根据协议图和场景配置，在CARLA中设置车辆位置
    
    Args:
        world: CARLA world
        scenario: 场景字典
        protocol_graph: 协议图字典（包含node_spawn_indices）
        ego_vehicle: ego车辆
        target_vehicle: target车辆
        npc_vehicles: NPC车辆列表
    """
    from scripts.online_multi.online_nav_eval_multi_base import reposition_npcs_for_scenario
    
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_idx = scenario["ego_spawn_idx"]
    target_spawn_idx = scenario["target_spawn_idx"]
    
    # 重置NPC位置（基于场景配置）
    reposition_npcs_for_scenario(
        world, npc_vehicles, scenario, spawn_points, ego_spawn_idx, target_spawn_idx
    )
    
    # 等待世界更新
    for _ in range(30):
        world.tick()
    
    # 重置ego位置
    if ego_vehicle.is_alive:
        ego_spawn = spawn_points[ego_spawn_idx]
        ego_vehicle.set_transform(ego_spawn)
        zero_control = carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=1.0,
            hand_brake=False, reverse=False, manual_gear_shift=False
        )
        ego_vehicle.apply_control(zero_control)
    
    # 等待
    for _ in range(30):
        world.tick()
    
    # 重置target位置
    if target_vehicle.is_alive:
        target_spawn = spawn_points[target_spawn_idx]
        target_vehicle.set_transform(target_spawn)
        target_vehicle.set_autopilot(False)
        target_vehicle.apply_control(zero_control)
    
    # 等待稳定
    for _ in range(40):
        world.tick()

