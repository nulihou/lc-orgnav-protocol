#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何工具函数：统一的相对坐标系计算
"""

import math
from typing import Tuple
import carla


def get_rel_xy(ego_transform: carla.Transform, target_transform: carla.Transform) -> Tuple[float, float]:
    """
    计算ego相对于target的局部坐标 (x_rel, y_rel)
    
    坐标系定义：
    - 原点：目标车中心
    - x轴：指向车后方（与目标车前进方向相反）
    - y轴：指向目标车右侧
    
    返回值：
    - x_rel: 纵向距离，>0表示ego在目标车后方，<0表示在前方
    - y_rel: 横向距离，>0表示ego在目标车右侧，<0表示在左侧
    """
    # 世界坐标
    ex, ey = ego_transform.location.x, ego_transform.location.y
    tx, ty = target_transform.location.x, target_transform.location.y
    
    # 目标车朝向（CARLA yaw是度）
    yaw_deg = target_transform.rotation.yaw
    yaw = math.radians(yaw_deg)
    
    # 目标车"前进方向"单位向量
    fx = math.cos(yaw)
    fy = math.sin(yaw)
    
    # 我们定义坐标系：
    #   x轴：车屁股方向（即"前进方向的反方向"）
    #   y轴：车右侧方向
    bx = -fx  # backward = -forward
    by = -fy
    rx = math.cos(yaw - math.pi/2.0)  # 右侧方向
    ry = math.sin(yaw - math.pi/2.0)
    
    # 世界坐标差向量：目标 -> ego
    dx = ex - tx
    dy = ey - ty
    
    # 投影到局部坐标系
    x_rel = dx * bx + dy * by  # 沿"车后"方向的距离（>0在后面）
    y_rel = dx * rx + dy * ry  # 向右为正
    
    return x_rel, y_rel


def make_back_goal_from_target(
    target_transform: carla.Transform,
    carla_map: carla.Map,
    back_dist: float,
    lat_offset: float = 0.0
) -> carla.Transform:
    """
    从目标车位置生成一个"后方N米"的目标点，并snap到地图上的合法车道
    
    Args:
        target_transform: 目标车的Transform
        carla_map: CARLA地图对象
        back_dist: 目标车后方的距离（米）
        lat_offset: 横向偏移（米），>0表示在目标车右侧
    
    Returns:
        目标点的Transform（已snap到合法车道）
    """
    loc = target_transform.location
    yaw_deg = target_transform.rotation.yaw
    yaw = math.radians(yaw_deg)
    
    # 目标车前进方向
    fx = math.cos(yaw)
    fy = math.sin(yaw)
    # 后退方向
    bx = -fx
    by = -fy
    # 右侧方向
    rx = math.cos(yaw - math.pi/2.0)
    ry = math.sin(yaw - math.pi/2.0)
    
    # 先在世界坐标里按几何偏移
    goal_x = loc.x + back_dist * bx + lat_offset * rx
    goal_y = loc.y + back_dist * by + lat_offset * ry
    goal_z = loc.z
    
    rough_goal = carla.Location(goal_x, goal_y, goal_z)
    
    # 再snap到地图上的合法driving lane
    wp = carla_map.get_waypoint(
        rough_goal,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    return wp.transform

