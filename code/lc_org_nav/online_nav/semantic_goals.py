"""
语义子目标管理模块

功能：根据目标车辆生成三个语义子目标（G1/G2/G3）
- G1: 远距离接近点（车后25m）
- G2: 中距离接近点（车后10m）
- G3: 停靠点（车后3m + 横向偏移0.0m，匹配NavSucc@4m的<4m判定）
"""

import math
import carla
from typing import List, Tuple, Optional
from dataclasses import dataclass
from lc_org_nav.online_nav.geometry_utils import make_back_goal_from_target


@dataclass
class SemanticGoal:
    """语义子目标数据结构"""
    transform: carla.Transform
    goal_type: str  # "G1", "G2", "G3"
    max_steps: int  # 该阶段最大步数
    target_speed: float  # 该阶段目标速度 (m/s)


class SemanticGoalManager:
    """语义子目标管理器"""
    
    def __init__(
        self,
        carla_map: carla.Map,
        g1_distance: float = 25.0,
        g2_distance: float = 9.0,   # 从10.0收紧到9.0，让进入G3的位置更集中
        g3_distance: float = 3.0,
        g3_lateral_offset: float = 0.0,
        g1_max_steps: int = 500,
        g2_max_steps: int = 300,
        g3_max_steps: int = 200,
        g1_target_speed: float = 11.0,  # m/s
        g2_target_speed: float = 7.0,
        g3_target_speed: float = 2.0,
    ):
        """
        初始化语义子目标管理器
        
        Args:
            carla_map: CARLA地图对象
            g1_distance: G1距离目标车的距离（米）
            g2_distance: G2距离目标车的距离（米）
            g3_distance: G3距离目标车的距离（米）
            g3_lateral_offset: G3的横向偏移（米，正值表示右侧）
            g1_max_steps: G1阶段最大步数
            g2_max_steps: G2阶段最大步数
            g3_max_steps: G3阶段最大步数
            g1_target_speed: G1阶段目标速度
            g2_target_speed: G2阶段目标速度
            g3_target_speed: G3阶段目标速度
        """
        self.carla_map = carla_map
        self.g1_distance = g1_distance
        self.g2_distance = g2_distance
        self.g3_distance = g3_distance
        self.g3_lateral_offset = g3_lateral_offset
        self.g1_max_steps = g1_max_steps
        self.g2_max_steps = g2_max_steps
        self.g3_max_steps = g3_max_steps
        self.g1_target_speed = g1_target_speed
        self.g2_target_speed = g2_target_speed
        self.g3_target_speed = g3_target_speed
    
    def _backward_waypoint(
        self,
        start_wp: carla.Waypoint,
        distance: float,
        step_size: float = 2.0
    ) -> Optional[carla.Waypoint]:
        """
        沿着车道向后查找指定距离的waypoint
        
        Args:
            start_wp: 起始waypoint
            distance: 要后退的距离（米）
            step_size: 每次后退的步长（米）
        
        Returns:
            目标waypoint，如果失败返回None
        """
        current_wp = start_wp
        remaining = distance
        
        while remaining > 0:
            step = min(step_size, remaining)
            previous_wps = current_wp.previous(step)
            
            if not previous_wps:
                # 如果previous返回空，尝试使用next作为兜底
                next_wps = current_wp.next(step)
                if next_wps:
                    current_wp = next_wps[0]
                else:
                    # 如果都失败，返回当前waypoint
                    return current_wp
            else:
                # 选择第一个（最近的）previous waypoint
                current_wp = previous_wps[0]
            
            remaining -= step
        
        return current_wp
    
    def _lateral_offset(
        self,
        waypoint: carla.Waypoint,
        offset: float
    ) -> carla.Transform:
        """
        对waypoint进行横向偏移
        
        Args:
            waypoint: 原始waypoint
            offset: 横向偏移量（米），正值表示右侧，负值表示左侧
        
        Returns:
            偏移后的Transform
        """
        loc = waypoint.transform.location
        yaw_rad = math.radians(waypoint.transform.rotation.yaw)
        
        # 计算横向偏移方向（垂直于车道方向）
        # 车道方向是yaw，横向是yaw + 90度
        dx = offset * math.cos(yaw_rad + math.pi / 2)
        dy = offset * math.sin(yaw_rad + math.pi / 2)
        
        new_loc = carla.Location(
            x=loc.x + dx,
            y=loc.y + dy,
            z=loc.z
        )
        
        return carla.Transform(new_loc, waypoint.transform.rotation)
    
    def generate_goals(
        self,
        target_vehicle: carla.Vehicle
    ) -> List[SemanticGoal]:
        """
        为目标车辆生成三个语义子目标
        
        使用纯几何方法 + map snap，不再依赖waypoint.previous()
        这样可以确保G1/G2/G3一定在目标车后方
        
        Args:
            target_vehicle: 目标车辆Actor
        
        Returns:
            三个语义子目标的列表 [G1, G2, G3]
        """
        target_transform = target_vehicle.get_transform()
        
        # 使用几何方法生成G1/G2/G3，然后snap到地图上的合法车道
        # 这样可以避免waypoint.previous()在多车道/支路环境下的问题
        
        # 1. 生成G1（远距离接近点，车后25m）
        g1_transform = make_back_goal_from_target(
            target_transform,
            self.carla_map,
            back_dist=self.g1_distance,
            lat_offset=0.0
        )
        
        goal_g1 = SemanticGoal(
            transform=g1_transform,
            goal_type="G1",
            max_steps=self.g1_max_steps,
            target_speed=self.g1_target_speed
        )
        
        # 2. 生成G2（中距离接近点，车后9m，横向偏移与G3对齐）
        g2_transform = make_back_goal_from_target(
            target_transform,
            self.carla_map,
            back_dist=self.g2_distance,
            lat_offset=self.g3_lateral_offset  # 与G3的横向偏移对齐
        )
        
        goal_g2 = SemanticGoal(
            transform=g2_transform,
            goal_type="G2",
            max_steps=self.g2_max_steps,
            target_speed=self.g2_target_speed
        )
        
        # 3. 生成G3（停靠点，车后3m + 横向偏移0.0m）
        g3_transform = make_back_goal_from_target(
            target_transform,
            self.carla_map,
            back_dist=self.g3_distance,
            lat_offset=self.g3_lateral_offset
        )
        
        goal_g3 = SemanticGoal(
            transform=g3_transform,
            goal_type="G3",
            max_steps=self.g3_max_steps,
            target_speed=self.g3_target_speed
        )
        
        return [goal_g1, goal_g2, goal_g3]
    
    def get_goal_waypoint(
        self,
        goal: SemanticGoal
    ) -> carla.Waypoint:
        """
        获取语义子目标对应的waypoint
        
        Args:
            goal: 语义子目标
        
        Returns:
            对应的waypoint
        """
        wp = self.carla_map.get_waypoint(
            goal.transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if wp is None:
            # 如果找不到，尝试使用目标位置附近的waypoint
            wp = self.carla_map.get_waypoint(
                goal.transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Any
            )
        
        return wp

