"""
目标车专用路线生成模块

核心功能：生成"目标车所在车道后方8-15m的接近点"，而不是直接使用目标车的当前位置。
这样可以让BehaviorAgent更容易到达目标，提高成功率。
"""

import math
import random
import carla
from typing import Optional, Tuple, List


def generate_approach_waypoint(
    world: carla.World,
    target_vehicle: carla.Vehicle,
    back_distance: float = 10.0,
    random_offset: float = 2.0,
) -> Optional[carla.Waypoint]:
    """
    生成目标车所在车道后方的接近点waypoint
    
    Args:
        world: CARLA世界对象
        target_vehicle: 目标车辆Actor
        back_distance: 后退距离（米），默认10米
        random_offset: 随机扰动范围（米），默认±2米，防止总是同一个点
    
    Returns:
        接近点的waypoint，如果失败返回None
    """
    try:
        # 1. 获取目标车辆的位置和朝向
        target_transform = target_vehicle.get_transform()
        target_location = target_transform.location
        target_yaw_deg = target_transform.rotation.yaw  # 度数
        
        # 2. 在CARLA地图上找到目标车所在的waypoint
        carla_map = world.get_map()
        target_waypoint = carla_map.get_waypoint(
            target_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if target_waypoint is None:
            print(f"[Warning] Could not find waypoint for target vehicle at {target_location}")
            return None
        
        # 3. 沿着反方向（车后方）退一段距离
        # 添加随机扰动，防止总是同一个点
        actual_back_distance = back_distance + random.uniform(-random_offset, random_offset)
        actual_back_distance = max(5.0, actual_back_distance)  # 至少5米
        
        # 使用previous方法沿lane backward查找
        # previous(distance)返回距离内所有waypoint的列表
        # 我们需要找到距离约actual_back_distance的waypoint
        previous_waypoints = target_waypoint.previous(actual_back_distance + 5.0)  # 多找一些，确保能找到
        
        if not previous_waypoints:
            # 如果previous返回空，尝试使用next（向前）或原地
            # 按照用户建议：万一previous返回空，就向前找一个点兜底
            next_waypoints = target_waypoint.next(actual_back_distance)
            if next_waypoints:
                # 使用第一个（最近的）forward waypoint
                approach_waypoint = next_waypoints[0]
                print(f"[Warning] previous() returned empty, using next() waypoint as fallback")
            else:
                # 实在不行就用原地
                approach_waypoint = target_waypoint
                print(f"[Warning] Both previous() and next() returned empty, using target waypoint itself")
            
            return approach_waypoint
        
        # 从previous_waypoints中找到最接近目标距离的waypoint
        # previous返回的waypoint列表是从近到远的
        approach_waypoint = None
        min_dist_diff = float('inf')
        
        for wp in previous_waypoints:
            wp_location = wp.transform.location
            dist_to_target = math.sqrt(
                (wp_location.x - target_location.x)**2 +
                (wp_location.y - target_location.y)**2
            )
            dist_diff = abs(dist_to_target - actual_back_distance)
            if dist_diff < min_dist_diff:
                min_dist_diff = dist_diff
                approach_waypoint = wp
        
        # 如果没找到合适的，使用最后一个（最远的）
        if approach_waypoint is None:
            approach_waypoint = previous_waypoints[-1]
        
        return approach_waypoint
        
    except Exception as e:
        print(f"[Error] Failed to generate approach waypoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_approach_location(
    world: carla.World,
    target_vehicle: carla.Vehicle,
    back_distance: float = 10.0,
    random_offset: float = 2.0,
) -> Optional[carla.Location]:
    """
    生成目标车所在车道后方的接近点位置
    
    Args:
        world: CARLA世界对象
        target_vehicle: 目标车辆Actor
        back_distance: 后退距离（米），默认10米
        random_offset: 随机扰动范围（米），默认±2米
    
    Returns:
        接近点的Location，如果失败返回None
    """
    approach_waypoint = generate_approach_waypoint(
        world, target_vehicle, back_distance, random_offset
    )
    
    if approach_waypoint is None:
        return None
    
    return approach_waypoint.transform.location


def generate_route_to_target(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    back_distance: float = 10.0,
    random_offset: float = 2.0,
    use_global_planner: bool = True,
) -> Optional[Tuple[carla.Location, List[carla.Waypoint]]]:
    """
    生成从ego到目标车后方的完整路线
    
    Args:
        world: CARLA世界对象
        ego_vehicle: Ego车辆
        target_vehicle: 目标车辆
        back_distance: 后退距离（米）
        random_offset: 随机扰动范围（米）
        use_global_planner: 是否使用GlobalRoutePlanner生成完整路线
    
    Returns:
        (approach_location, route_waypoints) 或 None
        - approach_location: 接近点位置
        - route_waypoints: 从ego到接近点的waypoint列表（如果use_global_planner=True）
    """
    # 生成接近点
    approach_waypoint = generate_approach_waypoint(
        world, target_vehicle, back_distance, random_offset
    )
    
    if approach_waypoint is None:
        return None
    
    approach_location = approach_waypoint.transform.location
    
    # 如果不需要全局路径规划，只返回接近点
    if not use_global_planner:
        return (approach_location, [approach_waypoint])
    
    # 使用GlobalRoutePlanner生成完整路线
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
        
        carla_map = world.get_map()
        
        # 创建GlobalRoutePlanner
        dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution=2.0)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        
        # 获取ego位置
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        
        # 生成路线
        route = grp.trace_route(ego_location, approach_location)
        
        if not route:
            print(f"[Warning] GlobalRoutePlanner returned empty route")
            return (approach_location, [approach_waypoint])
        
        # 提取waypoint列表
        route_waypoints = [waypoint for waypoint, _ in route]
        
        print(f"[Info] Generated route with {len(route_waypoints)} waypoints")
        print(f"[Info] Approach location: ({approach_location.x:.2f}, {approach_location.y:.2f}, {approach_location.z:.2f})")
        print(f"[Info] Back distance: {back_distance:.2f}m (with {random_offset:.2f}m random offset)")
        
        return (approach_location, route_waypoints)
        
    except ImportError:
        print(f"[Warning] Could not import GlobalRoutePlanner, using simple approach")
        return (approach_location, [approach_waypoint])
    except Exception as e:
        print(f"[Warning] Failed to generate route with GlobalRoutePlanner: {e}")
        return (approach_location, [approach_waypoint])


def get_target_vehicle_location(target_vehicle: carla.Vehicle) -> carla.Location:
    """获取目标车辆的当前位置（用于距离计算）"""
    return target_vehicle.get_transform().location

