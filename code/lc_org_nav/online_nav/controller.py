"""
在线导航控制器模块

提供 BehaviorAgent 封装和导航执行功能。
"""

import os
import sys
import math
import carla
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.navigation.behavior_agent import BehaviorAgent


def _find_carla_python_api() -> Optional[str]:
    """
    Try to find CARLA Python API path.
    Common locations:
    - CARLA_ROOT/PythonAPI/carla/dist/carla-X.X.X-py3.X-linux-x86_64.egg
    - Environment variable CARLA_ROOT
    """
    # Check environment variable
    carla_root = os.environ.get("CARLA_ROOT")
    if carla_root:
        python_api_path = os.path.join(carla_root, "PythonAPI", "carla")
        if os.path.exists(python_api_path):
            return python_api_path
    
    # Check common CARLA installation paths
    common_paths = [
        "C:/CARLA/PythonAPI/carla",
        "D:/CARLA/PythonAPI/carla",
        "E:/carla/WindowsNoEditor/PythonAPI/carla",
        "/opt/carla-simulator/PythonAPI/carla",
        os.path.expanduser("~/CARLA_0.9.X/PythonAPI/carla"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def _import_behavior_agent(carla_python_api_path: Optional[str] = None):
    """Import BehaviorAgent with proper path handling."""
    # Try direct import first
    try:
        from agents.navigation.behavior_agent import BehaviorAgent
        return BehaviorAgent
    except ImportError:
        pass
    
    try:
        from carla.agents.navigation.behavior_agent import BehaviorAgent
        return BehaviorAgent
    except ImportError:
        pass
    
    # Try to find and add CARLA Python API to path
    if carla_python_api_path is None:
        carla_python_api_path = _find_carla_python_api()
    
    if carla_python_api_path:
        pythonapi_path = os.path.dirname(carla_python_api_path)
        if pythonapi_path not in sys.path:
            sys.path.insert(0, pythonapi_path)
        
        # Also try adding the carla subdirectory
        carla_agents_path = os.path.join(carla_python_api_path, "carla")
        if os.path.exists(carla_agents_path) and carla_agents_path not in sys.path:
            sys.path.insert(0, carla_agents_path)
        
        try:
            from agents.navigation.behavior_agent import BehaviorAgent
            return BehaviorAgent
        except ImportError:
            pass
    
    raise ImportError(
        f"Could not import BehaviorAgent.\n"
        f"Please provide --carla-python-api-path or set CARLA_ROOT.\n"
        f"Expected location: {os.path.join(carla_python_api_path or 'CARLA_ROOT', 'PythonAPI', 'carla', 'agents', 'navigation', 'behavior_agent.py')}"
    )


def distance_between_locations(loc1: carla.Location, loc2: carla.Location) -> float:
    """计算两个位置之间的欧式距离"""
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    dz = loc1.z - loc2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_dock_control(
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    max_speed_ms: float = 3.0,   # 层3优化：从 2.0 → 3.0 m/s，更敢跑
    stop_dist: float = 3.5       # 在 3.5m 内刹停
) -> tuple:
    """
    DOCK 阶段专用贴车控制：
    - 目标：停在目标车后方约 2m 的位置
    - 只管最后 0~8m 这段，完全不用 BehaviorAgent
    
    返回: (control, dist_to_target, ego_speed)
    """
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_vel = ego_vehicle.get_velocity()
    ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
    
    target_tf = target_vehicle.get_transform()
    target_loc = target_tf.location
    target_forward = target_tf.get_forward_vector()
    
    # 期望停点：目标车后方 2m
    goal_loc = carla.Location(
        x=target_loc.x - 2.0 * target_forward.x,
        y=target_loc.y - 2.0 * target_forward.y,
        z=target_loc.z,
    )
    
    # ego → goal 的 2D 向量
    vec_x = goal_loc.x - ego_loc.x
    vec_y = goal_loc.y - ego_loc.y
    dist_2d = math.sqrt(vec_x**2 + vec_y**2)
    dist = dist_2d  # 用 2D 距离就行
    
    # ego 朝向 (世界坐标系下)
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    
    # 计算朝向 goal 的夹角（signed）
    dot = forward_x * vec_x + forward_y * vec_y
    cross = forward_x * vec_y - forward_y * vec_x
    angle = math.atan2(cross, dot)  # >0 左转，<0 右转
    
    # 简单比例控制转向
    k_angle = 1.5
    steer = max(-1.0, min(1.0, k_angle * angle))
    
    # 层3优化：目标速度更激进
    if dist > 8.0:
        target_speed = max_speed_ms  # 3.0 m/s
    elif dist > stop_dist:
        target_speed = 1.5  # 原来是 1.0
    else:
        target_speed = 0.8  # 原来是 0.5
    
    # 速度误差 -> 油门（层3优化：更激进）
    k_v = 0.6  # 从 0.5 略加
    speed_err = target_speed - ego_speed
    throttle = k_v * speed_err
    throttle = max(0.25, min(0.7, throttle))  # [0.25, 0.7] 比原来[0.2, 0.5]略猛
    
    control = carla.VehicleControl()
    control.steer = steer
    
    # 刹停条件：足够近 + 速度很低
    if dist <= stop_dist and ego_speed < 0.5:
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = True
    else:
        control.throttle = max(0.0, throttle)
        control.brake = 0.0
        control.hand_brake = False
    
    return control, dist, ego_speed


def setup_world(client: carla.Client, town: str, weather: str):
    """Setup CARLA world with specified town and weather."""
    WEATHER_PRESETS = {
        "clear_noon": carla.WeatherParameters.ClearNoon,
        "clear": carla.WeatherParameters.ClearNoon,
        "wet_cloudy_noon": carla.WeatherParameters.WetCloudyNoon,
        "rainy": carla.WeatherParameters.WetCloudyNoon,
    }
    
    world = client.load_world(town)
    if weather in WEATHER_PRESETS:
        world.set_weather(WEATHER_PRESETS[weather])
    
    # 设置同步模式和固定步长（所有实验统一）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    print(f"[Setup] Loaded {town} with {weather} weather")
    print(f"[Setup] Synchronous mode: True, Fixed delta: 0.05s")
    return world


def run_behavior_nav(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_location: carla.Location,
    BehaviorAgent: type,
    max_steps: int = 800,
    success_radius_4m: float = 4.0,
    success_radius_8m: float = 8.0,
    behavior: str = "normal",
    record_trajectory: bool = False,
    traj_subsample: int = 1,
    target_vehicle: Optional[carla.Vehicle] = None,
    use_approach_point: bool = True,
    approach_back_distance: float = 10.0,
    approach_random_offset: float = 2.0,
) -> Dict[str, Any]:
    """三阶段导航：APPROACH(接近点) → TARGET(目标车) → DOCK(贴车PID)"""

    carla_map = world.get_map()

    # 1) 确保 ego 在路上
    try:
        wp = carla_map.get_waypoint(
            ego_vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is not None:
            tf = wp.transform
            tf.location.z = ego_vehicle.get_transform().location.z
            ego_vehicle.set_transform(tf)
    except Exception as e:
        print(f"[Warning] Failed to snap ego to road: {e}")

    # 2) 生成接近点（导航目标），但用于成功判定的仍是 target_vehicle
    actual_target_location = target_location
    actual_target_vehicle_for_distance = target_vehicle

    if use_approach_point and target_vehicle is not None:
        try:
            from .target_route_planner import generate_approach_location
            ap = generate_approach_location(
                world, target_vehicle,
                approach_back_distance,
                approach_random_offset
            )
            if ap is not None:
                actual_target_location = ap
                print(f"[APPROACH] Use approach point ({ap.x:.2f},{ap.y:.2f})")
        except Exception as e:
            print(f"[Warning] generate_approach_location failed: {e}")

    # 3) BehaviorAgent 初始化
    agent = BehaviorAgent(ego_vehicle, behavior=behavior)

    # 初始模式：有接近点 → APPROACH；否则直接 TARGET
    if use_approach_point and target_vehicle is not None:
        nav_dest = actual_target_location  # 接近点
        mode = "APPROACH"
    else:
        nav_dest = target_location         # 直接目标车位置
        actual_target_location = target_location
        mode = "TARGET"

    try:
        agent.set_destination(nav_dest)
    except Exception as e:
        raise RuntimeError(f"set_destination failed: {e}")

    # 稍微 tick 一下让 planner 建好路径
    for _ in range(10):
        world.tick()

    # 4) 保证 target 车静止
    if target_vehicle is not None:
        try:
            target_vehicle.set_autopilot(False)
            target_vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0,
                hand_brake=False, reverse=False, manual_gear_shift=False
            ))
        except Exception as e:
            print(f"[Warning] Failed to freeze target vehicle: {e}")

    # 5) 初始距离（对目标车）
    ego_tf = ego_vehicle.get_transform()
    if actual_target_vehicle_for_distance is not None:
        tgt_loc = actual_target_vehicle_for_distance.get_transform().location
    else:
        tgt_loc = actual_target_location
    start_dist = distance_between_locations(ego_tf.location, tgt_loc)
    min_dist = start_dist
    final_dist = start_dist

    print(f"[Info] start_dist_to_target = {start_dist:.2f}m")

    # 阈值 - ORACLE_V3 (3-stage) 最终配置
    # setting3: DOCK=10, TARGET=18, behavior=aggressive, approach_back=6.0, offset=1.0
    DOCK_START_DIST = 10.0         # ≤10m 进入 DOCK（提前启动，让DOCK有更多机会）
    SWITCH_TO_TARGET_DIST = 18.0   # ≤18m 从 APPROACH 切到 TARGET（提前切换，让BA以"车"为目标时间更长）

    # 极简 stuck 检测：只在远距离用
    prev_dist = start_dist
    no_progress_steps = 0
    PROGRESS_THRESH = 0.1   # 距离减少 <0.1m 认为没明显进展
    PROGRESS_WINDOW = 250   # 连续 250 步无进展才认为 stuck（非常宽松）

    stuck = False
    success_4m = False
    success_8m = False
    steps = 0
    trajectory = []

    for step in range(max_steps):
        try:
            # --- target 车一直刹车 ---
            if target_vehicle is not None and target_vehicle.is_alive:
                try:
                    target_vehicle.set_autopilot(False)
                    target_vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=1.0,
                        hand_brake=False, reverse=False, manual_gear_shift=False
                    ))
                except Exception:
                    pass

            # --- 当前距离（对目标车） ---
            ego_tf = ego_vehicle.get_transform()
            ego_loc = ego_tf.location
            if actual_target_vehicle_for_distance is not None:
                tgt_loc = actual_target_vehicle_for_distance.get_transform().location
            else:
                tgt_loc = actual_target_location
            dist = distance_between_locations(ego_loc, tgt_loc)
            min_dist = min(min_dist, dist)

            # === 阶段切换逻辑 ===

            # 1) APPROACH → TARGET：接近到一定程度后，让 BehaviorAgent 直接以"车"为目标
            if (mode == "APPROACH"
                and target_vehicle is not None
                and dist <= SWITCH_TO_TARGET_DIST):
                new_dest = target_vehicle.get_transform().location
                try:
                    agent.set_destination(new_dest)
                    actual_target_location = new_dest  # 兜底用
                    mode = "TARGET"
                    print(f"[Mode] Switch APPROACH → TARGET at dist={dist:.2f}m")
                except Exception as e:
                    print(f"[Warning] Switch to TARGET failed: {e}")

            # 2) 任意阶段 → DOCK：真正贴车阶段
            if dist <= DOCK_START_DIST and mode != "DOCK":
                mode = "DOCK"
                print(f"[Mode] Switch to DOCK at dist={dist:.2f}m")

            # === 控制逻辑 ===

            if mode == "DOCK":
                # DOCK阶段： PID/贴车控制（有target_vehicle优先）
                if target_vehicle is not None:
                    control, dock_dist, ego_speed = compute_dock_control(
                        ego_vehicle, target_vehicle,
                        max_speed_ms=3.0,  # 层3优化：从 2.0 → 3.0
                        stop_dist=3.5,
                    )
                    dist_for_success = dock_dist
                else:
                    # 没有 target_vehicle 的兜底：朝 actual_target_location 做简易贴近
                    ego_tf = ego_vehicle.get_transform()
                    ego_loc = ego_tf.location
                    target_loc = actual_target_location

                    vec_x = target_loc.x - ego_loc.x
                    vec_y = target_loc.y - ego_loc.y
                    dist_2d = math.sqrt(vec_x**2 + vec_y**2)
                    dist_for_success = dist_2d

                    yaw_rad = math.radians(ego_tf.rotation.yaw)
                    forward_x = math.cos(yaw_rad)
                    forward_y = math.sin(yaw_rad)

                    dot = forward_x * vec_x + forward_y * vec_y
                    cross = forward_x * vec_y - forward_y * vec_x
                    angle = math.atan2(cross, dot)
                    k_angle = 1.5
                    steer = max(-1.0, min(1.0, k_angle * angle))

                    ego_vel = ego_vehicle.get_velocity()
                    ego_speed = math.sqrt(
                        ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2
                    )

                    if dist_for_success > 6.0:
                        target_speed = 2.0
                    elif dist_for_success > 3.5:
                        target_speed = 1.0
                    else:
                        target_speed = 0.5

                    k_v = 0.5
                    speed_err = target_speed - ego_speed
                    throttle = max(0.2, min(0.5, k_v * speed_err))

                    control = carla.VehicleControl()
                    if dist_for_success <= 3.5 and ego_speed < 0.5:
                        control.throttle = 0.0
                        control.brake = 1.0
                        control.hand_brake = True
                    else:
                        control.steer = steer
                        control.throttle = throttle
                        control.brake = 0.0
                        control.hand_brake = False

                ego_vehicle.apply_control(control)
                world.tick()

                dist = dist_for_success
                min_dist = min(min_dist, dist)

                # DOCK 内的成功判断
                if dist < success_radius_4m and ego_speed < 0.5:
                    success_4m = True
                    success_8m = True
                    final_dist = dist
                    steps = step + 1
                    print(f"[Success-DOCK] dist={dist:.2f}m speed={ego_speed:.2f}m/s")
                    ego_vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
                    world.tick()
                    break

                if dist < success_radius_8m:
                    success_8m = True

                # DOCK 不再跑 BehaviorAgent，也不做 stuck 检测
                steps = step + 1
                continue

            else:
                # APPROACH / TARGET 阶段：全权交给 BehaviorAgent
                try:
                    control = agent.run_step()
                except AttributeError as e:
                    print(f"[Warning] BehaviorAgent AttributeError at step {step}: {e}")
                    try:
                        # 恢复时使用当前的actual_target_location（会根据模式切换更新）
                        agent.set_destination(actual_target_location)
                        control = agent.run_step()
                    except Exception as e2:
                        print(f"[Error] BehaviorAgent recover failed: {e2}")
                        control = carla.VehicleControl()

                # === 层2：距离自适应最小油门（只在方向对齐时强制加速） ===
                # 重新计算当前距离、速度、朝向一致性
                ego_tf = ego_vehicle.get_transform()
                ego_loc = ego_tf.location
                if actual_target_vehicle_for_distance is not None:
                    tgt_loc = actual_target_vehicle_for_distance.get_transform().location
                else:
                    tgt_loc = actual_target_location
                
                # 计算距离
                vec_to_tgt = carla.Location(
                    x=tgt_loc.x - ego_loc.x,
                    y=tgt_loc.y - ego_loc.y,
                    z=0.0
                )
                dist_to_tgt = math.sqrt(vec_to_tgt.x**2 + vec_to_tgt.y**2)
                
                # 计算速度
                vel = ego_vehicle.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                
                # 计算朝向与目标方向的对齐程度
                if dist_to_tgt > 0.1:
                    to_tgt_dir = (vec_to_tgt.x / dist_to_tgt, vec_to_tgt.y / dist_to_tgt)
                else:
                    to_tgt_dir = (0.0, 0.0)
                
                yaw_rad = math.radians(ego_tf.rotation.yaw)
                forward = (math.cos(yaw_rad), math.sin(yaw_rad))
                direction_alignment = to_tgt_dir[0] * forward[0] + to_tgt_dir[1] * forward[1]
                
                # 只在"方向对得比较准"的时候强制多给油
                if direction_alignment > 0.4:
                    if dist_to_tgt > 35.0:
                        # 很远：必须保证跑起来
                        control.throttle = max(control.throttle, 0.55)
                        control.brake = 0.0
                    elif dist_to_tgt > 22.0:
                        # 中远：至少 0.45
                        control.throttle = max(control.throttle, 0.45)
                        control.brake = min(control.brake, 0.2)
                    elif dist_to_tgt > 15.0:
                        # 中距离：稍微推一推
                        control.throttle = max(control.throttle, 0.35)
                        control.brake = min(control.brake, 0.3)
                # 否则（转弯、掉头等），交给 BehaviorAgent 自己慢慢来

                ego_vehicle.apply_control(control)
                world.tick()

                # 更新一次距离
                ego_tf = ego_vehicle.get_transform()
                ego_loc = ego_tf.location
                if actual_target_vehicle_for_distance is not None:
                    tgt_loc = actual_target_vehicle_for_distance.get_transform().location
                else:
                    tgt_loc = actual_target_location
                dist = distance_between_locations(ego_loc, tgt_loc)
                min_dist = min(min_dist, dist)

                # APPROACH/TARGET 阶段不直接标记 success_8m，由 DOCK 负责精细贴近
                # 这里只负责极简 stuck 检测（只在远距离起作用）
                if dist > success_radius_8m + 5.0:  # 离得很远才考虑 stuck
                    progress = prev_dist - dist  # 正值表示在接近
                    if progress < PROGRESS_THRESH:
                        no_progress_steps += 1
                    else:
                        no_progress_steps = 0
                    prev_dist = dist

                    if no_progress_steps >= PROGRESS_WINDOW:
                        print(f"[Stuck] no progress for {no_progress_steps} steps, dist={dist:.2f}m")
                        stuck = True
                        final_dist = dist
                        steps = step + 1
                        break

            # --- 轨迹记录 ---
            if record_trajectory and (step % traj_subsample == 0):
                ego_tf = ego_vehicle.get_transform()
                ego_loc = ego_tf.location
                vel = ego_vehicle.get_velocity()
                speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
                trajectory.append({
                    "step": int(step),
                    "x": float(ego_loc.x),
                    "y": float(ego_loc.y),
                    "z": float(ego_loc.z),
                    "yaw": float(ego_tf.rotation.yaw),
                    "speed": float(speed),
                    "dist": float(dist),
                    "mode": mode,
                })

            steps = step + 1

        except Exception as e:
            print(f"[Error] Exception in run_behavior_nav loop at step {step}: {e}")
            stuck = True
            final_dist = dist
            break

    # 补 final_dist
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    if actual_target_vehicle_for_distance is not None:
        tgt_loc = actual_target_vehicle_for_distance.get_transform().location
    else:
        tgt_loc = actual_target_location
    final_dist = distance_between_locations(ego_loc, tgt_loc)

    return {
        "start_dist": start_dist,
        "min_dist": min_dist,
        "final_dist": final_dist,
        "steps": steps,
        "success_4m": success_4m,
        "success_8m": success_8m,
        "stuck": stuck,
        "trajectory": trajectory,
    }
