#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle + BehaviorAgent 最小 Demo

目的：验证 BehaviorAgent 控制器本身的能力
- 使用 Oracle 目标（ground truth）
- 动态追踪目标车辆前方的路点
- 统计导航成功率

如果 BehaviorAgent + Oracle 能达到可接受的成功率（如 30-60% @4m），
则可以放心地接入 GS 模型。
"""

import os
import sys
import math
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import io

import numpy as np
import carla

# Fix encoding for Windows without re-wrapping an already wrapped stream.
def _ensure_utf8_stdio():
    if sys.platform != 'win32':
        return
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        encoding = (getattr(stream, 'encoding', None) or '').lower().replace('-', '')
        if encoding == 'utf8' or not hasattr(stream, 'buffer'):
            continue
        setattr(
            sys,
            stream_name,
            io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace'),
        )


_ensure_utf8_stdio()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lc_org_nav.online_nav import _find_carla_python_api

# Weather presets
WEATHER_PRESETS = {
    "clear": carla.WeatherParameters.ClearNoon,
    "rainy": carla.WeatherParameters.WetCloudyNoon,
}


def setup_world(client, town: str, weather: str):
    """Setup CARLA world with specified town and weather."""
    world = client.load_world(town)
    if weather in WEATHER_PRESETS:
        world.set_weather(WEATHER_PRESETS[weather])
    print(f"[Setup] Loaded {town} with {weather} weather")
    return world


def setup_ego_and_target_vehicles(client, world, num_npcs: int = 0):
    """
    Spawn ego and target vehicles (and optionally NPCs).
    
    For Oracle experiment, we can set num_npcs=0 to reduce complexity.
    """
    blueprint_library = world.get_blueprint_library()
    
    # Spawn ego vehicle
    ego_bp = blueprint_library.filter("vehicle.tesla.model3")
    if not ego_bp:
        ego_bp = blueprint_library.filter("vehicle.*")
    ego_bp = ego_bp[0]
    ego_bp.set_attribute("role_name", "ego")
    
    # Spawn target vehicle
    target_bp = blueprint_library.filter("vehicle.*")
    target_bp = random.choice(target_bp)
    if target_bp.has_attribute("color"):
        color = random.choice(target_bp.get_attribute("color").recommended_values)
        target_bp.set_attribute("color", color)
    target_bp.set_attribute("role_name", "target")
    
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    if len(spawn_points) < 2:
        raise RuntimeError("Not enough spawn points found")
    
    # Spawn ego and target at random positions (will be repositioned per episode)
    ego_vehicle = world.spawn_actor(ego_bp, spawn_points[0])
    target_vehicle = world.spawn_actor(target_bp, spawn_points[1])
    
    # Make target static initially
    target_vehicle.set_autopilot(False)
    zero_control = carla.VehicleControl(
        throttle=0.0, steer=0.0, brake=1.0,
        hand_brake=False, reverse=False, manual_gear_shift=False
    )
    target_vehicle.apply_control(zero_control)
    
    # Optionally spawn NPCs (can be 0 for simpler experiment)
    npc_vehicles = []
    if num_npcs > 0:
        try:
            tm_client = world.get_client()
        except AttributeError:
            tm_client = client
        tm = tm_client.get_trafficmanager()
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        
        vehicle_bps = blueprint_library.filter("vehicle.*")
        for i in range(2, min(2 + num_npcs, len(spawn_points))):
            npc_bp = random.choice(vehicle_bps)
            if npc_bp.has_attribute("color"):
                color = random.choice(npc_bp.get_attribute("color").recommended_values)
                npc_bp.set_attribute("color", color)
            
            npc = world.try_spawn_actor(npc_bp, spawn_points[i])
            if npc is not None:
                npc_vehicles.append(npc)
                tm.vehicle_lane_offset(npc, random.uniform(-0.5, 0.5))
                tm.auto_lane_change(npc, True)
                tm.distance_to_leading_vehicle(npc, random.uniform(1.5, 3.0))
                tm.ignore_lights_percentage(npc, random.uniform(0, 50))
                npc.set_autopilot(True, tm.get_port())
    else:
        tm = None
    
    return ego_vehicle, target_vehicle, npc_vehicles, tm


def distance_between_locations(loc1: carla.Location, loc2: carla.Location) -> float:
    """Calculate Euclidean distance between two locations."""
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    return math.sqrt(dx * dx + dy * dy)


def sample_pair_same_lane(
    world: carla.World,
    min_dist: float = 30.0,
    max_dist: float = 60.0,
    max_trials: int = 50,
) -> Optional[tuple]:
    """
    Sample ego and target spawn points on the same lane.
    Ego is behind target by min_dist to max_dist meters.
    
    This ensures:
    - Topologically connected (same lane)
    - Simple path (drive forward along the lane)
    - Controllable distance (30-60m)
    
    Returns:
        (ego_spawn_point, target_spawn_point, euclidean_distance) or None
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    
    for _ in range(max_trials):
        # Randomly pick target spawn point
        sp_target = random.choice(spawn_points)
        wp_target = carla_map.get_waypoint(sp_target.location)
        
        # Sample distance
        dist = random.uniform(min_dist, max_dist)
        
        # Get waypoints behind target (ego will be behind target)
        wps_ego = wp_target.previous(dist)
        
        if not wps_ego:
            continue
        
        # Use the last waypoint (furthest from target)
        wp_ego = wps_ego[-1]
        sp_ego = wp_ego.transform
        
        # Calculate actual Euclidean distance (for logging)
        d = distance_between_locations(sp_ego.location, sp_target.location)
        
        # Verify distance is in reasonable range
        if min_dist * 0.8 <= d <= max_dist * 1.2:
            return sp_ego, sp_target, d
    
    return None


def sample_ego_target_spawnpoints(
    spawn_points: List[carla.Transform],
    min_dist: float = 25.0,
    max_dist: float = 80.0,
    max_trials: int = 100,
) -> Optional[tuple]:
    """
    Sample ego and target spawn points with distance constraint only.
    No road_id constraint for now (simpler, more stable).
    
    DEPRECATED: Use sample_pair_same_lane for better results.
    
    Returns:
        (ego_spawn_point, target_spawn_point) or None
    """
    for _ in range(max_trials):
        sp_target = random.choice(spawn_points)
        sp_ego = random.choice(spawn_points)
        
        if sp_target == sp_ego:
            continue
        
        d = distance_between_locations(sp_target.location, sp_ego.location)
        
        if min_dist <= d <= max_dist:
            return sp_ego, sp_target
    
    return None


def find_spawn_point_on_same_road(
    world: carla.World,
    target_location: carla.Location,
    min_distance: float = 30.0,
    max_distance: float = 80.0,
    max_trials: int = 100,
) -> Optional[carla.Transform]:
    """
    Find a spawn point on the same road as target, within distance range.
    
    Returns:
        Transform for ego spawn point, or None if not found.
    """
    carla_map = world.get_map()
    target_wp = carla_map.get_waypoint(target_location)
    target_road_id = target_wp.road_id
    
    # Try waypoints ahead and behind target
    for direction in ['next', 'previous']:
        for distance in range(int(min_distance), int(max_distance) + 1, 5):
            if direction == 'next':
                waypoints = target_wp.next(distance)
            else:
                waypoints = target_wp.previous(distance)
            
            for wp in waypoints:
                if wp.road_id == target_road_id:
                    # Check if distance is in range
                    dx = wp.transform.location.x - target_location.x
                    dy = wp.transform.location.y - target_location.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    
                    if min_distance <= dist <= max_distance:
                        return wp.transform
    
    return None


def sample_static_target_and_ego_position(
    world: carla.World,
    npc_vehicles: List[carla.Vehicle],
    max_trials: int = 50,
) -> Optional[tuple]:
    """
    Sample a target vehicle (will be made static) and find ego position on same road.
    
    Returns:
        (target_vehicle, ego_transform, rel_type) or None
    """
    from scripts.data.carla_collect_data import classify_relation
    
    if len(npc_vehicles) < 2:
        return None
    
    carla_map = world.get_map()
    
    for _ in range(max_trials):
        # Filter alive vehicles
        alive_vehicles = [v for v in npc_vehicles if v.is_alive]
        if len(alive_vehicles) < 2:
            continue
        
        # Sample anchor and target with relation
        anchor = random.choice(alive_vehicles)
        target = random.choice([v for v in alive_vehicles if v.id != anchor.id])
        
        try:
            anchor_tf = anchor.get_transform()
            target_tf = target.get_transform()
        except:
            # Actor might be destroyed, skip
            continue
        
        # Try each relation type
        for rel_type in range(5):  # 0-4: near/front/behind/left/right
            if classify_relation(anchor_tf, target_tf, rel_type):
                # Find ego spawn point on same road as target
                target_location = target_tf.location
                ego_transform = find_spawn_point_on_same_road(
                    world, target_location, min_distance=30.0, max_distance=80.0
                )
                
                if ego_transform is not None:
                    return (target, ego_transform, rel_type)
    
    return None


def _import_behavior_agent(carla_python_api_path: Optional[str] = None):
    """Import BehaviorAgent with proper path handling."""
    BehaviorAgent = None
    
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


def run_oracle_behavior_agent_episode(
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    BehaviorAgent,
    max_steps: int = 800,
    behavior: str = "normal",
) -> Dict[str, Any]:
    """
    Run a single Oracle episode using BehaviorAgent.
    
    Navigation to static target: directly navigate to target vehicle's location.
    Target vehicle is static (not moving), so this is a standard "A to B" navigation task.
    
    Args:
        world: CARLA world
        ego_vehicle: Ego vehicle actor
        target_vehicle: Target vehicle actor (Oracle: ground truth, static)
        BehaviorAgent: BehaviorAgent class (imported)
        max_steps: Maximum navigation steps
        behavior: BehaviorAgent behavior mode ("normal", "cautious", "aggressive")
        
    Returns:
        Dict with:
            - start_dist: float (initial ego→target distance)
            - min_dist: float (minimum distance achieved)
            - final_dist: float (final distance at end of episode)
            - steps: int (number of steps taken)
            - success_4m: bool
            - success_8m: bool
            - success_10m: bool
    """
    
    # Record initial distance
    ego_location_start = ego_vehicle.get_location()
    target_location = target_vehicle.get_location()
    dx_start = target_location.x - ego_location_start.x
    dy_start = target_location.y - ego_location_start.y
    start_dist = math.sqrt(dx_start * dx_start + dy_start * dy_start)
    
    # Wait a few ticks before initializing BehaviorAgent
    for _ in range(10):
        world.tick()
    
    # Initialize BehaviorAgent
    try:
        agent = BehaviorAgent(ego_vehicle, behavior=behavior)
    except Exception as e:
        # If initialization fails, wait more and try again
        for _ in range(10):
            world.tick()
        agent = BehaviorAgent(ego_vehicle, behavior=behavior)
    
    min_dist = float("inf")
    success_4m = False
    success_8m = False
    success_10m = False
    
    # Wait more ticks for BehaviorAgent to fully initialize path planning
    for _ in range(15):
        world.tick()
    
    # Set destination after initialization
    try:
        agent.set_destination(target_location)
    except Exception as e:
        # If set_destination fails, wait more and try again
        for _ in range(10):
            world.tick()
        try:
            agent.set_destination(target_location)
        except:
            # If still fails, raise the error
            raise RuntimeError(f"Failed to set destination after multiple attempts: {e}")
    
    # Main navigation loop
    final_dist = start_dist
    for step in range(max_steps):
        world.tick()
        
        # Get control from BehaviorAgent
        control = agent.run_step()
        ego_vehicle.apply_control(control)
        
        # Compute distance to target (static)
        ego_location = ego_vehicle.get_location()
        dx = target_location.x - ego_location.x
        dy = target_location.y - ego_location.y
        dist = math.sqrt(dx * dx + dy * dy)
        
        min_dist = min(min_dist, dist)
        final_dist = dist  # Update final distance
        
        # Check success thresholds
        if dist < 4.0:
            success_4m = True
        if dist < 8.0:
            success_8m = True
        if dist < 10.0:
            success_10m = True
        
        # Early exit if reached 4m
        if success_4m:
            break
    
    return {
        "start_dist": float(start_dist),
        "min_dist": float(min_dist),
        "final_dist": float(final_dist),
        "steps": step + 1,
        "success_4m": success_4m,
        "success_8m": success_8m,
        "success_10m": success_10m,
    }


def run_evaluation(
    client: carla.Client,
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    spawn_points: List[carla.Transform],
    BehaviorAgent,
    num_episodes: int = 30,
    max_steps: int = 800,
    behavior: str = "normal",
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run multiple Oracle episodes."""
    results = []
    
    # Use same-lane sampling by default (required for Step 1: Oracle Demo)
    # Only use random sampling if explicitly requested via output filename
    use_random_sampling = output_path is not None and "random" in output_path.lower()
    
    print(f"\n[Oracle Demo] Running {num_episodes} episodes with BehaviorAgent...")
    print(f"[Config] Behavior: {behavior}, Max steps: {max_steps}")
    print(f"[Config] Sampling: {'Random' if use_random_sampling else 'Same-lane (30-60m)'}")
    print("-" * 80)
    
    for episode in range(1, num_episodes + 1):
        
        if use_random_sampling:
            # Random sampling (old strategy)
            sample_result = sample_ego_target_spawnpoints(
                spawn_points, min_dist=25.0, max_dist=80.0, max_trials=100
            )
            if sample_result is None:
                print(f"  Episode {episode}/{num_episodes}... Failed to sample spawn points")
                continue
            ego_spawn, target_spawn = sample_result
        else:
            # Same-lane sampling (new strategy, default)
            sample_result = sample_pair_same_lane(
                world, min_dist=30.0, max_dist=60.0, max_trials=50
            )
            if sample_result is None:
                print(f"  Episode {episode}/{num_episodes}... Failed to sample same-lane pair")
                continue
            ego_spawn, target_spawn, euclidean_dist = sample_result
        
        # Reposition vehicles (no destroy/respawn, just set_transform)
        # Add retry mechanism for RPC timeouts
        reposition_success = False
        for retry in range(3):
            try:
                if not ego_vehicle.is_alive or not target_vehicle.is_alive:
                    print(f"    Warning: Vehicles not alive, skipping episode")
                    break
                
                # Reset ego vehicle (with retry on timeout)
                try:
                    ego_vehicle.set_transform(ego_spawn)
                except Exception as e:
                    if "timeout" in str(e).lower() and retry < 2:
                        # Wait a bit and retry
                        for _ in range(20):
                            world.tick()
                        continue
                    else:
                        raise
                
                zero_control = carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0,
                    hand_brake=False, reverse=False, manual_gear_shift=False
                )
                ego_vehicle.apply_control(zero_control)
                
                # Reset target vehicle (static, with retry on timeout)
                try:
                    target_vehicle.set_transform(target_spawn)
                except Exception as e:
                    if "timeout" in str(e).lower() and retry < 2:
                        # Wait a bit and retry
                        for _ in range(20):
                            world.tick()
                        continue
                    else:
                        raise
                
                target_vehicle.set_autopilot(False)
                target_vehicle.apply_control(zero_control)
                
                # Let world stabilize
                for _ in range(20):
                    world.tick()
                
                reposition_success = True
                break
            except Exception as e:
                if retry == 2:
                    print(f"    Warning: Failed to reposition vehicles after 3 retries: {e}")
                else:
                    # Wait longer before retry
                    for _ in range(30):
                        world.tick()
        
        if not reposition_success:
            continue
        
        # Run episode
        try:
            result = run_oracle_behavior_agent_episode(
                world=world,
                ego_vehicle=ego_vehicle,
                target_vehicle=target_vehicle,
                BehaviorAgent=BehaviorAgent,
                max_steps=max_steps,
                behavior=behavior,
            )
            
            result["episode"] = episode
            results.append(result)
            
            # Print progress
            status_4m = "✓" if result["success_4m"] else "✗"
            status_8m = "✓" if result["success_8m"] else "✗"
            print(f"  Episode {episode:2d}/{num_episodes}... "
                  f"Start={result['start_dist']:.1f}m → Min={result['min_dist']:.2f}m → Final={result['final_dist']:.2f}m, "
                  f"4m={status_4m}, 8m={status_8m}, Steps={result['steps']}")
        
        except Exception as e:
            print(f"  Episode {episode}/{num_episodes}... Error: {e}")
            continue
        
        # Stop vehicles for next episode (no destroy, just stop)
        try:
            zero_control = carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0,
                hand_brake=False, reverse=False, manual_gear_shift=False
            )
            if ego_vehicle.is_alive:
                ego_vehicle.apply_control(zero_control)
            if target_vehicle.is_alive:
                target_vehicle.apply_control(zero_control)
        except:
            pass
    
    return results


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from episode results."""
    if not results:
        return {
            "n": 0,
            "nav_succ_4m": 0.0,
            "nav_succ_8m": 0.0,
            "nav_succ_10m": 0.0,
            "min_dist_mean": None,
            "min_dist_median": None,
            "min_dist_std": None,
            "start_dist_mean": None,
            "start_dist_std": None,
            "start_dist_min": None,
            "start_dist_max": None,
            "final_dist_mean": None,
            "final_dist_median": None,
            "final_dist_std": None,
        }
    
    n = len(results)
    
    nav_succ_4m = sum(1 for r in results if r["success_4m"]) / n * 100.0
    nav_succ_8m = sum(1 for r in results if r["success_8m"]) / n * 100.0
    nav_succ_10m = sum(1 for r in results if r["success_10m"]) / n * 100.0
    
    min_dists = [r["min_dist"] for r in results]
    min_dist_mean = np.mean(min_dists)
    min_dist_median = np.median(min_dists)
    min_dist_std = np.std(min_dists)
    
    start_dists = [r["start_dist"] for r in results]
    start_dist_mean = np.mean(start_dists)
    start_dist_std = np.std(start_dists)
    start_dist_min = np.min(start_dists)
    start_dist_max = np.max(start_dists)
    
    final_dists = [r["final_dist"] for r in results]
    final_dist_mean = np.mean(final_dists)
    final_dist_median = np.median(final_dists)
    final_dist_std = np.std(final_dists)
    
    return {
        "n": n,
        "nav_succ_4m": round(nav_succ_4m, 2),
        "nav_succ_8m": round(nav_succ_8m, 2),
        "nav_succ_10m": round(nav_succ_10m, 2),
        "min_dist_mean": round(min_dist_mean, 2),
        "min_dist_median": round(min_dist_median, 2),
        "min_dist_std": round(min_dist_std, 2),
        "start_dist_mean": round(start_dist_mean, 2),
        "start_dist_std": round(start_dist_std, 2),
        "start_dist_min": round(start_dist_min, 2),
        "start_dist_max": round(start_dist_max, 2),
        "final_dist_mean": round(final_dist_mean, 2),
        "final_dist_median": round(final_dist_median, 2),
        "final_dist_std": round(final_dist_std, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Oracle + BehaviorAgent Demo - 验证控制器能力"
    )
    parser.add_argument("--town", type=str, default="Town10HD_Opt",
                        help="Town name")
    parser.add_argument("--weather", type=str, default="clear",
                        choices=["clear", "rainy"],
                        help="Weather condition")
    parser.add_argument("--num-episodes", type=int, default=30,
                        help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=800,
                        help="Maximum navigation steps per episode")
    parser.add_argument("--behavior", type=str, default="normal",
                        choices=["normal", "cautious", "aggressive"],
                        help="BehaviorAgent behavior mode")
    parser.add_argument("--num-npcs", type=int, default=0,
                        help="Number of NPC vehicles (0 for simpler experiment)")
    parser.add_argument("--carla-host", type=str, default="localhost",
                        help="CARLA server host")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA server port")
    parser.add_argument("--carla-python-api-path", type=str, default=None,
                        help="Path to CARLA PythonAPI directory")
    parser.add_argument("--output", type=str, default="logs_v2/oracle_town10_clear.json",
                        help="Output JSON file")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Oracle + BehaviorAgent Demo")
    print("=" * 80)
    print(f"目的: 验证 BehaviorAgent 控制器本身的能力")
    print(f"配置:")
    print(f"  Town: {args.town}")
    print(f"  Weather: {args.weather}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Behavior: {args.behavior}")
    print("=" * 80)
    
    # Connect to CARLA
    print(f"\n[CARLA] Connecting to {args.carla_host}:{args.carla_port}")
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(30.0)  # Increased timeout to reduce RPC timeouts
    
    # Try to connect and wait for world
    try:
        world = client.get_world()
        print(f"[CARLA] Connected successfully")
    except Exception as e:
        print(f"[ERROR] Failed to connect to CARLA: {e}")
        print("Please ensure CARLA server is running on localhost:2000")
        sys.exit(1)
    
    # Setup world (reload to ensure clean state)
    try:
        world = setup_world(client, args.town, args.weather)
    except Exception as e:
        print(f"[ERROR] Failed to setup world: {e}")
        sys.exit(1)
    
    # Spawn vehicles (can set num_npcs=0 for simpler experiment)
    print(f"[Setup] Spawning ego, target, and {args.num_npcs} NPCs...")
    ego_vehicle, target_vehicle, npc_vehicles, tm = setup_ego_and_target_vehicles(
        client, world, args.num_npcs
    )
    
    # Get spawn points for episode sampling
    spawn_points = world.get_map().get_spawn_points()
    
    # Let world stabilize
    for _ in range(30):
        world.tick()
    
    print(f"[Setup] Ready. Starting evaluation...")
    
    # Import BehaviorAgent
    print(f"[Setup] Importing BehaviorAgent...")
    try:
        BehaviorAgent = _import_behavior_agent(args.carla_python_api_path)
        print(f"[Setup] BehaviorAgent imported successfully")
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Run evaluation
    results = run_evaluation(
        client=client,
        world=world,
        ego_vehicle=ego_vehicle,
        target_vehicle=target_vehicle,
        spawn_points=spawn_points,
        BehaviorAgent=BehaviorAgent,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        behavior=args.behavior,
        output_path=args.output,
    )
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("实验结果统计")
    print("=" * 80)
    print(f"样本数: {stats['n']}")
    print(f"\n导航成功率:")
    print(f"  NavSucc@4m:  {stats['nav_succ_4m']:.1f}%")
    print(f"  NavSucc@8m:  {stats['nav_succ_8m']:.1f}%")
    print(f"  NavSucc@10m: {stats['nav_succ_10m']:.1f}%")
    print(f"\n距离统计:")
    if stats['min_dist_mean'] is not None:
        print(f"  MinDist:     Mean={stats['min_dist_mean']:.2f} m, Median={stats.get('min_dist_median', 0):.2f} m, Std={stats['min_dist_std']:.2f} m")
        print(f"  StartDist:   Mean={stats['start_dist_mean']:.2f} ± {stats['start_dist_std']:.2f} m (range: {stats['start_dist_min']:.2f} - {stats['start_dist_max']:.2f} m)")
        print(f"  FinalDist:   Mean={stats['final_dist_mean']:.2f} m, Median={stats.get('final_dist_median', 0):.2f} m, Std={stats['final_dist_std']:.2f} m")
    else:
        print(f"  MinDist:     N/A")
        print(f"  StartDist:   N/A")
        print(f"  FinalDist:   N/A")
    print("=" * 80)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": {
            "town": args.town,
            "weather": args.weather,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "behavior": args.behavior,
        },
        "statistics": stats,
        "episodes": results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("评估结论")
    print("=" * 80)
    if stats['nav_succ_4m'] >= 30.0:
        print("✅ BehaviorAgent 表现良好 (NavSucc@4m >= 30%)")
        print("   可以放心地接入 GS 模型进行完整实验")
    elif stats['nav_succ_8m'] >= 50.0:
        print("⚠️  BehaviorAgent 表现中等 (NavSucc@8m >= 50%)")
        print("   可以考虑放宽成功阈值到 8m，或进一步调优")
    else:
        print("❌ BehaviorAgent 表现较差")
        print("   需要检查控制器配置或 CARLA 环境")
    print("=" * 80)
    
    # Cleanup (destroy all actors only at the end)
    print("\n[Cleanup] Destroying actors...")
    try:
        if ego_vehicle.is_alive:
            ego_vehicle.destroy()
    except:
        pass
    
    try:
        if target_vehicle.is_alive:
            target_vehicle.destroy()
    except:
        pass
    
    for npc in npc_vehicles:
        try:
            if npc.is_alive:
                npc.destroy()
        except:
            pass


if __name__ == '__main__':
    main()
