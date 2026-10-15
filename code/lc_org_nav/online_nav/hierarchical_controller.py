"""
分层导航控制器

统一接口，整合语义子目标管理、路径规划、低层控制和恢复策略
"""

import math
import carla
import csv
import os
import sys
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .semantic_goals import SemanticGoalManager, SemanticGoal
from .route_planner import RoutePlanner
from .controllers import (
    PurePursuitController,
    SpeedPIDController,
    DockingController,
    ControlOutput
)
from .recovery import StuckDetector, RecoveryPolicy
from .geometry_utils import get_rel_xy, make_back_goal_from_target


def _ensure_carla_agents_path() -> None:
    """Expose CARLA's bundled agents package when PythonAPI/carla is nested."""
    candidates: List[str] = []
    api_path = os.environ.get("CARLA_PYTHON_API_PATH")
    carla_root = os.environ.get("CARLA_ROOT")
    if api_path:
        candidates.append(os.path.join(api_path, "carla"))
    if carla_root:
        candidates.extend([
            os.path.join(carla_root, "WindowsNoEditor", "PythonAPI", "carla"),
            os.path.join(carla_root, "PythonAPI", "carla"),
        ])
    candidates.extend([
        "E:/carla/WindowsNoEditor/PythonAPI/carla",
        "E:/carla/PythonAPI/carla",
        "C:/CARLA/PythonAPI/carla",
        "D:/CARLA/PythonAPI/carla",
    ])
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


CONTROLLER_PROFILES = {
    "paper_fixed": {
        "planner_mode": "simple",
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
    },
    "improved_v1": {
        "planner_mode": "simple",
        "goal_manager": {
            "g1_distance": 24.0,
            "g2_distance": 8.0,
            "g3_distance": 2.8,
            "g2_max_steps": 360,
            "g3_max_steps": 260,
            "g1_target_speed": 10.0,
            "g2_target_speed": 6.0,
            "g3_target_speed": 1.8,
        },
        "pure_pursuit": {"lookahead_distance": 7.0},
        "speed_pid": {"kp": 0.75, "ki": 0.0, "kd": 0.12},
        "docking_controller": {
            "target_longitudinal": 2.8,
            "k_longitudinal": 0.65,
            "max_speed": 2.0,
            "align_y_thresh": 0.7,
            "align_psi_thresh_deg": 14.0,
            "align_speed": 0.9,
        },
    },
    "improved_v2": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 20.0,
        "ego_side_replace_margin": 2.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "stop_after_goal_types": ["G2"],
        "stop_after_success_radius": 8.0,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 9.0,
            "g3_distance": 3.0,
            "g2_max_steps": 320,
            "g3_max_steps": 240,
            "g1_target_speed": 10.0,
            "g2_target_speed": 6.0,
            "g3_target_speed": 1.8,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
    },
    "improved_v3": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 20.0,
        "ego_side_replace_margin": 2.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "use_min_distance_success_flags": True,
        "stop_after_goal_types": ["G2"],
        "stop_after_success_radius": 8.0,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 9.0,
            "g3_distance": 3.0,
            "g1_max_steps": 180,
            "g2_max_steps": 420,
            "g3_max_steps": 300,
            "g1_target_speed": 10.0,
            "g2_target_speed": 6.0,
            "g3_target_speed": 1.8,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
    },
    "improved_v4": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "use_min_distance_success_flags": True,
        "stop_after_goal_types": ["G2"],
        "stop_after_success_radius": 8.0,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 9.0,
            "g3_distance": 3.0,
            "g2_max_steps": 320,
            "g3_max_steps": 240,
            "g1_target_speed": 10.0,
            "g2_target_speed": 6.0,
            "g3_target_speed": 1.8,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
    },
    "improved_v5": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
    },
    "improved_v6": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 3.0},
        "ego_side_min_start_distance": 34.0,
        "ego_side_replace_margin": 3.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 34.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 7.0,
            "g3_distance": 2.5,
            "g1_max_steps": 180,
            "g2_max_steps": 460,
            "g3_max_steps": 380,
            "g1_target_speed": 10.0,
            "g2_target_speed": 4.8,
            "g3_target_speed": 1.2,
        },
        "pure_pursuit": {"lookahead_distance": 7.0},
        "speed_pid": {"kp": 0.75, "ki": 0.0, "kd": 0.12},
        "docking_controller": {
            "target_longitudinal": 2.5,
            "target_lateral": 0.0,
            "k_longitudinal": 0.55,
            "max_speed": 1.6,
            "align_y_thresh": 0.85,
            "align_psi_thresh_deg": 18.0,
            "align_speed": 0.75,
            "k_stanley": 0.85,
        },
        "dock_g2_soft": {
            "enter_dist": 18.0,
            "exit_dist": 21.0,
            "min_dwell_seconds": 0.8,
        },
        "dock_trigger": {
            "dist_min": 4.0,
            "dist_max": 18.0,
            "x_rel": 0.75,
            "y_rel_max": 5.5,
            "yaw_err_max": 55.0,
        },
        "dock_entry_guard": {
            "x_guard": 0.75,
            "psi_guard_deg": 28.0,
            "y_guard": 3.0,
        },
        "dock_orbit": {
            "attempts": 4,
            "back_dist": 12.0,
            "lat_offset": 0.0,
            "target_speed": 3.5,
            "max_steps": 240,
        },
    },
    "improved_v7": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 9.0,
            "g3_distance": 2.0,
            "g1_max_steps": 500,
            "g2_max_steps": 320,
            "g3_max_steps": 300,
            "g1_target_speed": 11.0,
            "g2_target_speed": 7.0,
            "g3_target_speed": 1.6,
        },
        "pure_pursuit": {"lookahead_distance": 7.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {
            "target_longitudinal": 2.0,
            "target_lateral": 0.0,
            "k_longitudinal": 0.7,
            "max_speed": 2.0,
            "align_y_thresh": 1.0,
            "align_psi_thresh_deg": 24.0,
            "align_speed": 1.0,
            "k_stanley": 0.8,
        },
        "dock_g2_soft": {
            "enter_dist": 18.0,
            "exit_dist": 21.0,
            "min_dwell_seconds": 0.5,
        },
        "dock_trigger": {
            "dist_min": 3.0,
            "dist_max": 18.0,
            "x_rel": 0.5,
            "y_rel_max": 5.5,
            "yaw_err_max": 65.0,
        },
        "dock_entry_guard": {
            "x_guard": 0.5,
            "psi_guard_deg": 30.0,
            "y_guard": 3.0,
        },
        "dock_orbit": {
            "attempts": 3,
            "back_dist": 10.0,
            "lat_offset": 0.0,
            "target_speed": 4.0,
            "max_steps": 200,
        },
    },
    "improved_v8": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 9.0,
            "g3_distance": 2.5,
            "g1_max_steps": 500,
            "g2_max_steps": 320,
            "g3_max_steps": 260,
            "g1_target_speed": 11.0,
            "g2_target_speed": 7.0,
            "g3_target_speed": 1.8,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {
            "target_longitudinal": 2.5,
            "target_lateral": 0.0,
            "k_longitudinal": 0.65,
            "max_speed": 2.2,
            "align_y_thresh": 0.8,
            "align_psi_thresh_deg": 18.0,
            "align_speed": 1.0,
            "k_stanley": 1.0,
        },
        "dock_trigger": {
            "dist_min": 5.0,
            "dist_max": 16.0,
            "x_rel": 1.0,
            "y_rel_max": 5.0,
            "yaw_err_max": 50.0,
        },
        "dock_entry_guard": {
            "x_guard": 1.0,
            "psi_guard_deg": 22.0,
            "y_guard": 2.0,
        },
    },
    "improved_v9": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "goal_reached_radius": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {
            "g1_distance": 25.0,
            "g2_distance": 9.0,
            "g3_distance": 2.7,
            "g1_max_steps": 500,
            "g2_max_steps": 300,
            "g3_max_steps": 240,
            "g1_target_speed": 11.0,
            "g2_target_speed": 7.0,
            "g3_target_speed": 1.8,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {
            "target_longitudinal": 2.7,
            "target_lateral": 0.0,
            "k_longitudinal": 0.65,
            "max_speed": 2.0,
            "align_y_thresh": 0.7,
            "align_psi_thresh_deg": 16.0,
            "align_speed": 0.9,
            "k_stanley": 1.0,
        },
    },
    "improved_v10": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {
            "g3_distance": 2.9,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {
            "target_longitudinal": 2.9,
            "target_lateral": 0.0,
            "k_longitudinal": 0.7,
            "max_speed": 2.0,
            "align_y_thresh": 0.7,
            "align_psi_thresh_deg": 14.0,
            "align_speed": 0.9,
        },
    },
    "improved_v11": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "direct_close_in": {
            "enter_dist": 10.0,
            "target_dist": 3.3,
            "max_speed": 2.0,
            "min_speed": 0.35,
            "k_speed": 0.75,
            "lookahead": 4.0,
        },
    },
    "improved_v12": {
        "planner_mode": "auto",
        "run_mode": "direct_acquisition",
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "direct_acquisition": {
            "route_replan_steps": 40,
            "direct_enter_dist": 14.0,
            "stop_radius": 3.6,
            "far_speed": 9.0,
            "mid_speed": 5.5,
            "near_max_speed": 2.2,
            "near_min_speed": 0.35,
            "near_k_speed": 0.8,
            "lookahead": 5.0,
            "turnaround_dist": 12.0,
        },
    },
    "improved_v13": {
        "planner_mode": "simple",
        "run_mode": "direct_acquisition",
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "direct_acquisition": {
            "route_replan_steps": 999999,
            "direct_enter_dist": 999.0,
            "stop_radius": 2.8,
            "near_max_speed": 4.0,
            "near_min_speed": 0.6,
            "near_k_speed": 1.0,
            "lookahead": 6.0,
            "turnaround_dist": 14.0,
        },
    },
    "improved_v14": {
        "planner_mode": "auto",
        "run_mode": "direct_acquisition",
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "direct_acquisition": {
            "use_basic_agent": True,
            "agent_target_speed_kmh": 28.0,
            "route_replan_steps": 60,
            "direct_enter_dist": 12.0,
            "stop_radius": 3.2,
            "near_max_speed": 3.0,
            "near_min_speed": 0.45,
            "near_k_speed": 0.9,
            "lookahead": 5.0,
            "turnaround_dist": 12.0,
        },
    },
    "improved_v15": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "post_phase_direct_tail": {
            "max_steps": 500,
            "only_if_not_4m": True,
            "stop_radius": 2.6,
            "near_max_speed": 3.0,
            "near_min_speed": 0.45,
            "near_k_speed": 0.9,
            "lookahead": 4.0,
            "turnaround_dist": 18.0,
            "reverse_enabled": True,
            "reverse_dist": 9.0,
            "reverse_throttle": 0.24,
        },
    },
    "improved_v16": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {
            "g1_max_steps": 400,
            "g2_max_steps": 250,
            "g3_max_steps": 150,
            "g1_target_speed": 11.0,
            "g2_target_speed": 7.0,
            "g3_target_speed": 2.0,
        },
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "dock_orbit": {"attempts": 0},
        "post_phase_direct_tail": {
            "max_steps": 1200,
            "only_if_not_4m": True,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
    },
    "improved_v17": {
        "planner_mode": "simple",
        "run_mode": "behavior_acquisition",
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "behavior_acquisition": {
            "behavior": "aggressive",
            "approach_back_distance": 6.0,
            "approach_random_offset": 0.0,
            "switch_to_target_dist": 18.0,
            "direct_enter_dist": 12.0,
            "replan_steps": 80,
            "stop_radius": 2.6,
            "near_max_speed": 3.0,
            "near_min_speed": 0.45,
            "near_k_speed": 0.9,
            "lookahead": 4.0,
            "turnaround_dist": 18.0,
            "reverse_enabled": True,
            "reverse_dist": 9.0,
            "reverse_throttle": 0.24,
        },
    },
    "improved_v18": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "post_phase_direct_tail": {
            "max_steps": 1200,
            "only_if_not_4m": True,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "lookahead_dist_cap": 5.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
    },
    "improved_v19": {
        "planner_mode": "simple",
        "run_mode": "multi_approach_acquisition",
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "multi_approach_acquisition": {
            "approaches": [
                {"name": "ego_side_12m", "type": "ego_side", "distance": 12.0, "max_steps": 360, "target_speed": 7.0},
                {"name": "rear_10m", "back_dist": 10.0, "lat_offset": 0.0, "max_steps": 360, "target_speed": 7.0},
                {"name": "rear_right_7m", "back_dist": 7.0, "lat_offset": 5.5, "max_steps": 360, "target_speed": 5.5},
                {"name": "rear_left_7m", "back_dist": 7.0, "lat_offset": -5.5, "max_steps": 360, "target_speed": 5.5},
                {"name": "front_6m", "back_dist": -6.0, "lat_offset": 0.0, "max_steps": 300, "target_speed": 4.5},
                {"name": "ego_side_6m", "type": "ego_side", "distance": 6.0, "max_steps": 300, "target_speed": 4.5},
            ],
            "direct_after_each": True,
            "direct_steps": 260,
            "final_direct_steps": 700,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
    },
    "improved_v20": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "post_phase_direct_tail": {
            "max_steps": 1200,
            "only_if_not_4m": True,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
        "post_failure_approaches": {
            "only_if_not_4m": True,
            "approaches": [
                {"name": "rear_right_7m", "back_dist": 7.0, "lat_offset": 5.5, "max_steps": 260, "target_speed": 5.5},
                {"name": "rear_left_7m", "back_dist": 7.0, "lat_offset": -5.5, "max_steps": 260, "target_speed": 5.5},
                {"name": "front_6m", "back_dist": -6.0, "lat_offset": 0.0, "max_steps": 260, "target_speed": 4.5},
                {"name": "ego_side_6m", "type": "ego_side", "distance": 6.0, "max_steps": 220, "target_speed": 4.5},
            ],
            "direct_after_each": True,
            "direct_steps": 180,
            "final_direct_steps": 450,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
    },
    "improved_v21": {
        "planner_mode": "simple",
        "use_ego_side_entry_goals": True,
        "ego_side_entry_distances": {"G1": 12.0, "G2": 2.0},
        "ego_side_min_start_distance": 40.0,
        "ego_side_replace_margin": 2.0,
        "skip_initial_route_waypoint": True,
        "skip_initial_route_min_start_distance": 38.0,
        "use_min_distance_success_flags": True,
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
        "post_phase_direct_tail": {
            "max_steps": 1200,
            "only_if_not_4m": True,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
        "post_failure_behavior": {
            "only_if_not_8m": True,
            "max_steps": 2200,
            "behavior": "aggressive",
            "approach_back_distance": 6.0,
            "approach_random_offset": 0.0,
            "switch_to_target_dist": 18.0,
            "direct_enter_dist": 12.0,
            "replan_steps": 80,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
        "post_failure_approaches": {
            "only_if_not_4m": True,
            "approaches": [
                {"name": "rear_right_7m", "back_dist": 7.0, "lat_offset": 5.5, "max_steps": 260, "target_speed": 5.5},
                {"name": "rear_left_7m", "back_dist": 7.0, "lat_offset": -5.5, "max_steps": 260, "target_speed": 5.5},
                {"name": "front_6m", "back_dist": -6.0, "lat_offset": 0.0, "max_steps": 260, "target_speed": 4.5},
                {"name": "ego_side_6m", "type": "ego_side", "distance": 6.0, "max_steps": 220, "target_speed": 4.5},
            ],
            "direct_after_each": True,
            "direct_steps": 180,
            "final_direct_steps": 450,
            "stop_radius": 2.6,
            "near_max_speed": 3.2,
            "near_min_speed": 0.55,
            "near_k_speed": 0.95,
            "lookahead": 4.0,
            "turnaround_dist": 20.0,
            "reverse_enabled": True,
            "reverse_dist": 11.0,
            "reverse_throttle": 0.28,
            "max_throttle": 0.75,
            "stuck_boost_throttle": 0.65,
        },
    },
    "grp_diagnostic": {
        "planner_mode": "auto",
        "goal_manager": {},
        "pure_pursuit": {"lookahead_distance": 8.0},
        "speed_pid": {"kp": 0.8, "ki": 0.0, "kd": 0.15},
        "docking_controller": {"k_longitudinal": 0.7, "max_speed": 2.0},
    },
}


@dataclass
class NavigationResult:
    """导航结果"""
    start_dist: float
    min_dist: float
    final_dist: float
    steps: int
    success_4m: bool
    success_8m: bool
    stuck: bool
    trajectory: List[Dict[str, Any]]
    phase_results: Dict[str, Dict[str, Any]]  # 每个阶段的结果

    safety: Dict[str, Any] = field(default_factory=dict)

class SafetyEventMonitor:
    """Episode-level CARLA safety monitor shared by all selectors."""

    LOGGER_VERSION = "safety_v2"
    SEVERE_IMPULSE_THRESHOLD = 500.0

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle, enabled: bool = True):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.enabled = bool(enabled)
        self.sensors: List[Any] = []
        self.sensor_status: Dict[str, str] = {}
        self.collision_count = 0
        self.collision_actor_ids: List[int] = []
        self.collision_actor_type_ids: List[str] = []
        self.collision_events: List[Dict[str, Any]] = []
        self.collision_max_normal_impulse = 0.0
        self.collision_first_frame: Optional[int] = None
        self.lane_invasion_count = 0
        self.lane_marking_types: List[str] = []
        self.min_ttc_s: Optional[float] = None
        self.min_ttc_actor_id: Optional[int] = None
        self.min_ttc_distance_m: Optional[float] = None
        self.min_distance_to_vehicle_m: Optional[float] = None
        if self.enabled:
            self._attach_sensors()

    def _attach_sensors(self) -> None:
        try:
            bp_lib = self.world.get_blueprint_library()
        except Exception as exc:
            self.sensor_status["blueprint_library"] = f"unavailable:{exc}"
            return

        def spawn_sensor(type_id: str, callback) -> None:
            try:
                bp = bp_lib.find(type_id)
                sensor = self.world.spawn_actor(
                    bp,
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
                    attach_to=self.ego_vehicle,
                )
                sensor.listen(callback)
                self.sensors.append(sensor)
                self.sensor_status[type_id] = "attached"
            except Exception as exc:
                self.sensor_status[type_id] = f"failed:{exc}"

        spawn_sensor("sensor.other.collision", self._on_collision)
        spawn_sensor("sensor.other.lane_invasion", self._on_lane_invasion)

    def _on_collision(self, event: Any) -> None:
        self.collision_count += 1
        frame = int(getattr(event, "frame", -1))
        if self.collision_first_frame is None:
            self.collision_first_frame = frame
        other_actor = getattr(event, "other_actor", None)
        actor_id: Optional[int] = None
        type_id: Optional[str] = None
        if other_actor is not None:
            raw_actor_id = getattr(other_actor, "id", None)
            type_id_raw = getattr(other_actor, "type_id", None)
            if raw_actor_id is not None:
                actor_id = int(raw_actor_id)
                self.collision_actor_ids.append(actor_id)
            if type_id_raw:
                type_id = str(type_id_raw)
                self.collision_actor_type_ids.append(type_id)
        mag = 0.0
        impulse = getattr(event, "normal_impulse", None)
        if impulse is not None:
            mag = math.sqrt(float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2)
            self.collision_max_normal_impulse = max(self.collision_max_normal_impulse, mag)
        self.collision_events.append(
            {
                "frame": frame,
                "actor_id": actor_id,
                "actor_type_id": type_id,
                "normal_impulse": float(mag),
                "dynamic_actor": bool(type_id and (type_id.startswith("vehicle.") or type_id.startswith("walker."))),
                "severe": bool(mag >= self.SEVERE_IMPULSE_THRESHOLD),
            }
        )

    def _on_lane_invasion(self, event: Any) -> None:
        self.lane_invasion_count += 1
        markings = getattr(event, "crossed_lane_markings", []) or []
        for marking in markings:
            self.lane_marking_types.append(str(getattr(marking, "type", "")))

    def update_ttc(self, ego_vehicle: Optional[carla.Vehicle] = None) -> None:
        if not self.enabled:
            return
        ego = ego_vehicle or self.ego_vehicle
        try:
            if not ego.is_alive:
                return
            ego_loc = ego.get_transform().location
            ego_vel = ego.get_velocity()
            actors = self.world.get_actors().filter("vehicle.*")
        except Exception:
            return
        ego_id = getattr(ego, "id", None)
        for actor in actors:
            try:
                if getattr(actor, "id", None) == ego_id or not actor.is_alive:
                    continue
                loc = actor.get_transform().location
                vel = actor.get_velocity()
                rel_x = float(loc.x - ego_loc.x)
                rel_y = float(loc.y - ego_loc.y)
                rel_z = float(loc.z - ego_loc.z)
                dist = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
                if dist <= 1e-6:
                    continue
                if self.min_distance_to_vehicle_m is None or dist < self.min_distance_to_vehicle_m:
                    self.min_distance_to_vehicle_m = dist
                rel_vx = float(vel.x - ego_vel.x)
                rel_vy = float(vel.y - ego_vel.y)
                rel_vz = float(vel.z - ego_vel.z)
                closing_speed = -(rel_x * rel_vx + rel_y * rel_vy + rel_z * rel_vz) / dist
                if closing_speed <= 1e-3:
                    continue
                ttc = dist / closing_speed
                if self.min_ttc_s is None or ttc < self.min_ttc_s:
                    self.min_ttc_s = ttc
                    self.min_ttc_actor_id = int(getattr(actor, "id", -1))
                    self.min_ttc_distance_m = dist
            except Exception:
                continue

    def summary(self) -> Dict[str, Any]:
        dynamic_collision_count = sum(1 for item in self.collision_events if item.get("dynamic_actor"))
        severe_collision_count = sum(1 for item in self.collision_events if item.get("severe"))
        has_dynamic = dynamic_collision_count > 0
        has_static_only = self.collision_count > 0 and not has_dynamic
        return {
            "safety_logger_version": self.LOGGER_VERSION,
            "collision": bool(self.collision_count > 0),
            "collision_count": int(self.collision_count),
            "collision_actor_ids": sorted(set(self.collision_actor_ids)),
            "collision_actor_type_ids": sorted(set(self.collision_actor_type_ids)),
            "collision_events": list(self.collision_events),
            "collision_first_frame": self.collision_first_frame,
            "collision_max_normal_impulse": float(self.collision_max_normal_impulse),
            "dynamic_actor_collision": bool(has_dynamic),
            "dynamic_actor_collision_count": int(dynamic_collision_count),
            "static_only_collision": bool(has_static_only),
            "severe_collision": bool(severe_collision_count > 0),
            "severe_collision_count": int(severe_collision_count),
            "severe_impulse_threshold": float(self.SEVERE_IMPULSE_THRESHOLD),
            "lane_invasion": bool(self.lane_invasion_count > 0),
            "lane_invasion_count": int(self.lane_invasion_count),
            "lane_marking_types": sorted(set(self.lane_marking_types)),
            "min_ttc_s": float(self.min_ttc_s) if self.min_ttc_s is not None else None,
            "min_ttc_actor_id": self.min_ttc_actor_id,
            "min_ttc_distance_m": float(self.min_ttc_distance_m) if self.min_ttc_distance_m is not None else None,
            "min_distance_to_vehicle_m": float(self.min_distance_to_vehicle_m) if self.min_distance_to_vehicle_m is not None else None,
            "safety_sensor_status": dict(self.sensor_status),
        }

    def close(self) -> None:
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        self.sensors = []


class HierarchicalNavController:
    """分层导航控制器"""
    
    def __init__(
        self,
        world: carla.World,
        carla_map: carla.Map,
        success_radius_4m: float = 4.0,
        success_radius_8m: float = 8.0,
        record_trajectory: bool = False,
        traj_subsample: int = 1,
        carla_python_api_path: Optional[str] = None,
        controller_profile: str = "paper_fixed",
        planner_mode: Optional[str] = None,
        record_safety: bool = True,
    ):
        """
        初始化分层导航控制器
        
        Args:
            world: CARLA世界对象
            carla_map: CARLA地图对象
            success_radius_4m: 4m成功半径
            success_radius_8m: 8m成功半径
            record_trajectory: 是否记录轨迹
            traj_subsample: 轨迹采样间隔
            carla_python_api_path: CARLA PythonAPI路径（用于导入GlobalRoutePlanner）
        """
        self.world = world
        self.carla_map = carla_map
        self.success_radius_4m = success_radius_4m
        self.success_radius_8m = success_radius_8m
        self.record_trajectory = record_trajectory
        self.traj_subsample = traj_subsample
        self.record_safety = bool(record_safety)
        if controller_profile not in CONTROLLER_PROFILES:
            valid = ", ".join(sorted(CONTROLLER_PROFILES))
            raise ValueError(f"Unknown controller_profile={controller_profile!r}. Valid profiles: {valid}")
        profile = CONTROLLER_PROFILES[controller_profile]
        self.controller_profile = controller_profile
        self.controller_profile_config = profile
        resolved_planner_mode = planner_mode or profile.get("planner_mode", "simple")
        
        # 初始化各个模块
        self.goal_manager = SemanticGoalManager(carla_map, **profile.get("goal_manager", {}))
        self.route_planner = RoutePlanner(
            carla_map,
            carla_python_api_path=carla_python_api_path,
            mode=resolved_planner_mode,
        )
        # Pure Pursuit：默认lookahead=8m，各阶段会动态调整
        self.pure_pursuit = PurePursuitController(**profile.get("pure_pursuit", {}))
        # Speed PID：优化后的参数
        self.speed_pid = SpeedPIDController(**profile.get("speed_pid", {}))
        # DOCK控制器：优化后的参数
        self.docking_controller = DockingController(**profile.get("docking_controller", {}))
        self.stuck_detector = StuckDetector()
        self.recovery_policy = RecoveryPolicy()
        
        # 当前状态
        self.current_phase: Optional[str] = None
        self.current_route: List[carla.Waypoint] = []
        self.current_route_idx = 0
        
        # Debug日志
        self.debug_log_enabled = False
        self.debug_log_file: Optional[str] = None
        self.debug_log_writer: Optional[csv.DictWriter] = None
        self.debug_log_fp: Optional[Any] = None
    
    def _distance_between_locations(
        self,
        loc1: carla.Location,
        loc2: carla.Location
    ) -> float:
        """计算两个位置之间的距离"""
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.z - loc2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _make_transform_near_target_from_ego_side(
        self,
        ego_loc: carla.Location,
        target_transform: carla.Transform,
        distance_from_target: float,
    ) -> carla.Transform:
        target_loc = target_transform.location
        dx = ego_loc.x - target_loc.x
        dy = ego_loc.y - target_loc.y
        norm = math.sqrt(dx * dx + dy * dy)
        if norm < 1e-6:
            raw_loc = target_loc
        else:
            ratio = max(0.0, min(float(distance_from_target), norm)) / norm
            raw_loc = carla.Location(
                x=target_loc.x + dx * ratio,
                y=target_loc.y + dy * ratio,
                z=target_loc.z,
            )

        waypoint = self.carla_map.get_waypoint(
            raw_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is not None:
            return waypoint.transform
        return carla.Transform(raw_loc, target_transform.rotation)

    def _maybe_use_ego_side_entry_goals(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        goals: List[SemanticGoal],
    ) -> List[SemanticGoal]:
        if not self.controller_profile_config.get("use_ego_side_entry_goals", False):
            return goals

        ego_loc = ego_vehicle.get_transform().location
        target_transform = target_vehicle.get_transform()
        target_loc = target_transform.location
        dist_to_target = self._distance_between_locations(ego_loc, target_loc)
        min_start_distance = float(self.controller_profile_config.get("ego_side_min_start_distance", 0.0))
        if dist_to_target < min_start_distance:
            return goals

        replace_margin = float(self.controller_profile_config.get("ego_side_replace_margin", 8.0))
        entry_distances = self.controller_profile_config.get("ego_side_entry_distances", {})

        rewritten = []
        replaced = []
        for goal in goals:
            entry_distance = entry_distances.get(goal.goal_type)
            if entry_distance is None:
                rewritten.append(goal)
                continue

            dist_to_goal = self._distance_between_locations(ego_loc, goal.transform.location)
            if dist_to_goal <= dist_to_target + replace_margin:
                rewritten.append(goal)
                continue

            entry_tf = self._make_transform_near_target_from_ego_side(
                ego_loc,
                target_transform,
                float(entry_distance),
            )
            rewritten.append(SemanticGoal(
                transform=entry_tf,
                goal_type=goal.goal_type,
                max_steps=goal.max_steps,
                target_speed=goal.target_speed,
            ))
            replaced.append(goal.goal_type)

        if replaced:
            print(
                "[HierarchicalNav] Ego-side entry goals: replaced "
                f"{','.join(replaced)} because original back goals were farther than the target"
            )
        return rewritten
    
    def _get_vehicle_speed(self, vehicle: carla.Vehicle) -> float:
        """获取车辆速度（m/s）"""
        vel = vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def _compute_direct_close_in_control(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        cfg: Dict[str, Any],
    ):
        ego_tf = ego_vehicle.get_transform()
        target_tf = target_vehicle.get_transform()
        ego_loc = ego_tf.location
        target_loc = target_tf.location
        dx = target_loc.x - ego_loc.x
        dy = target_loc.y - ego_loc.y
        dist = math.sqrt(dx * dx + dy * dy)

        yaw = math.radians(float(ego_tf.rotation.yaw))
        x_local = dx * math.cos(yaw) + dy * math.sin(yaw)
        y_local = -dx * math.sin(yaw) + dy * math.cos(yaw)
        lookahead = max(float(cfg.get("lookahead", 4.0)), dist, 1e-3)
        curvature = 2.0 * y_local / (lookahead * lookahead)
        steer = math.atan(curvature * 2.9) / 0.7
        steer = max(-1.0, min(1.0, steer))

        target_dist = float(cfg.get("target_dist", 3.3))
        max_speed = float(cfg.get("max_speed", 2.0))
        min_speed = float(cfg.get("min_speed", 0.35))
        k_speed = float(cfg.get("k_speed", 0.75))
        current_speed = self._get_vehicle_speed(ego_vehicle)

        if x_local <= 0.0 or dist <= target_dist:
            throttle = 0.0
            brake = 1.0 if current_speed > 0.15 else 0.4
            hand_brake = dist <= target_dist and current_speed < 0.3
        else:
            desired_speed = min(max_speed, max(min_speed, k_speed * max(0.0, dist - target_dist)))
            if current_speed > desired_speed + 0.25:
                throttle = 0.0
                brake = min(0.8, 0.25 + 0.25 * (current_speed - desired_speed))
            else:
                throttle = min(0.45, 0.18 + 0.18 * max(0.0, desired_speed - current_speed))
                brake = 0.0
            hand_brake = False

        control = ControlOutput(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
            hand_brake=bool(hand_brake),
        )
        x_rel, y_rel = get_rel_xy(ego_tf, target_tf)
        diag = {
            "dock_submode": "DIRECT_CLOSE_IN",
            "e_y": None,
            "e_psi_deg": None,
            "ref_yaw_deg": float(target_tf.rotation.yaw),
        }
        return control, x_rel, y_rel, diag

    def _compute_target_radius_control(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        cfg: Dict[str, Any],
    ) -> carla.VehicleControl:
        ego_tf = ego_vehicle.get_transform()
        target_tf = target_vehicle.get_transform()
        ego_loc = ego_tf.location
        target_loc = target_tf.location
        dx = target_loc.x - ego_loc.x
        dy = target_loc.y - ego_loc.y
        dist = math.sqrt(dx * dx + dy * dy)

        yaw = math.radians(float(ego_tf.rotation.yaw))
        x_local = dx * math.cos(yaw) + dy * math.sin(yaw)
        y_local = -dx * math.sin(yaw) + dy * math.cos(yaw)

        stop_radius = float(cfg.get("stop_radius", 3.6))
        current_speed = self._get_vehicle_speed(ego_vehicle)
        if dist <= stop_radius:
            return carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
                hand_brake=current_speed < 0.3,
            )

        if (
            cfg.get("reverse_enabled", False)
            and x_local < -0.5
            and dist <= float(cfg.get("reverse_dist", 9.0))
        ):
            steer = -0.65 if y_local >= 0.0 else 0.65
            if abs(y_local) < 1.0:
                steer = 0.0
            throttle = float(cfg.get("reverse_throttle", 0.22))
            return carla.VehicleControl(
                throttle=max(0.0, min(0.45, throttle)),
                brake=0.0,
                steer=float(steer),
                reverse=True,
                hand_brake=False,
                manual_gear_shift=False,
            )

        turnaround_dist = float(cfg.get("turnaround_dist", 12.0))
        if x_local < -1.0 and dist <= turnaround_dist:
            steer = 0.85 if y_local >= 0.0 else -0.85
            return carla.VehicleControl(
                throttle=0.25,
                brake=0.0,
                steer=steer,
                reverse=False,
                hand_brake=False,
                manual_gear_shift=False,
            )

        base_lookahead = max(float(cfg.get("lookahead", 5.0)), 1e-3)
        if cfg.get("fixed_lookahead", False):
            lookahead = base_lookahead
        elif cfg.get("lookahead_dist_cap") is not None:
            lookahead = max(base_lookahead, min(dist, float(cfg.get("lookahead_dist_cap", dist))))
        else:
            lookahead = max(base_lookahead, dist, 1e-3)
        curvature = 2.0 * y_local / (lookahead * lookahead)
        steer = math.atan(curvature * 2.9) / 0.7
        steer = max(-1.0, min(1.0, steer))

        max_speed = float(cfg.get("near_max_speed", 2.2))
        min_speed = float(cfg.get("near_min_speed", 0.35))
        k_speed = float(cfg.get("near_k_speed", 0.8))
        desired_speed = min(max_speed, max(min_speed, k_speed * max(0.0, dist - stop_radius)))

        if current_speed > desired_speed + 0.3:
            throttle = 0.0
            brake = min(0.9, 0.2 + 0.25 * (current_speed - desired_speed))
        else:
            max_throttle = float(cfg.get("max_throttle", 0.55))
            throttle = min(max_throttle, 0.16 + 0.16 * max(0.0, desired_speed - current_speed))
            if current_speed < 0.15 and dist > stop_radius + 0.4:
                throttle = max(throttle, float(cfg.get("stuck_boost_throttle", 0.0)))
            brake = 0.0

        return carla.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
            reverse=False,
            hand_brake=False,
            manual_gear_shift=False,
        )

    def _run_direct_recovery_tail(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        cfg: Dict[str, Any],
        max_steps: int,
        min_dist_in: float,
        safety_monitor: Optional[SafetyEventMonitor] = None,
    ) -> Dict[str, Any]:
        min_dist = float(min_dist_in)
        final_dist = float(min_dist_in)
        trajectory: List[Dict[str, Any]] = []
        steps = 0

        for step in range(max(0, int(max_steps))):
            ego_tf = ego_vehicle.get_transform()
            target_tf = target_vehicle.get_transform()
            ego_loc = ego_tf.location
            target_loc = target_tf.location
            dist = self._distance_between_locations(ego_loc, target_loc)
            min_dist = min(min_dist, dist)
            final_dist = dist

            x_rel, y_rel = get_rel_xy(ego_tf, target_tf)
            psi_rel = float(ego_tf.rotation.yaw - target_tf.rotation.yaw)
            while psi_rel > 180:
                psi_rel -= 360
            while psi_rel < -180:
                psi_rel += 360

            if self.record_trajectory and step % self.traj_subsample == 0:
                trajectory.append({
                    "step": step,
                    "x": float(ego_loc.x),
                    "y": float(ego_loc.y),
                    "z": float(ego_loc.z),
                    "yaw": float(ego_tf.rotation.yaw),
                    "speed": self._get_vehicle_speed(ego_vehicle),
                    "dist": float(dist),
                    "phase": "DIRECT_TAIL",
                    "x_rel": float(x_rel),
                    "y_rel": float(y_rel),
                    "psi_rel_deg": float(psi_rel),
                    "dock_mode": False,
                    "dock_submode": "POST_PHASE_TARGET_RADIUS",
                    "e_y": None,
                    "e_psi_deg": None,
                    "ref_yaw_deg": float(target_tf.rotation.yaw),
                })

            if min_dist < self.success_radius_4m:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.world.tick()
                steps = step + 1
                if safety_monitor is not None:
                    safety_monitor.update_ttc(ego_vehicle)
                break

            control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)
            ego_vehicle.apply_control(control)
            self.world.tick()
            steps = step + 1
            if safety_monitor is not None:
                safety_monitor.update_ttc(ego_vehicle)

        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        final_dist = self._distance_between_locations(ego_loc, target_loc)
        min_dist = min(min_dist, final_dist)

        return {
            "phase": "DIRECT_TAIL",
            "success": bool(min_dist < self.success_radius_4m),
            "min_dist": float(min_dist),
            "final_dist": float(final_dist),
            "steps": int(steps),
            "reason": "post_phase_target_radius",
            "trajectory": trajectory,
        }

    def _run_direct_acquisition_episode(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        start_dist: float,
        max_steps_total: int,
    ) -> NavigationResult:
        cfg = self.controller_profile_config.get("direct_acquisition", {})
        safety_monitor = SafetyEventMonitor(self.world, ego_vehicle, enabled=self.record_safety)
        self.pure_pursuit.reset()
        self.speed_pid.reset()
        self.current_route = []
        basic_agent = None
        if cfg.get("use_basic_agent", False):
            try:
                _ensure_carla_agents_path()
                from agents.navigation.basic_agent import BasicAgent
                basic_agent = BasicAgent(
                    ego_vehicle,
                    target_speed=float(cfg.get("agent_target_speed_kmh", 28.0)),
                )
                if hasattr(basic_agent, "ignore_traffic_lights"):
                    basic_agent.ignore_traffic_lights(True)
                if hasattr(basic_agent, "ignore_stop_signs"):
                    basic_agent.ignore_stop_signs(True)
                if hasattr(basic_agent, "ignore_vehicles"):
                    basic_agent.ignore_vehicles(False)
                basic_agent.set_destination(target_vehicle.get_transform().location)
                print("[DirectAcq] BasicAgent enabled for far-range acquisition")
            except Exception as exc:
                print(f"[DirectAcq] BasicAgent unavailable, falling back to route/direct control: {exc}")
                basic_agent = None

        min_dist = float(start_dist)
        final_dist = float(start_dist)
        success_4m = start_dist < self.success_radius_4m
        success_8m = start_dist < self.success_radius_8m
        trajectory: List[Dict[str, Any]] = []
        route_replan_steps = max(1, int(cfg.get("route_replan_steps", 40)))
        direct_enter_dist = float(cfg.get("direct_enter_dist", 14.0))
        stuck = False

        for step in range(max_steps_total):
            ego_tf = ego_vehicle.get_transform()
            target_tf = target_vehicle.get_transform()
            ego_loc = ego_tf.location
            target_loc = target_tf.location
            dist = self._distance_between_locations(ego_loc, target_loc)
            min_dist = min(min_dist, dist)
            final_dist = dist

            if min_dist < self.success_radius_4m:
                success_4m = True
                success_8m = True
            elif min_dist < self.success_radius_8m:
                success_8m = True

            x_rel, y_rel = get_rel_xy(ego_tf, target_tf)
            psi_rel = float(ego_tf.rotation.yaw - target_tf.rotation.yaw)
            while psi_rel > 180:
                psi_rel -= 360
            while psi_rel < -180:
                psi_rel += 360

            if self.record_trajectory and step % self.traj_subsample == 0:
                trajectory.append({
                    "step": step,
                    "x": float(ego_loc.x),
                    "y": float(ego_loc.y),
                    "z": float(ego_loc.z),
                    "yaw": float(ego_tf.rotation.yaw),
                    "speed": self._get_vehicle_speed(ego_vehicle),
                    "dist": float(dist),
                    "phase": "DIRECT",
                    "x_rel": float(x_rel),
                    "y_rel": float(y_rel),
                    "psi_rel_deg": float(psi_rel),
                    "dock_mode": False,
                    "dock_submode": "TARGET_RADIUS",
                    "e_y": None,
                    "e_psi_deg": None,
                    "ref_yaw_deg": float(target_tf.rotation.yaw),
                })

            if dist <= float(cfg.get("stop_radius", 3.6)):
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.world.tick()
                if safety_monitor is not None:
                    safety_monitor.update_ttc(ego_vehicle)
                break

            if dist <= direct_enter_dist:
                carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)
            elif basic_agent is not None:
                try:
                    if step % route_replan_steps == 0:
                        basic_agent.set_destination(target_loc)
                    carla_control = basic_agent.run_step()
                except Exception as exc:
                    print(f"[DirectAcq] BasicAgent step failed, falling back to route/direct control: {exc}")
                    basic_agent = None
                    carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)
            else:
                if step % route_replan_steps == 0 or not self.current_route:
                    try:
                        self.current_route = self.route_planner.plan_route_between(ego_loc, target_loc)
                        self.pure_pursuit.reset()
                    except Exception as exc:
                        print(f"[DirectAcq] Route planning failed: {exc}")
                        self.current_route = []

                if self.current_route:
                    steer = self.pure_pursuit.compute_steer(ego_tf, self.current_route)
                    current_speed = self._get_vehicle_speed(ego_vehicle)
                    target_speed = float(cfg.get("far_speed", 9.0)) if dist > 25.0 else float(cfg.get("mid_speed", 5.5))
                    throttle, brake = self.speed_pid.compute_control(target_speed, current_speed)
                    carla_control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
                else:
                    carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)

            ego_vehicle.apply_control(carla_control)
            self.world.tick()
            if safety_monitor is not None:
                safety_monitor.update_ttc(ego_vehicle)

        safety_summary = safety_monitor.summary()
        safety_monitor.close()
        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        final_dist = self._distance_between_locations(ego_loc, target_loc)
        min_dist = min(min_dist, final_dist)
        if min_dist < self.success_radius_4m:
            success_4m = True
            success_8m = True
        elif min_dist < self.success_radius_8m:
            success_8m = True

        return NavigationResult(
            start_dist=start_dist,
            min_dist=min_dist,
            final_dist=final_dist,
            steps=step + 1 if "step" in locals() else 0,
            success_4m=success_4m,
            success_8m=success_8m,
            stuck=stuck,
            trajectory=trajectory if self.record_trajectory else [],
            phase_results={
                "DIRECT": {
                    "phase": "DIRECT",
                    "success": bool(success_4m or success_8m),
                    "min_dist": float(min_dist),
                    "final_dist": float(final_dist),
                    "steps": int(step + 1 if "step" in locals() else 0),
                    "reason": "target_radius_acquisition" if success_4m else "timeout",
                }
            },
            safety=safety_summary,
        )

    def _run_behavior_acquisition_episode(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        start_dist: float,
        max_steps_total: int,
    ) -> NavigationResult:
        cfg = self.controller_profile_config.get("behavior_acquisition", {})
        safety_monitor = SafetyEventMonitor(self.world, ego_vehicle, enabled=self.record_safety)
        min_dist = float(start_dist)
        final_dist = float(start_dist)
        success_4m = start_dist < self.success_radius_4m
        success_8m = start_dist < self.success_radius_8m
        trajectory: List[Dict[str, Any]] = []
        mode = "APPROACH"
        agent = None

        try:
            _ensure_carla_agents_path()
            from agents.navigation.behavior_agent import BehaviorAgent
            agent = BehaviorAgent(ego_vehicle, behavior=str(cfg.get("behavior", "aggressive")))
            try:
                from .target_route_planner import generate_approach_location
                dest = generate_approach_location(
                    self.world,
                    target_vehicle,
                    back_distance=float(cfg.get("approach_back_distance", 6.0)),
                    random_offset=float(cfg.get("approach_random_offset", 0.0)),
                )
            except Exception as exc:
                print(f"[BehaviorAcq] Approach-point generation failed: {exc}")
                dest = None
            if dest is None:
                dest = target_vehicle.get_transform().location
                mode = "TARGET"
            agent.set_destination(dest)
            print(f"[BehaviorAcq] BehaviorAgent enabled, mode={mode}")
        except Exception as exc:
            print(f"[BehaviorAcq] BehaviorAgent unavailable, using direct target-radius control: {exc}")
            agent = None
            mode = "DIRECT"

        direct_enter_dist = float(cfg.get("direct_enter_dist", 12.0))
        switch_to_target_dist = float(cfg.get("switch_to_target_dist", 18.0))
        replan_steps = max(1, int(cfg.get("replan_steps", 80)))
        stuck = False

        for step in range(max_steps_total):
            ego_tf = ego_vehicle.get_transform()
            target_tf = target_vehicle.get_transform()
            ego_loc = ego_tf.location
            target_loc = target_tf.location
            dist = self._distance_between_locations(ego_loc, target_loc)
            min_dist = min(min_dist, dist)
            final_dist = dist

            if min_dist < self.success_radius_4m:
                success_4m = True
                success_8m = True
            elif min_dist < self.success_radius_8m:
                success_8m = True

            x_rel, y_rel = get_rel_xy(ego_tf, target_tf)
            psi_rel = float(ego_tf.rotation.yaw - target_tf.rotation.yaw)
            while psi_rel > 180:
                psi_rel -= 360
            while psi_rel < -180:
                psi_rel += 360

            if self.record_trajectory and step % self.traj_subsample == 0:
                trajectory.append({
                    "step": step,
                    "x": float(ego_loc.x),
                    "y": float(ego_loc.y),
                    "z": float(ego_loc.z),
                    "yaw": float(ego_tf.rotation.yaw),
                    "speed": self._get_vehicle_speed(ego_vehicle),
                    "dist": float(dist),
                    "phase": mode,
                    "x_rel": float(x_rel),
                    "y_rel": float(y_rel),
                    "psi_rel_deg": float(psi_rel),
                    "dock_mode": mode == "DIRECT",
                    "dock_submode": "BEHAVIOR_TARGET_RADIUS" if mode == "DIRECT" else "",
                    "e_y": None,
                    "e_psi_deg": None,
                    "ref_yaw_deg": float(target_tf.rotation.yaw),
                })

            if min_dist < self.success_radius_4m:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.world.tick()
                if safety_monitor is not None:
                    safety_monitor.update_ttc(ego_vehicle)
                break

            if dist <= direct_enter_dist:
                mode = "DIRECT"
                carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)
            elif agent is not None:
                if mode == "APPROACH" and dist <= switch_to_target_dist:
                    try:
                        agent.set_destination(target_loc)
                        mode = "TARGET"
                    except Exception as exc:
                        print(f"[BehaviorAcq] Switch to target failed: {exc}")
                elif mode == "TARGET" and step % replan_steps == 0:
                    try:
                        agent.set_destination(target_loc)
                    except Exception:
                        pass
                try:
                    carla_control = agent.run_step()
                    if hasattr(carla_control, "throttle") and dist > 25.0:
                        carla_control.throttle = max(float(carla_control.throttle), 0.45)
                except Exception as exc:
                    print(f"[BehaviorAcq] Agent step failed, using direct control: {exc}")
                    agent = None
                    mode = "DIRECT"
                    carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)
            else:
                mode = "DIRECT"
                carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)

            ego_vehicle.apply_control(carla_control)
            self.world.tick()
            if safety_monitor is not None:
                safety_monitor.update_ttc(ego_vehicle)

        safety_summary = safety_monitor.summary()
        safety_monitor.close()
        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        final_dist = self._distance_between_locations(ego_loc, target_loc)
        min_dist = min(min_dist, final_dist)
        if min_dist < self.success_radius_4m:
            success_4m = True
            success_8m = True
        elif min_dist < self.success_radius_8m:
            success_8m = True

        return NavigationResult(
            start_dist=start_dist,
            min_dist=min_dist,
            final_dist=final_dist,
            steps=step + 1 if "step" in locals() else 0,
            success_4m=success_4m,
            success_8m=success_8m,
            stuck=stuck,
            trajectory=trajectory if self.record_trajectory else [],
            phase_results={
                "BEHAVIOR_ACQ": {
                    "phase": "BEHAVIOR_ACQ",
                    "success": bool(success_4m or success_8m),
                    "min_dist": float(min_dist),
                    "final_dist": float(final_dist),
                    "steps": int(step + 1 if "step" in locals() else 0),
                    "reason": "behavior_target_acquisition" if success_4m else "timeout",
                }
            },
            safety=safety_summary,
        )

    def _make_multi_approach_goal(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        approach_cfg: Dict[str, Any],
    ) -> SemanticGoal:
        target_tf = target_vehicle.get_transform()
        if approach_cfg.get("type") == "ego_side":
            goal_tf = self._make_transform_near_target_from_ego_side(
                ego_vehicle.get_transform().location,
                target_tf,
                float(approach_cfg.get("distance", 10.0)),
            )
        else:
            goal_tf = make_back_goal_from_target(
                target_tf,
                self.carla_map,
                back_dist=float(approach_cfg.get("back_dist", 10.0)),
                lat_offset=float(approach_cfg.get("lat_offset", 0.0)),
            )
        return SemanticGoal(
            transform=goal_tf,
            goal_type=str(approach_cfg.get("goal_type", "G2")),
            max_steps=int(approach_cfg.get("max_steps", 300)),
            target_speed=float(approach_cfg.get("target_speed", 6.0)),
        )

    def _run_multi_approach_sequence(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        cfg: Dict[str, Any],
        max_steps: int,
        min_dist_in: float,
        safety_monitor: Optional[SafetyEventMonitor] = None,
        key_prefix: str = "POST",
    ) -> Dict[str, Any]:
        approaches = list(cfg.get("approaches", []))
        min_dist = float(min_dist_in)
        final_dist = float(min_dist_in)
        total_steps = 0
        phase_results: Dict[str, Dict[str, Any]] = {}
        all_trajectory: List[Dict[str, Any]] = []

        for idx, approach_cfg in enumerate(approaches):
            if total_steps >= max_steps or min_dist < self.success_radius_4m:
                break
            remaining_steps = max(0, int(max_steps) - int(total_steps))
            try:
                goal = self._make_multi_approach_goal(ego_vehicle, target_vehicle, approach_cfg)
            except Exception as exc:
                print(f"[MultiApproach] Failed to build post approach {idx}: {exc}")
                continue
            phase_steps = min(int(goal.max_steps), remaining_steps)
            if phase_steps <= 0:
                break

            name = str(approach_cfg.get("name", f"approach_{idx}"))
            print(f"[MultiApproach] Running {key_prefix} {name}: steps={phase_steps}")
            phase_result = self._run_phase(
                ego_vehicle,
                target_vehicle,
                goal,
                phase_steps,
                min_dist,
                safety_monitor=safety_monitor,
            )
            phase_key = f"{key_prefix}_APPROACH_{idx}_{name}"
            phase_results[phase_key] = phase_result
            total_steps += int(phase_result.get("steps", 0))
            min_dist = min(min_dist, float(phase_result.get("min_dist", min_dist)))
            final_dist = float(phase_result.get("final_dist", final_dist))
            if "trajectory" in phase_result:
                all_trajectory.extend(phase_result["trajectory"])

            if bool(cfg.get("direct_after_each", True)) and total_steps < max_steps and min_dist >= self.success_radius_4m:
                remaining_steps = max(0, int(max_steps) - int(total_steps))
                direct_steps = min(int(cfg.get("direct_steps", 200)), remaining_steps)
                if direct_steps > 0:
                    tail_result = self._run_direct_recovery_tail(
                        ego_vehicle=ego_vehicle,
                        target_vehicle=target_vehicle,
                        cfg=cfg,
                        max_steps=direct_steps,
                        min_dist_in=min_dist,
                        safety_monitor=safety_monitor,
                    )
                    tail_key = f"{key_prefix}_DIRECT_AFTER_{idx}_{name}"
                    phase_results[tail_key] = tail_result
                    total_steps += int(tail_result.get("steps", 0))
                    min_dist = min(min_dist, float(tail_result.get("min_dist", min_dist)))
                    final_dist = float(tail_result.get("final_dist", final_dist))
                    if "trajectory" in tail_result:
                        all_trajectory.extend(tail_result["trajectory"])

        if total_steps < max_steps and min_dist >= self.success_radius_4m:
            remaining_steps = max(0, int(max_steps) - int(total_steps))
            final_direct_steps = min(int(cfg.get("final_direct_steps", 0)), remaining_steps)
            if final_direct_steps > 0:
                print(f"[MultiApproach] Running {key_prefix} final direct tail: steps={final_direct_steps}")
                tail_result = self._run_direct_recovery_tail(
                    ego_vehicle=ego_vehicle,
                    target_vehicle=target_vehicle,
                    cfg=cfg,
                    max_steps=final_direct_steps,
                    min_dist_in=min_dist,
                    safety_monitor=safety_monitor,
                )
                phase_results[f"{key_prefix}_DIRECT_FINAL"] = tail_result
                total_steps += int(tail_result.get("steps", 0))
                min_dist = min(min_dist, float(tail_result.get("min_dist", min_dist)))
                final_dist = float(tail_result.get("final_dist", final_dist))
                if "trajectory" in tail_result:
                    all_trajectory.extend(tail_result["trajectory"])

        return {
            "steps": int(total_steps),
            "min_dist": float(min_dist),
            "final_dist": float(final_dist),
            "phase_results": phase_results,
            "trajectory": all_trajectory,
        }

    def _run_behavior_recovery_sequence(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        cfg: Dict[str, Any],
        max_steps: int,
        min_dist_in: float,
        safety_monitor: Optional[SafetyEventMonitor] = None,
    ) -> Dict[str, Any]:
        min_dist = float(min_dist_in)
        final_dist = float(min_dist_in)
        trajectory: List[Dict[str, Any]] = []
        mode = "APPROACH"
        agent = None
        steps = 0

        try:
            _ensure_carla_agents_path()
            from agents.navigation.behavior_agent import BehaviorAgent
            agent = BehaviorAgent(ego_vehicle, behavior=str(cfg.get("behavior", "aggressive")))
            try:
                from .target_route_planner import generate_approach_location
                dest = generate_approach_location(
                    self.world,
                    target_vehicle,
                    back_distance=float(cfg.get("approach_back_distance", 6.0)),
                    random_offset=float(cfg.get("approach_random_offset", 0.0)),
                )
            except Exception as exc:
                print(f"[BehaviorRecovery] Approach-point generation failed: {exc}")
                dest = None
            if dest is None:
                dest = target_vehicle.get_transform().location
                mode = "TARGET"
            agent.set_destination(dest)
            print(f"[BehaviorRecovery] BehaviorAgent enabled, mode={mode}")
        except Exception as exc:
            print(f"[BehaviorRecovery] BehaviorAgent unavailable: {exc}")
            agent = None
            mode = "DIRECT"

        direct_enter_dist = float(cfg.get("direct_enter_dist", 12.0))
        switch_to_target_dist = float(cfg.get("switch_to_target_dist", 18.0))
        replan_steps = max(1, int(cfg.get("replan_steps", 80)))

        for step in range(max(0, int(max_steps))):
            ego_tf = ego_vehicle.get_transform()
            target_tf = target_vehicle.get_transform()
            ego_loc = ego_tf.location
            target_loc = target_tf.location
            dist = self._distance_between_locations(ego_loc, target_loc)
            min_dist = min(min_dist, dist)
            final_dist = dist

            x_rel, y_rel = get_rel_xy(ego_tf, target_tf)
            psi_rel = float(ego_tf.rotation.yaw - target_tf.rotation.yaw)
            while psi_rel > 180:
                psi_rel -= 360
            while psi_rel < -180:
                psi_rel += 360

            if self.record_trajectory and step % self.traj_subsample == 0:
                trajectory.append({
                    "step": step,
                    "x": float(ego_loc.x),
                    "y": float(ego_loc.y),
                    "z": float(ego_loc.z),
                    "yaw": float(ego_tf.rotation.yaw),
                    "speed": self._get_vehicle_speed(ego_vehicle),
                    "dist": float(dist),
                    "phase": f"BEHAVIOR_RECOVERY_{mode}",
                    "x_rel": float(x_rel),
                    "y_rel": float(y_rel),
                    "psi_rel_deg": float(psi_rel),
                    "dock_mode": mode == "DIRECT",
                    "dock_submode": "BEHAVIOR_RECOVERY_RADIUS" if mode == "DIRECT" else "",
                    "e_y": None,
                    "e_psi_deg": None,
                    "ref_yaw_deg": float(target_tf.rotation.yaw),
                })

            if min_dist < self.success_radius_4m:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                self.world.tick()
                steps = step + 1
                if safety_monitor is not None:
                    safety_monitor.update_ttc(ego_vehicle)
                break

            if dist <= direct_enter_dist or agent is None:
                mode = "DIRECT"
                carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)
            else:
                if mode == "APPROACH" and dist <= switch_to_target_dist:
                    try:
                        agent.set_destination(target_loc)
                        mode = "TARGET"
                    except Exception as exc:
                        print(f"[BehaviorRecovery] Switch to target failed: {exc}")
                elif mode == "TARGET" and step % replan_steps == 0:
                    try:
                        agent.set_destination(target_loc)
                    except Exception:
                        pass
                try:
                    carla_control = agent.run_step()
                    if hasattr(carla_control, "throttle") and dist > 25.0:
                        carla_control.throttle = max(float(carla_control.throttle), 0.45)
                except Exception as exc:
                    print(f"[BehaviorRecovery] Agent step failed, using direct control: {exc}")
                    agent = None
                    mode = "DIRECT"
                    carla_control = self._compute_target_radius_control(ego_vehicle, target_vehicle, cfg)

            ego_vehicle.apply_control(carla_control)
            self.world.tick()
            steps = step + 1
            if safety_monitor is not None:
                safety_monitor.update_ttc(ego_vehicle)

        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        final_dist = self._distance_between_locations(ego_loc, target_loc)
        min_dist = min(min_dist, final_dist)
        return {
            "phase": "BEHAVIOR_RECOVERY",
            "success": bool(min_dist < self.success_radius_4m or min_dist < self.success_radius_8m),
            "min_dist": float(min_dist),
            "final_dist": float(final_dist),
            "steps": int(steps),
            "reason": "behavior_recovery",
            "trajectory": trajectory,
        }

    def _run_multi_approach_acquisition_episode(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        start_dist: float,
        max_steps_total: int,
    ) -> NavigationResult:
        cfg = self.controller_profile_config.get("multi_approach_acquisition", {})
        approaches = list(cfg.get("approaches", []))
        safety_monitor = SafetyEventMonitor(self.world, ego_vehicle, enabled=self.record_safety)
        self.pure_pursuit.reset()
        self.speed_pid.reset()
        self.current_route = []

        min_dist = float(start_dist)
        final_dist = float(start_dist)
        total_steps = 0
        success_4m = start_dist < self.success_radius_4m
        success_8m = start_dist < self.success_radius_8m
        stuck = False
        phase_results: Dict[str, Dict[str, Any]] = {}
        all_trajectory: List[Dict[str, Any]] = []

        def update_success() -> None:
            nonlocal success_4m, success_8m
            if min_dist < self.success_radius_4m:
                success_4m = True
                success_8m = True
            elif min_dist < self.success_radius_8m:
                success_8m = True

        for idx, approach_cfg in enumerate(approaches):
            if total_steps >= max_steps_total or success_4m:
                break
            remaining_steps = max(0, int(max_steps_total) - int(total_steps))
            try:
                goal = self._make_multi_approach_goal(ego_vehicle, target_vehicle, approach_cfg)
            except Exception as exc:
                print(f"[MultiApproach] Failed to build approach {idx}: {exc}")
                continue
            phase_steps = min(int(goal.max_steps), remaining_steps)
            if phase_steps <= 0:
                break

            name = str(approach_cfg.get("name", f"approach_{idx}"))
            print(f"[MultiApproach] Running {name}: steps={phase_steps}")
            phase_result = self._run_phase(
                ego_vehicle,
                target_vehicle,
                goal,
                phase_steps,
                min_dist,
                safety_monitor=safety_monitor,
            )
            phase_key = f"APPROACH_{idx}_{name}"
            phase_results[phase_key] = phase_result
            total_steps += int(phase_result.get("steps", 0))
            min_dist = min(min_dist, float(phase_result.get("min_dist", min_dist)))
            final_dist = float(phase_result.get("final_dist", final_dist))
            if "trajectory" in phase_result:
                all_trajectory.extend(phase_result["trajectory"])
            update_success()
            if success_4m:
                break

            if bool(cfg.get("direct_after_each", True)) and total_steps < max_steps_total:
                remaining_steps = max(0, int(max_steps_total) - int(total_steps))
                direct_steps = min(int(cfg.get("direct_steps", 200)), remaining_steps)
                if direct_steps > 0:
                    tail_result = self._run_direct_recovery_tail(
                        ego_vehicle=ego_vehicle,
                        target_vehicle=target_vehicle,
                        cfg=cfg,
                        max_steps=direct_steps,
                        min_dist_in=min_dist,
                        safety_monitor=safety_monitor,
                    )
                    tail_key = f"DIRECT_AFTER_{idx}_{name}"
                    phase_results[tail_key] = tail_result
                    total_steps += int(tail_result.get("steps", 0))
                    min_dist = min(min_dist, float(tail_result.get("min_dist", min_dist)))
                    final_dist = float(tail_result.get("final_dist", final_dist))
                    if "trajectory" in tail_result:
                        all_trajectory.extend(tail_result["trajectory"])
                    update_success()

        if (not success_4m) and total_steps < max_steps_total:
            remaining_steps = max(0, int(max_steps_total) - int(total_steps))
            final_direct_steps = min(int(cfg.get("final_direct_steps", 500)), remaining_steps)
            if final_direct_steps > 0:
                print(f"[MultiApproach] Running final direct tail: steps={final_direct_steps}")
                tail_result = self._run_direct_recovery_tail(
                    ego_vehicle=ego_vehicle,
                    target_vehicle=target_vehicle,
                    cfg=cfg,
                    max_steps=final_direct_steps,
                    min_dist_in=min_dist,
                    safety_monitor=safety_monitor,
                )
                phase_results["DIRECT_FINAL"] = tail_result
                total_steps += int(tail_result.get("steps", 0))
                min_dist = min(min_dist, float(tail_result.get("min_dist", min_dist)))
                final_dist = float(tail_result.get("final_dist", final_dist))
                if "trajectory" in tail_result:
                    all_trajectory.extend(tail_result["trajectory"])
                update_success()

        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        final_dist = self._distance_between_locations(ego_loc, target_loc)
        min_dist = min(min_dist, final_dist)
        update_success()
        safety_summary = safety_monitor.summary()
        safety_monitor.close()

        return NavigationResult(
            start_dist=start_dist,
            min_dist=min_dist,
            final_dist=final_dist,
            steps=int(total_steps),
            success_4m=success_4m,
            success_8m=success_8m,
            stuck=stuck,
            trajectory=all_trajectory if self.record_trajectory else [],
            phase_results=phase_results,
            safety=safety_summary,
        )

    def _should_skip_initial_route_waypoint(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
    ) -> bool:
        if not self.controller_profile_config.get("skip_initial_route_waypoint", False):
            return False
        min_start_distance = self.controller_profile_config.get("skip_initial_route_min_start_distance")
        if min_start_distance is None:
            return True
        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        current_dist = self._distance_between_locations(ego_loc, target_loc)
        return current_dist >= float(min_start_distance)
    
    def _run_phase(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        goal: SemanticGoal,
        max_steps: int,
        start_dist: float,
        safety_monitor: Optional[SafetyEventMonitor] = None,
    ) -> Dict[str, Any]:
        """
        运行单个阶段
        
        Args:
            ego_vehicle: ego车辆
            target_vehicle: 目标车辆
            goal: 语义子目标
            max_steps: 最大步数
            start_dist: 起始距离
        
        Returns:
            阶段结果字典
        """
        phase = goal.goal_type
        self.current_phase = phase
        
        # 重置控制器状态
        self.pure_pursuit.reset()
        self.speed_pid.reset()
        self.stuck_detector.reset()
        self.recovery_policy.reset(phase)
        
        # 获取ego和目标车辆的waypoint
        ego_wp = self.route_planner.get_waypoint_at_location(
            ego_vehicle.get_transform().location
        )
        goal_wp = self.goal_manager.get_goal_waypoint(goal)
        
        # G3(DOCK)阶段不依赖goal waypoint做路径规划：即使goal_wp解析失败，也应该继续进入DOCK控制。
        # 否则会出现“永远进不了G3”的假性失败（尤其在路口/分叉处get_waypoint可能返回None）。
        if phase == "G3":
            if ego_wp is None:
                return {
                    "phase": phase,
                    "success": False,
                    "steps": 0,
                    "min_dist": start_dist,
                    "final_dist": start_dist,
                    "reason": "waypoint_not_found"
                }
        else:
            if ego_wp is None or goal_wp is None:
                return {
                    "phase": phase,
                    "success": False,
                    "steps": 0,
                    "min_dist": start_dist,
                    "final_dist": start_dist,
                    "reason": "waypoint_not_found"
                }
        
        # 规划路径
        if phase == "G3":
            # G3阶段使用DOCK控制，不需要路径规划
            self.current_route = []
        else:
            self.current_route = self.route_planner.plan_route_from_waypoint(
                ego_wp, goal_wp
            )
            if self._should_skip_initial_route_waypoint(ego_vehicle, target_vehicle):
                if len(self.current_route) > 1:
                    self.current_route = self.current_route[1:]
            if not self.current_route:
                return {
                    "phase": phase,
                    "success": False,
                    "steps": 0,
                    "min_dist": start_dist,
                    "final_dist": start_dist,
                    "reason": "route_planning_failed"
                }
        
        # 阶段循环
        min_dist = start_dist
        steps = 0
        trajectory = []
        
        # 进度监控（用于检测是否陷入局部最优）
        progress_window_size = 20  # 每20步检查一次进度
        progress_windows = []  # 记录每个窗口的最小距离
        last_progress_check_step = 0
        
        # G3阶段失败检测
        g3_enter_dist = None
        g3_best_dist = None
        g3_steps = 0

        # === Step 3.2(B): Dock阶段切换滞回 + 最短停留时间（防抖）===
        # 仅在近距（G2/G3）启用一个小状态机：TARGET(追踪路径) ↔ DOCK(对接几何控制)
        # 这样可以避免“G2只重规划到G3 waypoint，但不进入Dock控制”的问题，从而提升4m成功率。
        try:
            dt = float(self.world.get_settings().fixed_delta_seconds or 0.05)
        except Exception:
            dt = 0.05
        dock_soft_cfg = self.controller_profile_config.get("dock_g2_soft", {})
        d2_enter = float(dock_soft_cfg.get("enter_dist", 15.0))
        d2_exit = float(dock_soft_cfg.get("exit_dist", 18.0))
        min_dwell_seconds = float(dock_soft_cfg.get("min_dwell_seconds", 1.0))
        min_dwell_steps = max(1, int(round(min_dwell_seconds / max(1e-6, dt))))
        dock_trigger_cfg = self.controller_profile_config.get("dock_trigger", {})
        d2_x_rel_min = float(dock_trigger_cfg.get("x_rel", 2.0))
        d2_y_rel_max = float(dock_trigger_cfg.get("y_rel_max", 5.0))
        d2_yaw_err_max = float(dock_trigger_cfg.get("yaw_err_max", 45.0))
        # 注意：dock_mode 表示“是否启用Dock几何控制”
        # - G3阶段：应始终为True（不要被G2的滞回退出逻辑关掉）
        # - G2阶段：由滞回状态机决定（软切Dock）
        dock_mode = (phase == "G3")
        last_switch_step = 0
        dock_switches = 0
        
        for step in range(max_steps):
            # 更新卡死检测
            is_stuck = self.stuck_detector.update(ego_vehicle)
            
            # 如果卡死且可以恢复，执行恢复
            if is_stuck and self.recovery_policy.can_recover(phase):
                if not self.recovery_policy.is_recovering:
                    self.recovery_policy.start_recovery(ego_vehicle, phase)
                    print(f"[Recovery] Phase {phase} stuck, starting recovery...")
            
            # 如果正在恢复，执行恢复控制
            if self.recovery_policy.is_recovering:
                recovery_control, recovery_done = self.recovery_policy.update_recovery(ego_vehicle)
                ego_vehicle.apply_control(recovery_control)
                self.world.tick()
                if safety_monitor is not None:
                    safety_monitor.update_ttc(ego_vehicle)
                
                if recovery_done:
                    self.recovery_policy.finish_recovery()
                    # 重新规划路径
                    ego_wp = self.route_planner.get_waypoint_at_location(
                        ego_vehicle.get_transform().location
                    )
                    if ego_wp and goal_wp:
                        self.current_route = self.route_planner.plan_route_from_waypoint(
                            ego_wp, goal_wp
                        )
                        self.pure_pursuit.reset()
                    print(f"[Recovery] Phase {phase} recovery done, replanning...")
                
                steps += 1
                continue
            
            # 计算到目标车辆的距离
            ego_loc = ego_vehicle.get_transform().location
            target_loc = target_vehicle.get_transform().location
            dist_to_target = self._distance_between_locations(ego_loc, target_loc)
            min_dist = min(min_dist, dist_to_target)

            # === Dock-Entry 量化统计所需：相对几何（目标车坐标系）===
            # get_rel_xy 定义：x轴指向目标车后方（>0 表示 ego 在后方），y轴指向目标车右侧
            ego_tf_rel = ego_vehicle.get_transform()
            target_tf_rel = target_vehicle.get_transform()
            x_rel_to_target, y_rel_to_target = get_rel_xy(ego_tf_rel, target_tf_rel)
            psi_rel_deg = float(ego_tf_rel.rotation.yaw - target_tf_rel.rotation.yaw)
            while psi_rel_deg > 180:
                psi_rel_deg -= 360
            while psi_rel_deg < -180:
                psi_rel_deg += 360
            
            # 计算到DOCK目标点G3的距离
            dist_to_dock_goal = 0.0
            if phase == "G3":
                # G3阶段：计算到目标车辆的距离（DOCK目标就是目标车辆后方4m）
                dist_to_dock_goal = dist_to_target
            else:
                # G1/G2阶段：计算到目标waypoint的距离
                dist_to_dock_goal = self._distance_between_locations(
                    ego_loc, goal.transform.location
                )
            
            # 确定阶段ID和名称
            phase_id_map = {"G1": 0, "G2": 1, "G3": 2}
            phase_id = phase_id_map.get(phase, -1)
            # 先初始化；真正的 dock_active 在状态机更新后再确定
            in_dock_phase = (phase == "G3")
            
            # 根据阶段选择控制策略
            x_rel = None
            y_rel = None
            throttle = None
            brake = None
            steer = None
            dock_diag = None

            # 在G2阶段启用dock_mode状态机（带滞回和dwell）
            # G3阶段始终Dock，不参与“退出Dock”的滞回逻辑（否则会把dock_mode关掉，导致日志/诊断缺失）
            if phase == "G2":
                ego_transform_sm = ego_vehicle.get_transform()
                target_transform_sm = target_vehicle.get_transform()
                x_rel_sm, y_rel_sm = get_rel_xy(ego_transform_sm, target_transform_sm)
                dist_sm = math.sqrt(x_rel_sm**2 + y_rel_sm**2)

                # 朝向误差（度）
                yaw_t_deg = target_transform_sm.rotation.yaw
                yaw_e_deg = ego_transform_sm.rotation.yaw
                yaw_err_deg = yaw_t_deg - yaw_e_deg
                while yaw_err_deg > 180:
                    yaw_err_deg -= 360
                while yaw_err_deg < -180:
                    yaw_err_deg += 360
                yaw_err_abs = abs(yaw_err_deg)

                can_switch = (step - last_switch_step) >= min_dwell_steps

                # enter DOCK
                if (not dock_mode) and can_switch:
                    enter_ok = (
                        dist_sm <= d2_enter and
                        x_rel_sm > d2_x_rel_min and
                        abs(y_rel_sm) < d2_y_rel_max and
                        yaw_err_abs < d2_yaw_err_max
                    )
                    if enter_ok:
                        dock_mode = True
                        last_switch_step = step
                        dock_switches += 1
                        # 进入DOCK时重置积分等内部状态
                        try:
                            self.docking_controller.reset()
                        except Exception:
                            pass
                        # 初始化G3收敛监控
                        g3_enter_dist = None
                        g3_best_dist = None
                        g3_steps = 0

                # exit DOCK（滞回）
                if dock_mode and can_switch and dist_sm > d2_exit:
                    dock_mode = False
                    last_switch_step = step
                    dock_switches += 1

            # 本步是否处于Dock几何控制（统一口径）
            dock_active = (phase == "G3") or (phase == "G2" and dock_mode)
            in_dock_phase = dock_active

            # === DOCK 模式 ===
            if dock_active:
                # DOCK阶段：使用几何控制
                # 运行时保护：在G3阶段内部，如果车跑到了目标前方，强制退出G3，回到G2重规划
                ego_transform = ego_vehicle.get_transform()
                target_transform = target_vehicle.get_transform()
                x_rel_runtime, y_rel_runtime = get_rel_xy(ego_transform, target_transform)
                
                # 记录G3进入时的距离（第一次进入时）
                if g3_enter_dist is None:
                    g3_enter_dist = dist_to_target
                    g3_best_dist = dist_to_target
                    print(f"[HierarchicalNav] G3 entered: enter_dist={g3_enter_dist:.2f}m")
                
                g3_steps += 1
                
                # 更新G3阶段最佳距离
                if dist_to_target < g3_best_dist:
                    g3_best_dist = dist_to_target
                
                # 如果在DOCK阶段过程中，车跑到了目标前方（x_rel <= 1.0），就强制退出G3
                if x_rel_runtime <= 1.0:
                    print(f"[HierarchicalNav] G3 runtime protection: x_rel={x_rel_runtime:.2f}m <= 1.0m, exiting G3 and returning to G2")
                    # 优先“软切回TARGET”，避免直接失败（配合dwell+滞回）
                    if phase == "G2":
                        dock_mode = False
                        last_switch_step = step
                        dock_switches += 1
                    else:
                        # phase==G3：保持原逻辑，向上层报告失败，促使重新规划
                        # 关键：在提前return前也记录一条轨迹点，否则下游会误判“从未进入Dock/G3”
                        if self.record_trajectory and step % self.traj_subsample == 0:
                            try:
                                ego_tf_now = ego_vehicle.get_transform()
                                ego_loc_now = ego_tf_now.location
                                target_tf_now = target_vehicle.get_transform()
                                x_rel_now, y_rel_now = get_rel_xy(ego_tf_now, target_tf_now)
                                psi_rel_now = float(ego_tf_now.rotation.yaw - target_tf_now.rotation.yaw)
                                while psi_rel_now > 180:
                                    psi_rel_now -= 360
                                while psi_rel_now < -180:
                                    psi_rel_now += 360
                                v_now = self._get_vehicle_speed(ego_vehicle)
                                trajectory.append({
                                    "step": step,
                                    "x": float(ego_loc_now.x),
                                    "y": float(ego_loc_now.y),
                                    "z": float(ego_loc_now.z),
                                    "yaw": float(ego_tf_now.rotation.yaw),
                                    "speed": v_now,
                                    "dist": dist_to_target,
                                    "phase": "G3",
                                    "x_rel": float(x_rel_now),
                                    "y_rel": float(y_rel_now),
                                    "psi_rel_deg": float(psi_rel_now),
                                    "dock_mode": True,
                                    "dock_submode": "",
                                    "e_y": None,
                                    "e_psi_deg": None,
                                    "ref_yaw_deg": None,
                                })
                            except Exception:
                                pass
                    return {
                        "phase": phase,
                        "success": False,
                        "steps": step + 1,
                        "min_dist": min_dist,
                        "final_dist": dist_to_target,
                        "reason": "g3_exit_to_front",
                            "trajectory": trajectory,
                            "dock_switches": dock_switches,
                    }
                
                # G3失败检测：如果连续100步内best_dist几乎不变，且dist > 8m，判定为不收敛
                if g3_steps > 100 and g3_best_dist > 8.0 and (g3_best_dist > g3_enter_dist - 2.0):
                    print(f"[HierarchicalNav] G3 convergence failure: steps={g3_steps}, best_dist={g3_best_dist:.2f}m, enter_dist={g3_enter_dist:.2f}m, exiting G3 for replan")
                    if phase == "G2":
                        # 在G2内部：软切回TARGET，避免直接失败
                        dock_mode = False
                        last_switch_step = step
                        dock_switches += 1
                        # 重置监控，给下一次DOCK机会
                        g3_enter_dist = None
                        g3_best_dist = None
                        g3_steps = 0
                    else:
                        # phase==G3：保持原逻辑，向上层报告失败
                        # 关键：在提前return前也记录一条轨迹点，否则下游会误判“从未进入Dock/G3”
                        if self.record_trajectory and step % self.traj_subsample == 0:
                            try:
                                ego_tf_now = ego_vehicle.get_transform()
                                ego_loc_now = ego_tf_now.location
                                target_tf_now = target_vehicle.get_transform()
                                x_rel_now, y_rel_now = get_rel_xy(ego_tf_now, target_tf_now)
                                psi_rel_now = float(ego_tf_now.rotation.yaw - target_tf_now.rotation.yaw)
                                while psi_rel_now > 180:
                                    psi_rel_now -= 360
                                while psi_rel_now < -180:
                                    psi_rel_now += 360
                                v_now = self._get_vehicle_speed(ego_vehicle)
                                trajectory.append({
                                    "step": step,
                                    "x": float(ego_loc_now.x),
                                    "y": float(ego_loc_now.y),
                                    "z": float(ego_loc_now.z),
                                    "yaw": float(ego_tf_now.rotation.yaw),
                                    "speed": v_now,
                                    "dist": dist_to_target,
                                    "phase": "G3",
                                    "x_rel": float(x_rel_now),
                                    "y_rel": float(y_rel_now),
                                    "psi_rel_deg": float(psi_rel_now),
                                    "dock_mode": True,
                                    "dock_submode": "",
                                    "e_y": None,
                                    "e_psi_deg": None,
                                    "ref_yaw_deg": None,
                                })
                            except Exception:
                                pass
                    return {
                        "phase": phase,
                        "success": False,
                        "steps": step + 1,
                        "min_dist": min_dist,
                        "final_dist": dist_to_target,
                        "reason": "g3_convergence_failure",
                            "trajectory": trajectory,
                            "dock_switches": dock_switches,
                        }
                
                # === Step 3.2+(A): 对齐诊断量（优先用lane waypoint yaw）===
                # 用与G3目标生成一致的 make_back_goal_from_target 做一次map-snap，
                # 得到更稳定的 ref_point/ref_yaw（lane方向），用于 e_y/e_psi 与 Stanley。
                ref_point = None
                ref_yaw_deg = None
                try:
                    # v4发现：直接map-snap可能在路口/并道附近吸到对向/邻道，造成ref_yaw翻转（|e_psi|极大）。
                    # 这里优先使用“沿目标车所在lane向后走”的方式，强制 lane-consistent。
                    back_dist = float(getattr(self.docking_controller, "target_longitudinal", 3.0))
                    lat_offset = float(getattr(self.docking_controller, "target_lateral", 0.0))

                    target_wp = self.carla_map.get_waypoint(
                        target_transform.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    dock_ref_tf = None
                    if target_wp is not None:
                        cur = target_wp
                        remaining = max(0.0, back_dist)
                        step_size = 1.0
                        while remaining > 1e-6:
                            step = min(step_size, remaining)
                            prev_wps = cur.previous(step)
                            if not prev_wps:
                                break
                            # 在junction附近previous可能分叉：优先选择同road/lane的分支
                            best = None
                            for cand in prev_wps:
                                if cand.road_id == cur.road_id and cand.lane_id == cur.lane_id:
                                    best = cand
                                    break
                            cur = best if best is not None else prev_wps[0]
                            remaining -= step

                        dock_ref_tf = cur.transform
                        # 应用横向偏移（沿lane右侧方向）
                        if abs(lat_offset) > 1e-6:
                            yaw_rad = math.radians(float(dock_ref_tf.rotation.yaw))
                            rx = math.cos(yaw_rad - math.pi / 2.0)
                            ry = math.sin(yaw_rad - math.pi / 2.0)
                            dock_ref_tf = carla.Transform(
                                carla.Location(
                                    x=dock_ref_tf.location.x + lat_offset * rx,
                                    y=dock_ref_tf.location.y + lat_offset * ry,
                                    z=dock_ref_tf.location.z,
                                ),
                                dock_ref_tf.rotation,
                            )

                    # 兜底：lane-consistent失败时再退回原来的map-snap几何方法
                    if dock_ref_tf is None:
                        dock_ref_tf = make_back_goal_from_target(
                            target_transform,
                            self.carla_map,
                            back_dist=back_dist,
                            lat_offset=lat_offset,
                        )

                    ref_point = dock_ref_tf.location
                    ref_yaw_deg = float(dock_ref_tf.rotation.yaw)
                except Exception:
                    # 兜底：DockingController内部会退化到 target yaw + 几何对接点
                    ref_point = None
                    ref_yaw_deg = None

                direct_close_cfg = self.controller_profile_config.get("direct_close_in", {})
                if direct_close_cfg and dist_to_target <= float(direct_close_cfg.get("enter_dist", 10.0)):
                    control, x_rel, y_rel, dock_diag = self._compute_direct_close_in_control(
                        ego_vehicle,
                        target_vehicle,
                        direct_close_cfg,
                    )
                else:
                    control, x_rel, y_rel, dock_diag = self.docking_controller.compute_control(
                        ego_vehicle,
                        target_vehicle,
                        ref_point=ref_point,
                        ref_yaw_deg=ref_yaw_deg,
                    )
                
                throttle = control.throttle
                brake = control.brake
                steer = control.steer
                
                # 检查是否成功
                current_speed = self._get_vehicle_speed(ego_vehicle)
                if self.docking_controller.is_success(x_rel, y_rel, current_speed):
                    # 成功停靠
                    ego_vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    ))
                    self.world.tick()
                    if safety_monitor is not None:
                        safety_monitor.update_ttc(ego_vehicle)
                    
                    # 记录最后一次debug日志
                    if self.debug_log_enabled:
                        self._write_debug_log(
                            step, phase_id, phase, dist_to_target, dist_to_dock_goal,
                            current_speed, in_dock_phase, x_rel, y_rel, 0.0, 1.0, steer
                        )
                    
                    # 关键：把最后一步轨迹也写进去，避免success时“轨迹里没有Dock/G3”的假象
                    if self.record_trajectory and step % self.traj_subsample == 0:
                        try:
                            ego_tf_now = ego_vehicle.get_transform()
                            ego_loc_now = ego_tf_now.location
                            target_tf_now = target_vehicle.get_transform()
                            x_rel_now, y_rel_now = get_rel_xy(ego_tf_now, target_tf_now)
                            psi_rel_now = float(ego_tf_now.rotation.yaw - target_tf_now.rotation.yaw)
                            while psi_rel_now > 180:
                                psi_rel_now -= 360
                            while psi_rel_now < -180:
                                psi_rel_now += 360
                            v_now = self._get_vehicle_speed(ego_vehicle)
                            dock_submode_out = ""
                            e_y_out = None
                            e_psi_out = None
                            ref_yaw_out = None
                            if isinstance(dock_diag, dict):
                                dock_submode_out = str(dock_diag.get("dock_submode", ""))
                                e_y_out = dock_diag.get("e_y", None)
                                e_psi_out = dock_diag.get("e_psi_deg", None)
                                ref_yaw_out = dock_diag.get("ref_yaw_deg", None)
                            trajectory.append({
                                "step": step,
                                "x": float(ego_loc_now.x),
                                "y": float(ego_loc_now.y),
                                "z": float(ego_loc_now.z),
                                "yaw": float(ego_tf_now.rotation.yaw),
                                "speed": v_now,
                                "dist": dist_to_target,
                                "phase": "G3",
                                "x_rel": float(x_rel_now),
                                "y_rel": float(y_rel_now),
                                "psi_rel_deg": float(psi_rel_now),
                                "dock_mode": True,
                                "dock_submode": dock_submode_out,
                                "e_y": float(e_y_out) if e_y_out is not None else None,
                                "e_psi_deg": float(e_psi_out) if e_psi_out is not None else None,
                                "ref_yaw_deg": float(ref_yaw_out) if ref_yaw_out is not None else None,
                            })
                        except Exception:
                            pass
                    
                    return {
                        # 统一标记为G3成功，让上层直接结束episode（即使当前外层phase是G2）
                        "phase": "G3",
                        "success": True,
                        "steps": step + 1,
                        "min_dist": min_dist,
                        "final_dist": dist_to_target,
                        "reason": "docking_success",
                        "dock_switches": dock_switches,
                        "trajectory": trajectory,
                    }
                
                # 应用控制
                carla_control = carla.VehicleControl(
                    throttle=control.throttle,
                    brake=control.brake,
                    steer=control.steer,
                    hand_brake=control.hand_brake
                )
                
            else:
                # G1/G2阶段：使用Pure Pursuit + PID
                # 根据阶段调整前视距离
                if phase == "G1":
                    self.pure_pursuit.lookahead_distance = 10.0  # G1阶段：10-12m（高速阶段）
                elif phase == "G2":
                    self.pure_pursuit.lookahead_distance = 6.0   # G2阶段：6-8m（中速阶段）
                
                # 进度监控：每隔20步检查一次进度
                if step - last_progress_check_step >= progress_window_size:
                    progress_windows.append(min_dist)
                    last_progress_check_step = step
                    
                    # 如果最近3个窗口（60步）内距离没有明显下降（至少2m），触发重规划
                    if len(progress_windows) >= 3:
                        recent_best = min(progress_windows[-3:])
                        prev_best = min(progress_windows[-4:-1]) if len(progress_windows) >= 4 else progress_windows[0]
                        
                        if recent_best > prev_best - 2.0:
                            print(f"[HierarchicalNav] {phase}: Poor progress detected (best_dist={recent_best:.2f}m, prev_best={prev_best:.2f}m), triggering replan")
                            # 重新规划到G3
                            goals = self.goal_manager.generate_goals(target_vehicle)
                            g3_goal = None
                            for g in goals:
                                if g.goal_type == "G3":
                                    g3_goal = g
                                    break
                            
                            if g3_goal:
                                ego_wp = self.route_planner.get_waypoint_at_location(ego_loc)
                                g3_wp = self.goal_manager.get_goal_waypoint(g3_goal)
                                if ego_wp and g3_wp:
                                    self.current_route = self.route_planner.plan_route_from_waypoint(
                                        ego_wp, g3_wp
                                    )
                                    goal = g3_goal
                                    print(f"[HierarchicalNav] {phase}: Replanned route to G3, route_length={len(self.current_route)}")
                                    # 重置进度窗口
                                    progress_windows = []
                
                # G2阶段：强制绕到后面的补救逻辑
                if phase == "G2":
                    ego_transform = ego_vehicle.get_transform()
                    target_transform = target_vehicle.get_transform()
                    x_rel_g2, y_rel_g2 = get_rel_xy(ego_transform, target_transform)
                    dist_g2 = math.sqrt(x_rel_g2**2 + y_rel_g2**2)
                    
                    # 计算朝向误差
                    yaw_t_deg = target_transform.rotation.yaw
                    yaw_e_deg = ego_transform.rotation.yaw
                    yaw_err_deg = yaw_t_deg - yaw_e_deg
                    while yaw_err_deg > 180:
                        yaw_err_deg -= 360
                    while yaw_err_deg < -180:
                        yaw_err_deg += 360
                    yaw_err_abs = abs(yaw_err_deg)
                    
                    # 如果离目标很近了，但还在前半平面 → 强制重规划到G3
                    if dist_g2 < 15.0 and x_rel_g2 <= 0.5:
                        print(f"[HierarchicalNav] G2: dist={dist_g2:.2f}m < 15m and x_rel={x_rel_g2:.2f}m <= 0.5m, forcing replan to G3")
                        # 强制把当前阶段目标改为G3（后方4m）
                        # 重新规划route: ego当前waypoint -> G3 waypoint
                        goals = self.goal_manager.generate_goals(target_vehicle)
                        g3_goal = None
                        for g in goals:
                            if g.goal_type == "G3":
                                g3_goal = g
                                break
                        
                        if g3_goal:
                            ego_wp = self.route_planner.get_waypoint_at_location(ego_loc)
                            g3_wp = self.goal_manager.get_goal_waypoint(g3_goal)
                            if ego_wp and g3_wp:
                                self.current_route = self.route_planner.plan_route_from_waypoint(
                                    ego_wp, g3_wp
                                )
                                # 更新goal为G3，但保持当前phase为G2（让循环继续，但路径已更新）
                                goal = g3_goal
                                print(f"[HierarchicalNav] G2: Replanned route to G3, route_length={len(self.current_route)}")
                
                if not self.current_route:
                    # 如果没有路径，停止
                    carla_control = carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    )
                    throttle = 0.0
                    brake = 1.0
                    steer = 0.0
                else:
                    # Pure Pursuit转向控制
                    steer = self.pure_pursuit.compute_steer(
                        ego_vehicle.get_transform(),
                        self.current_route
                    )
                    
                    # PID速度控制
                    current_speed = self._get_vehicle_speed(ego_vehicle)
                    throttle, brake = self.speed_pid.compute_control(
                        goal.target_speed,
                        current_speed
                    )
                    
                    carla_control = carla.VehicleControl(
                        throttle=throttle,
                        brake=brake,
                        steer=steer
                    )
            
            # 记录debug日志
            if self.debug_log_enabled:
                current_speed = self._get_vehicle_speed(ego_vehicle)
                self._write_debug_log(
                    step, phase_id, phase, dist_to_target, dist_to_dock_goal,
                    current_speed, dock_active, x_rel, y_rel, throttle, brake, steer
                )
            
            # 应用控制
            ego_vehicle.apply_control(carla_control)
            self.world.tick()
            if safety_monitor is not None:
                safety_monitor.update_ttc(ego_vehicle)
            
            # 检查是否到达目标
            if phase != "G3":
                # 对于G1/G2，检查是否接近目标waypoint
                goal_dist = self._distance_between_locations(
                    ego_loc, goal.transform.location
                )
                goal_reached_radius = float(self.controller_profile_config.get("goal_reached_radius", 3.0))
                if goal_dist < goal_reached_radius:  # 到达目标附近
                    return {
                        "phase": phase,
                        "success": True,
                        "steps": step + 1,
                        "min_dist": min_dist,
                        "final_dist": dist_to_target,
                        "reason": "goal_reached"
                    }
            
            # 记录轨迹
            if self.record_trajectory and step % self.traj_subsample == 0:
                # 统一：trajectory里用 dock_active 代表“本步是否执行Dock几何控制”
                dock_mode_flag = bool(dock_active)

                dock_submode = ""
                e_y = None
                e_psi_deg = None
                ref_yaw_deg_out = None
                if isinstance(dock_diag, dict):
                    dock_submode = str(dock_diag.get("dock_submode", ""))
                    e_y = dock_diag.get("e_y", None)
                    e_psi_deg = dock_diag.get("e_psi_deg", None)
                    ref_yaw_deg_out = dock_diag.get("ref_yaw_deg", None)

                # 统一：如果本步处于Dock几何控制，phase字段也标成"G3"
                traj_phase = "G3" if dock_active else phase

                trajectory.append({
                    "step": step,
                    "x": float(ego_loc.x),
                    "y": float(ego_loc.y),
                    "z": float(ego_loc.z),
                    "yaw": float(ego_vehicle.get_transform().rotation.yaw),
                    "speed": current_speed,
                    "dist": dist_to_target,
                    "phase": traj_phase,
                    # --- Dock-entry 结构性分析（目标车坐标系）---
                    "x_rel": float(x_rel_to_target),
                    "y_rel": float(y_rel_to_target),
                    "psi_rel_deg": float(psi_rel_deg),
                    # --- Dock 对齐诊断（Step 3.2+）---
                    "dock_mode": dock_mode_flag,
                    "dock_submode": dock_submode,
                    "e_y": float(e_y) if e_y is not None else None,
                    "e_psi_deg": float(e_psi_deg) if e_psi_deg is not None else None,
                    "ref_yaw_deg": float(ref_yaw_deg_out) if ref_yaw_deg_out is not None else None,
                })
            
            steps += 1
        
        # 阶段超时
        return {
            "phase": phase,
            "success": False,
            "steps": steps,
            "min_dist": min_dist,
            "final_dist": dist_to_target,
            "reason": "timeout",
            "trajectory": trajectory
        }

    def _dock_entry_guard_ok(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        x_guard: Optional[float] = None,
        psi_guard_deg: Optional[float] = None,
        y_guard: Optional[float] = None,
    ) -> bool:
        """
        Dock-Entry Guard：只允许在“目标车后半平面 + 航向大致一致 (+可选横向不离谱)”时进入真正Dock(close-range)。

        注意：get_rel_xy 的 x 轴指向目标车后方，因此：
        - x_rel > 0 代表 ego 在目标车后方
        - x_rel < 0 代表 ego 在目标车前方
        """
        ego_tf = ego_vehicle.get_transform()
        tgt_tf = target_vehicle.get_transform()
        x_rel, y_rel = get_rel_xy(ego_tf, tgt_tf)
        guard_cfg = self.controller_profile_config.get("dock_entry_guard", {})
        if x_guard is None:
            x_guard = float(guard_cfg.get("x_guard", 1.5))
        if psi_guard_deg is None:
            psi_guard_deg = float(guard_cfg.get("psi_guard_deg", 15.0))
        if y_guard is None:
            y_guard = float(guard_cfg.get("y_guard", 1.0))

        psi_rel = float(ego_tf.rotation.yaw - tgt_tf.rotation.yaw)
        while psi_rel > 180:
            psi_rel -= 360
        while psi_rel < -180:
            psi_rel += 360

        # 必须在后方一定余量（避免“刚擦到侧前方就进Dock”）
        if x_rel <= x_guard:
            return False
        # 航向大致一致
        if abs(psi_rel) > psi_guard_deg:
            return False
        # 横向不离谱（不要设太紧，否则引入timeout）
        if abs(y_rel) > y_guard:
            return False
        return True

    def _make_orbit_goal(
        self,
        target_vehicle: carla.Vehicle,
        back_dist: float = 10.0,
        lat_offset: float = 0.0,
        target_speed: float = 4.0,
        max_steps: int = 200,
    ) -> SemanticGoal:
        """
        在G2阶段使用的“绕后”目标：追踪目标车后方更远的虚拟点（8–12m），让车辆先绕到后半平面。
        目标点优先 lane-consistent（沿目标车所在车道 previous() 向后走），失败再退回 map-snap。
        """
        target_transform = target_vehicle.get_transform()
        try:
            target_wp = self.carla_map.get_waypoint(
                target_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            target_wp = None

        orbit_tf = None
        try:
            if target_wp is not None:
                cur = target_wp
                remaining = max(0.0, float(back_dist))
                step_size = 1.0
                while remaining > 1e-6:
                    step = min(step_size, remaining)
                    prev_wps = cur.previous(step)
                    if not prev_wps:
                        break
                    best = None
                    for cand in prev_wps:
                        if cand.road_id == cur.road_id and cand.lane_id == cur.lane_id:
                            best = cand
                            break
                    cur = best if best is not None else prev_wps[0]
                    remaining -= step

                orbit_tf = cur.transform

                # 可选横向偏移（沿lane右侧方向）
                if abs(lat_offset) > 1e-6:
                    yaw_rad = math.radians(float(orbit_tf.rotation.yaw))
                    rx = math.cos(yaw_rad - math.pi / 2.0)
                    ry = math.sin(yaw_rad - math.pi / 2.0)
                    orbit_tf = carla.Transform(
                        carla.Location(
                            x=orbit_tf.location.x + lat_offset * rx,
                            y=orbit_tf.location.y + lat_offset * ry,
                            z=orbit_tf.location.z,
                        ),
                        orbit_tf.rotation,
                    )
        except Exception:
            orbit_tf = None

        if orbit_tf is None:
            orbit_tf = make_back_goal_from_target(
                target_transform,
                self.carla_map,
                back_dist=float(back_dist),
                lat_offset=float(lat_offset),
            )

        return SemanticGoal(
            transform=orbit_tf,
            goal_type="G2",
            target_speed=float(target_speed),
            max_steps=int(max_steps),
        )
    
    def enable_debug_log(self, log_file: str):
        """启用debug日志"""
        self.debug_log_enabled = True
        self.debug_log_file = log_file
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 打开文件并创建CSV writer
        self.debug_log_fp = open(log_file, 'w', newline='', encoding='utf-8')
        self.debug_log_writer = csv.DictWriter(
            self.debug_log_fp,
            fieldnames=['step', 'phase_id', 'phase_name', 'dist_to_target', 'dist_to_dock_goal', 
                       'speed', 'in_dock_phase', 'x_rel', 'y_rel', 'throttle', 'brake', 'steer']
        )
        self.debug_log_writer.writeheader()
        print(f"[Debug] Enabled debug logging to {log_file}")
    
    def _write_debug_log(
        self,
        step: int,
        phase_id: int,
        phase_name: str,
        dist_to_target: float,
        dist_to_dock_goal: float,
        speed: float,
        in_dock_phase: bool,
        x_rel: Optional[float] = None,
        y_rel: Optional[float] = None,
        throttle: Optional[float] = None,
        brake: Optional[float] = None,
        steer: Optional[float] = None,
    ):
        """写入debug日志"""
        if not self.debug_log_enabled or self.debug_log_writer is None:
            return
        
        self.debug_log_writer.writerow({
            'step': step,
            'phase_id': phase_id,
            'phase_name': phase_name,
            'dist_to_target': f'{dist_to_target:.3f}',
            'dist_to_dock_goal': f'{dist_to_dock_goal:.3f}',
            'speed': f'{speed:.3f}',
            'in_dock_phase': '1' if in_dock_phase else '0',
            'x_rel': f'{x_rel:.3f}' if x_rel is not None else '',
            'y_rel': f'{y_rel:.3f}' if y_rel is not None else '',
            'throttle': f'{throttle:.3f}' if throttle is not None else '',
            'brake': f'{brake:.3f}' if brake is not None else '',
            'steer': f'{steer:.3f}' if steer is not None else '',
        })
        self.debug_log_fp.flush()
    
    def disable_debug_log(self):
        """禁用debug日志并关闭文件"""
        if self.debug_log_fp:
            self.debug_log_fp.close()
            self.debug_log_fp = None
            self.debug_log_writer = None
        self.debug_log_enabled = False
    
    def run_episode(
        self,
        ego_vehicle: carla.Vehicle,
        target_vehicle: carla.Vehicle,
        max_steps_total: int = 1000
    ) -> NavigationResult:
        """
        运行完整episode
        
        Args:
            ego_vehicle: ego车辆
            target_vehicle: 目标车辆
            max_steps_total: 总最大步数
        
        Returns:
            导航结果
        """
        # 确保目标车辆静止
        target_vehicle.set_autopilot(False)
        target_vehicle.apply_control(carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=1.0,
            hand_brake=False, reverse=False, manual_gear_shift=False
        ))
        
        # 计算初始距离
        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        start_dist = self._distance_between_locations(ego_loc, target_loc)
        min_dist = start_dist
        final_dist = start_dist

        if self.controller_profile_config.get("run_mode") == "direct_acquisition":
            return self._run_direct_acquisition_episode(
                ego_vehicle=ego_vehicle,
                target_vehicle=target_vehicle,
                start_dist=start_dist,
                max_steps_total=max_steps_total,
            )
        if self.controller_profile_config.get("run_mode") == "behavior_acquisition":
            return self._run_behavior_acquisition_episode(
                ego_vehicle=ego_vehicle,
                target_vehicle=target_vehicle,
                start_dist=start_dist,
                max_steps_total=max_steps_total,
            )
        if self.controller_profile_config.get("run_mode") == "multi_approach_acquisition":
            return self._run_multi_approach_acquisition_episode(
                ego_vehicle=ego_vehicle,
                target_vehicle=target_vehicle,
                start_dist=start_dist,
                max_steps_total=max_steps_total,
            )
        
        # 生成语义子目标
        goals = self.goal_manager.generate_goals(target_vehicle)
        goals = self._maybe_use_ego_side_entry_goals(ego_vehicle, target_vehicle, goals)
        
        print(f"[HierarchicalNav] Generated {len(goals)} semantic goals")
        for i, goal in enumerate(goals):
            print(f"  {goal.goal_type}: distance={self._distance_between_locations(target_loc, goal.transform.location):.2f}m, "
                  f"target_speed={goal.target_speed:.1f}m/s, max_steps={goal.max_steps}")
        
        # 启用debug日志（如果目录存在）
        debug_log_path = f"logs_v2/online_multi/debug/control_debug_ep{int(time.time())}.csv"
        try:
            self.enable_debug_log(debug_log_path)
        except Exception as e:
            print(f"[Warning] Failed to enable debug log: {e}")
        
        # 运行三个阶段
        total_steps = 0
        success_4m = False
        success_8m = False
        stuck = False
        all_trajectory = []
        phase_results = {}
        safety_monitor = SafetyEventMonitor(self.world, ego_vehicle, enabled=self.record_safety)
        
        # DOCK触发距离阈值（放宽到12m）
        DOCK_TRIGGER_DISTANCE = 12.0
        
        for goal_idx, goal in enumerate(goals):
            if total_steps >= max_steps_total:
                stuck = True
                break

            # === Dock-Entry Guard：真正进入G3(close-range Dock)之前必须先到后半平面并对齐 ===
            # 如果当前 goal 是 G3 但不满足守卫条件，则先在 G2 执行“绕后”目标（追更远的后方点）
            if goal.goal_type == "G3":
                # 给绕后一个有限预算，避免死循环
                orbit_cfg = self.controller_profile_config.get("dock_orbit", {})
                max_orbit_attempts = int(orbit_cfg.get("attempts", 3))
                orbit_attempts = 0
                while (
                    (not self._dock_entry_guard_ok(ego_vehicle, target_vehicle))
                    and (total_steps < max_steps_total)
                    and orbit_attempts < max_orbit_attempts
                ):
                    orbit_attempts += 1
                    remaining_steps = max_steps_total - total_steps
                    # 绕后点设置更远一些（默认10m），速度适中，避免贴太近导致碰撞/抖动
                    orbit_goal = self._make_orbit_goal(
                        target_vehicle=target_vehicle,
                        back_dist=float(orbit_cfg.get("back_dist", 10.0)),
                        lat_offset=float(orbit_cfg.get("lat_offset", 0.0)),
                        target_speed=float(orbit_cfg.get("target_speed", 4.0)),
                        max_steps=int(min(int(orbit_cfg.get("max_steps", 200)), remaining_steps)),
                    )
                    print(f"[HierarchicalNav] Dock-Entry Guard blocked G3, running G2-orbit (attempt={orbit_attempts})")
                    orbit_res = self._run_phase(
                        ego_vehicle,
                        target_vehicle,
                        orbit_goal,
                        int(min(int(orbit_goal.max_steps), remaining_steps)),
                        min_dist,
                        safety_monitor=safety_monitor,
                    )
                    phase_results[f"G2_orbit_{orbit_attempts}"] = orbit_res
                    total_steps += orbit_res.get("steps", 0)
                    min_dist = min(min_dist, orbit_res.get("min_dist", min_dist))
                    final_dist = orbit_res.get("final_dist", final_dist)
                    if "trajectory" in orbit_res:
                        all_trajectory.extend(orbit_res["trajectory"])
                    # 如果orbit本身就彻底失败且不可恢复，标记stuck
                    if (not orbit_res.get("success", False)) and (not self.recovery_policy.can_recover("G2")):
                        stuck = True
                        break

                if stuck or total_steps >= max_steps_total:
                    break
            
            # 检查是否应该提前切换到G3阶段
            # DOCK触发条件：必须在目标车后方且距离足够近
            ego_transform = ego_vehicle.get_transform()
            target_transform = target_vehicle.get_transform()
            ego_loc = ego_transform.location
            target_loc = target_transform.location
            current_dist = self._distance_between_locations(ego_loc, target_loc)
            
            # 使用统一的get_rel_xy计算相对位置
            x_rel_check, y_rel_check = get_rel_xy(ego_transform, target_transform)
            dist_check = math.sqrt(x_rel_check**2 + y_rel_check**2)
            
            # 计算朝向误差（用于进入G3前的最后确认）
            yaw_t_deg = target_transform.rotation.yaw
            yaw_e_deg = ego_transform.rotation.yaw
            yaw_err_deg = yaw_t_deg - yaw_e_deg
            while yaw_err_deg > 180:
                yaw_err_deg -= 360
            while yaw_err_deg < -180:
                yaw_err_deg += 360
            yaw_err_abs = abs(yaw_err_deg)
            
            # 强化DOCK触发条件：
            # 1. 距离在合理范围内（6m-15m，避免太近或太远）
            # 2. 明确在"后半平面"（x_rel > 2.0，留margin防止在目标正上方）
            # 3. 横向偏差不要离谱（|y_rel| < 5.0）
            # 4. 朝向误差不要太大（< 45度，新增）
            dock_trigger_cfg = self.controller_profile_config.get("dock_trigger", {})
            DOCK_TRIGGER_DIST_MIN = float(dock_trigger_cfg.get("dist_min", 6.0))
            DOCK_TRIGGER_DIST_MAX = float(dock_trigger_cfg.get("dist_max", 15.0))
            DOCK_TRIGGER_X_REL = float(dock_trigger_cfg.get("x_rel", 2.0))
            DOCK_TRIGGER_Y_REL_MAX = float(dock_trigger_cfg.get("y_rel_max", 5.0))
            DOCK_TRIGGER_YAW_ERR_MAX = float(dock_trigger_cfg.get("yaw_err_max", 45.0))
            
            enter_dock = (
                dist_check >= DOCK_TRIGGER_DIST_MIN and
                dist_check <= DOCK_TRIGGER_DIST_MAX and
                x_rel_check > DOCK_TRIGGER_X_REL and
                abs(y_rel_check) < DOCK_TRIGGER_Y_REL_MAX and
                yaw_err_abs < DOCK_TRIGGER_YAW_ERR_MAX
            )
            
            # 如果当前在G1或G2阶段，且满足DOCK触发条件，提前切换到G3
            # 额外加入 Dock-Entry Guard：必须“在后方+航向对齐(+横向合理)”才允许进入真正Dock(close)
            if goal.goal_type in ["G1", "G2"] and enter_dock and self._dock_entry_guard_ok(ego_vehicle, target_vehicle):
                print(f"[HierarchicalNav] Early transition to G3: dist={dist_check:.2f}m, x_rel={x_rel_check:.2f}m, y_rel={y_rel_check:.2f}m, yaw_err={yaw_err_abs:.1f}deg")
                # 跳过当前阶段，直接进入G3
                # 找到G3目标
                g3_goal = None
                for g in goals:
                    if g.goal_type == "G3":
                        g3_goal = g
                        break
                
                if g3_goal:
                    # 直接运行G3阶段
                    remaining_steps = max_steps_total - total_steps
                    phase_max_steps = int(min(int(g3_goal.max_steps), remaining_steps))
                    print(f"[HierarchicalNav] Starting phase G3 (early transition, max_steps={phase_max_steps})")
                    
                    phase_result = self._run_phase(
                        ego_vehicle,
                        target_vehicle,
                        g3_goal,
                        phase_max_steps,
                        min_dist,
                        safety_monitor=safety_monitor,
                    )
                    
                    phase_results["G3"] = phase_result
                    total_steps += phase_result["steps"]
                    min_dist = min(min_dist, phase_result["min_dist"])
                    final_dist = phase_result["final_dist"]
                    
                    if "trajectory" in phase_result:
                        all_trajectory.extend(phase_result["trajectory"])
                    
                    if final_dist < self.success_radius_4m:
                        success_4m = True
                        success_8m = True
                    
                    if final_dist < self.success_radius_8m:
                        success_8m = True
                    
                    if phase_result.get("phase") == "G3" and phase_result.get("success"):
                        print(f"[HierarchicalNav] Phase G3 succeeded, episode complete")
                        break
                    
                    if not phase_result["success"] and not self.recovery_policy.can_recover("G3"):
                        stuck = True
                        print(f"[HierarchicalNav] Phase G3 failed and cannot recover")
                    
                    break  # 跳出循环，不再执行后续阶段
            
            # 计算该阶段的最大步数
            remaining_steps = max_steps_total - total_steps
            phase_max_steps = int(min(int(goal.max_steps), remaining_steps))
            
            print(f"[HierarchicalNav] Starting phase {goal.goal_type} (max_steps={phase_max_steps})")
            
            # 运行阶段
            phase_result = self._run_phase(
                ego_vehicle,
                target_vehicle,
                goal,
                phase_max_steps,
                min_dist,
                safety_monitor=safety_monitor,
            )
            
            phase_results[goal.goal_type] = phase_result
            total_steps += phase_result["steps"]
            
            # 更新最小距离
            min_dist = min(min_dist, phase_result["min_dist"])
            final_dist = phase_result["final_dist"]
            
            # 检查成功条件
            if final_dist < self.success_radius_4m:
                success_4m = True
                success_8m = True
            
            if final_dist < self.success_radius_8m:
                success_8m = True

            stop_goal_types = set(self.controller_profile_config.get("stop_after_goal_types", []))
            stop_radius = self.controller_profile_config.get("stop_after_success_radius")
            if (
                stop_radius is not None and
                goal.goal_type in stop_goal_types and
                phase_result.get("success", False) and
                final_dist < float(stop_radius)
            ):
                print(
                    f"[HierarchicalNav] Profile early stop after {goal.goal_type}: "
                    f"final_dist={final_dist:.2f}m < {float(stop_radius):.2f}m"
                )
                break
            
            # 如果Dock成功，提前结束（允许在G2内部进入Dock成功后结束）
            if phase_result.get("phase") == "G3" and phase_result.get("success"):
                print(f"[HierarchicalNav] Phase G3 succeeded, episode complete")
                break
            
            # 如果阶段失败且无法恢复，可能卡死
            if not phase_result["success"] and not self.recovery_policy.can_recover(goal.goal_type):
                stuck = True
                print(f"[HierarchicalNav] Phase {goal.goal_type} failed and cannot recover")
            
            # 合并轨迹
            if "trajectory" in phase_result:
                all_trajectory.extend(phase_result["trajectory"])
        
        # 最终距离
        ego_loc = ego_vehicle.get_transform().location
        target_loc = target_vehicle.get_transform().location
        final_dist = self._distance_between_locations(ego_loc, target_loc)
        if self.controller_profile_config.get("use_min_distance_success_flags", False):
            if min_dist < self.success_radius_4m:
                success_4m = True
                success_8m = True
            elif min_dist < self.success_radius_8m:
                success_8m = True
        
        # 关闭debug日志
        tail_cfg = self.controller_profile_config.get("post_phase_direct_tail")
        if tail_cfg:
            only_if_not_4m = bool(tail_cfg.get("only_if_not_4m", True))
            should_run_tail = (not only_if_not_4m) or (not success_4m)
            remaining_steps = max(0, int(max_steps_total) - int(total_steps))
            tail_steps = min(int(tail_cfg.get("max_steps", 0)), remaining_steps)
            if should_run_tail and tail_steps > 0:
                print(
                    f"[HierarchicalNav] Running post-phase direct tail: "
                    f"steps={tail_steps}, min_dist={min_dist:.2f}m"
                )
                tail_result = self._run_direct_recovery_tail(
                    ego_vehicle=ego_vehicle,
                    target_vehicle=target_vehicle,
                    cfg=tail_cfg,
                    max_steps=tail_steps,
                    min_dist_in=min_dist,
                    safety_monitor=safety_monitor,
                )
                phase_results["DIRECT_TAIL"] = tail_result
                total_steps += tail_result.get("steps", 0)
                min_dist = min(min_dist, tail_result.get("min_dist", min_dist))
                final_dist = tail_result.get("final_dist", final_dist)
                if "trajectory" in tail_result:
                    all_trajectory.extend(tail_result["trajectory"])
                if min_dist < self.success_radius_4m:
                    success_4m = True
                    success_8m = True
                elif min_dist < self.success_radius_8m:
                    success_8m = True

        behavior_cfg = self.controller_profile_config.get("post_failure_behavior")
        if behavior_cfg:
            only_if_not_8m = bool(behavior_cfg.get("only_if_not_8m", True))
            should_run_behavior = (not only_if_not_8m) or (not success_8m)
            remaining_steps = max(0, int(max_steps_total) - int(total_steps))
            behavior_steps = min(int(behavior_cfg.get("max_steps", remaining_steps)), remaining_steps)
            if should_run_behavior and behavior_steps > 0:
                print(
                    f"[HierarchicalNav] Running post-failure behavior recovery: "
                    f"steps={behavior_steps}, min_dist={min_dist:.2f}m"
                )
                behavior_result = self._run_behavior_recovery_sequence(
                    ego_vehicle=ego_vehicle,
                    target_vehicle=target_vehicle,
                    cfg=behavior_cfg,
                    max_steps=behavior_steps,
                    min_dist_in=min_dist,
                    safety_monitor=safety_monitor,
                )
                phase_results["POST_FAIL_BEHAVIOR"] = behavior_result
                total_steps += int(behavior_result.get("steps", 0))
                min_dist = min(min_dist, float(behavior_result.get("min_dist", min_dist)))
                final_dist = float(behavior_result.get("final_dist", final_dist))
                if "trajectory" in behavior_result:
                    all_trajectory.extend(behavior_result["trajectory"])
                if min_dist < self.success_radius_4m:
                    success_4m = True
                    success_8m = True
                elif min_dist < self.success_radius_8m:
                    success_8m = True

        post_cfg = self.controller_profile_config.get("post_failure_approaches")
        if post_cfg:
            only_if_not_4m = bool(post_cfg.get("only_if_not_4m", True))
            should_run_post = (not only_if_not_4m) or (not success_4m)
            remaining_steps = max(0, int(max_steps_total) - int(total_steps))
            if should_run_post and remaining_steps > 0:
                print(
                    f"[HierarchicalNav] Running post-failure approaches: "
                    f"steps={remaining_steps}, min_dist={min_dist:.2f}m"
                )
                post_result = self._run_multi_approach_sequence(
                    ego_vehicle=ego_vehicle,
                    target_vehicle=target_vehicle,
                    cfg=post_cfg,
                    max_steps=remaining_steps,
                    min_dist_in=min_dist,
                    safety_monitor=safety_monitor,
                    key_prefix="POST_FAIL",
                )
                phase_results.update(post_result.get("phase_results", {}))
                total_steps += int(post_result.get("steps", 0))
                min_dist = min(min_dist, float(post_result.get("min_dist", min_dist)))
                final_dist = float(post_result.get("final_dist", final_dist))
                if "trajectory" in post_result:
                    all_trajectory.extend(post_result["trajectory"])
                if min_dist < self.success_radius_4m:
                    success_4m = True
                    success_8m = True
                elif min_dist < self.success_radius_8m:
                    success_8m = True

        self.disable_debug_log()
        safety_summary = safety_monitor.summary()
        safety_monitor.close()
        
        return NavigationResult(
            start_dist=start_dist,
            min_dist=min_dist,
            final_dist=final_dist,
            steps=total_steps,
            success_4m=success_4m,
            success_8m=success_8m,
            stuck=stuck,
            trajectory=all_trajectory if self.record_trajectory else [],
            phase_results=phase_results,
            safety=safety_summary,
        )
