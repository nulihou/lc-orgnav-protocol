"""
在线导航模块

提供在线导航评估所需的核心功能。
"""

try:
    from .controller import (
        setup_world,
        run_behavior_nav,
        distance_between_locations,
        _import_behavior_agent,
        _find_carla_python_api,
    )
except ModuleNotFoundError as exc:
    if exc.name != "carla":
        raise
    setup_world = None
    run_behavior_nav = None
    distance_between_locations = None
    _import_behavior_agent = None
    _find_carla_python_api = None

try:
    from .graph_builder import build_graph_from_scene
except ModuleNotFoundError as exc:
    if exc.name != "carla":
        raise
    build_graph_from_scene = None

from .target_select import (
    select_target_oracle,
    select_target_gs,
    select_target_random,
)

from .gs_loader import load_gs_model

try:
    from .target_route_planner import (
        generate_approach_waypoint,
        generate_approach_location,
        generate_route_to_target,
        get_target_vehicle_location,
    )
except ModuleNotFoundError as exc:
    if exc.name != "carla":
        raise
    generate_approach_waypoint = None
    generate_approach_location = None
    generate_route_to_target = None
    get_target_vehicle_location = None

__all__ = [
    # Controller
    "setup_world",
    "run_behavior_nav",
    "distance_between_locations",
    "_import_behavior_agent",
    "_find_carla_python_api",
    # Graph builder
    "build_graph_from_scene",
    # Target selection
    "select_target_oracle",
    "select_target_gs",
    "select_target_random",
    # Model loader
    "load_gs_model",
    # Target route planner
    "generate_approach_waypoint",
    "generate_approach_location",
    "generate_route_to_target",
    "get_target_vehicle_location",
]
