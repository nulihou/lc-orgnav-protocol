#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协议驱动的Random模式在线评估

关键区别：
- 不再在CARLA里临时构图决定候选
- 候选集合、target_idx_gt全部来自protocol_graphs/scen_xxx.npz
- 在线只做：从协议图随机选择目标车 + 用G1/G2/G3控制器开车
"""

import os
import sys
import json
import argparse
import numpy as np
import random
import torch
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import io

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
_script_dir = os.path.dirname(os.path.abspath(__file__))
_online_multi_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_online_multi_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# CARLA Python API path
_carla_python_api_paths = [
    os.environ.get("CARLA_PYTHON_API_PATH"),
    "E:/carla/WindowsNoEditor/PythonAPI",
    "/opt/carla/PythonAPI",
    os.path.join(os.path.expanduser("~"), "CARLA_0.9.15", "PythonAPI"),
]

for path in _carla_python_api_paths:
    if path and os.path.exists(path):
        if path not in sys.path:
            sys.path.insert(0, path)
        break

try:
    import carla
except ImportError as e:
    print(f"[ERROR] Failed to import carla: {e}")
    sys.exit(1)

from scripts.online_multi.online_nav_eval_multi_base import (
    setup_world,
)
from scripts.online.oracle_behavior_demo import setup_ego_and_target_vehicles
from scripts.online_multi.proto_driven_utils import (
    graph_index_to_actor,
    setup_world_for_protocol_scenario,
)
from scripts.online_multi.safety_fields import ensure_safety_fields
from lc_org_nav.online_nav.hierarchical_controller import HierarchicalNavController

MODE = "random_proto_driven"

def load_seed_items(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 支持两种格式：
    # 1) 直接 list[dict]
    # 2) { "configs": [...] }（当前工程输出）
    if isinstance(data, dict) and "configs" in data:
        items = data["configs"]
    else:
        items = data
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(f"Invalid seed config format: {path}")
    return items


def normalize_seed_item(seed_item: Dict[str, Any], fallback_seed_id: int) -> Dict[str, Any]:
    """把各种字段名归一到 python_seed/numpy_seed/torch_seed 等。"""
    out = dict(seed_item)
    out.setdefault("seed_id", out.get("seed_id", fallback_seed_id))
    # 兼容旧字段名
    if "python_seed" not in out and "python_random_seed" in out:
        out["python_seed"] = out["python_random_seed"]
    if "numpy_seed" not in out and "np_seed" in out:
        out["numpy_seed"] = out["np_seed"]
    if "torch_seed" not in out and "pytorch_seed" in out:
        out["torch_seed"] = out["pytorch_seed"]
    return out


def apply_all_seeds(seed_item: Dict[str, Any]) -> None:
    py = int(seed_item.get("python_seed", 0))
    np_seed = int(seed_item.get("numpy_seed", 0))
    th = int(seed_item.get("torch_seed", 0))
    random.seed(py)
    np.random.seed(np_seed)
    try:
        torch.manual_seed(th)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(th)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def setup_world_determinism(
    client: "carla.Client",
    world: "carla.World",
    tm_port: int,
    tm_seed: int,
    carla_seed: int,
    fixed_delta_seconds: float = 0.05,
    no_rendering: bool = False,
):
    """同步模式 + fixed_delta + TM同步 + TM seed，每轮调用一次。"""
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(fixed_delta_seconds)
    if no_rendering:
        settings.no_rendering_mode = True
    world.apply_settings(settings)

    try:
        world.set_pedestrians_seed(int(carla_seed))
    except Exception:
        pass

    tm = client.get_trafficmanager(int(tm_port))
    tm.set_synchronous_mode(True)
    try:
        tm.set_random_device_seed(int(tm_seed))
    except Exception:
        pass

    world.tick()
    return tm


def destroy_actors_safe(ego_vehicle, target_vehicle, npc_vehicles, world):
    try:
        if ego_vehicle is not None and getattr(ego_vehicle, "is_alive", False):
            ego_vehicle.destroy()
    except Exception:
        pass
    try:
        if target_vehicle is not None and getattr(target_vehicle, "is_alive", False):
            target_vehicle.destroy()
    except Exception:
        pass
    for npc in npc_vehicles or []:
        try:
            if getattr(npc, "is_alive", False):
                npc.destroy()
        except Exception:
            pass
    try:
        world.tick()
    except Exception:
        pass


def run_evaluation(
    scenarios: List[Dict[str, Any]],
    world: carla.World,
    ego_vehicle: carla.Vehicle,
    target_vehicle: carla.Vehicle,
    npc_vehicles: List[carla.Vehicle],
    max_steps: int,
    output_path: str,
    protocol_dir: str,
    max_episodes: Optional[int] = None,
    episode_id: Optional[int] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    controller_profile: str = "paper_fixed",
    planner_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    运行协议驱动的Random模式评估
    
    关键：候选集合、target_idx_gt全部来自protocol_graphs/scen_xxx.npz
    """
    protocol_dir = Path(protocol_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _emit(result: Dict[str, Any]) -> None:
        """每个episode写一行JSONL（成功/失败都写）。"""
        ensure_safety_fields(result)
        with open(output_path, 'a', encoding='utf-8') as f:
            json_str = json.dumps(result, ensure_ascii=False, separators=(',', ':'), indent=None)
            f.write(json_str + '\n')
            f.flush()

    if max_episodes is not None and max_episodes > 0:
        scenarios = scenarios[:max_episodes]
        print(f"[Config] Limited to first {max_episodes} episodes")
    
    results = []
    failed_scenarios = []
    
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    
    # 初始化分层导航控制器
    nav_controller = HierarchicalNavController(
        world=world,
        carla_map=carla_map,
        success_radius_4m=4.0,
        success_radius_8m=8.0,
        record_trajectory=True,
        traj_subsample=2,
        controller_profile=controller_profile,
        planner_mode=planner_mode,
    )
    
    print(f"\n[Evaluation] Running {len(scenarios)} episodes in {MODE.upper()} mode...")
    print(f"[Progress] 0/{len(scenarios)} episodes completed")
    
    for episode_idx, scen in enumerate(scenarios):
        scen_id = scen.get("id", episode_idx)
        print(f"\n[Episode {episode_idx+1}/{len(scenarios)}] scen_id={scen_id}")
        
        try:
            # 1. 加载协议图
            protocol_graph_path = protocol_dir / f"scen_{scen_id:03d}.npz"
            if not protocol_graph_path.exists():
                print(f"  [FAIL] Protocol graph not found: {protocol_graph_path}")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": "Protocol graph not found"
                })
                fail_result = {
                    "episode": int(episode_id if episode_id is not None else episode_idx),
                    "mode": str(MODE),
                    "scen_id": int(scen_id),
                    "valid": False,
                    "error": "Protocol graph not found",
                }
                if extra_meta:
                    fail_result.update(extra_meta)
                _emit(fail_result)
                continue
            
            proto_data = np.load(str(protocol_graph_path), allow_pickle=False)
            
            # 提取协议图信息
            candidate_mask_np = proto_data["candidate_mask"].astype(bool)
            target_idx_gt = int(proto_data["target_idx_gt"])
            node_spawn_indices = proto_data["node_spawn_indices"].tolist()
            
            protocol_graph = {
                "node_spawn_indices": node_spawn_indices,
                "candidate_indices": proto_data["candidate_indices"].tolist() if "candidate_indices" in proto_data else None,
            }
            
            # 2. 在CARLA里还原场景
            setup_world_for_protocol_scenario(
                world, scen, protocol_graph,
                ego_vehicle, target_vehicle, npc_vehicles
            )
            
            # 记录初始位置
            ego_start_transform = ego_vehicle.get_transform()
            target_start_transform = target_vehicle.get_transform()
            start_dist = math.sqrt(
                (ego_start_transform.location.x - target_start_transform.location.x) ** 2 +
                (ego_start_transform.location.y - target_start_transform.location.y) ** 2
            )
            
            # 记录目标车辆位置（用于可视化）
            target_position = {
                "x": float(target_start_transform.location.x),
                "y": float(target_start_transform.location.y),
                "z": float(target_start_transform.location.z),
            }
            
            # 3. 目标选择（Random模式）- 只在候选集合内随机选择
            cand_indices = np.nonzero(candidate_mask_np)[0]
            if len(cand_indices) == 0:
                print(f"  [FAIL] No candidates in protocol graph")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": "No candidates in protocol graph"
                })
                fail_result = {
                    "episode": int(episode_id if episode_id is not None else episode_idx),
                    "mode": str(MODE),
                    "scen_id": int(scen_id),
                    "valid": False,
                    "error": "No candidates in protocol graph",
                    "target_idx_gt": int(target_idx_gt),
                    "num_candidates": int(candidate_mask_np.sum()),
                }
                if extra_meta:
                    fail_result.update(extra_meta)
                _emit(fail_result)
                continue
            
            chosen_idx_graph = int(np.random.choice(cand_indices))
            random_correct = (chosen_idx_graph == target_idx_gt)
            
            print(f"  [Protocol] chosen_idx_graph={chosen_idx_graph}, target_idx_gt={target_idx_gt}, random_correct={'✓' if random_correct else '✗'}")
            
            # 4. 把chosen_idx_graph映射到具体车辆
            chosen_vehicle = graph_index_to_actor(
                chosen_idx_graph, world, ego_vehicle, target_vehicle, npc_vehicles,
                protocol_graph, scen, max_distance=10.0  # 增加到10米，更宽松的匹配
            )
            
            if chosen_vehicle is None:
                print(f"  [FAIL] Failed to find vehicle for chosen_idx_graph={chosen_idx_graph}")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": f"Failed to find vehicle for chosen_idx_graph={chosen_idx_graph}"
                })
                fail_result = {
                    "episode": int(episode_id if episode_id is not None else episode_idx),
                    "mode": str(MODE),
                    "scen_id": int(scen_id),
                    "valid": False,
                    "error": "Failed to find vehicle for chosen_idx_graph",
                    "target_idx_gt": int(target_idx_gt),
                    "chosen_idx_graph": int(chosen_idx_graph),
                    "num_candidates": int(candidate_mask_np.sum()),
                }
                if extra_meta:
                    fail_result.update(extra_meta)
                _emit(fail_result)
                continue
            
            # 5. 调用统一控制栈（G1/G2/G3）
            try:
                nav_result_obj = nav_controller.run_episode(
                    ego_vehicle=ego_vehicle,
                    target_vehicle=chosen_vehicle,
                    max_steps_total=max_steps
                )
                
                nav_result = {
                    "start_dist": nav_result_obj.start_dist,
                    "min_dist": nav_result_obj.min_dist,
                    "final_dist": nav_result_obj.final_dist,
                    "steps": nav_result_obj.steps,
                    "success_4m": nav_result_obj.success_4m,
                    "success_8m": nav_result_obj.success_8m,
                    "stuck": nav_result_obj.stuck,
                    "trajectory": nav_result_obj.trajectory,
                    "phase_results": nav_result_obj.phase_results,
                    "safety": nav_result_obj.safety,
                }
            except Exception as e:
                print(f"  [ERROR] Navigation failed: {e}")
                import traceback
                traceback.print_exc()
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": f"Navigation failed: {str(e)}"
                })
                fail_result = {
                    "episode": int(episode_id if episode_id is not None else episode_idx),
                    "mode": str(MODE),
                    "scen_id": int(scen_id),
                    "valid": False,
                    "error": f"Navigation failed: {str(e)}",
                    "target_idx_gt": int(target_idx_gt),
                    "chosen_idx_graph": int(chosen_idx_graph),
                    "num_candidates": int(candidate_mask_np.sum()),
                }
                if extra_meta:
                    fail_result.update(extra_meta)
                _emit(fail_result)
                continue
            
            # 6. 记录结果
            result = {
                "episode": int(episode_id if episode_id is not None else episode_idx),
                "mode": str(MODE),
                "scen_id": int(scen_id),
                "valid": True,
                "gs_correct": bool(random_correct),  # 保持字段名一致
                "min_dist": float(nav_result["min_dist"]),
                "success_4m": bool(nav_result["success_4m"]),
                "success_8m": bool(nav_result["success_8m"]),
                "start_dist": float(start_dist),
                "final_dist": float(nav_result["final_dist"]),
                "steps": int(nav_result["steps"]),
                "stuck": bool(nav_result.get("stuck", False)),
                "target_idx_gt": int(target_idx_gt),
                "chosen_idx_graph": int(chosen_idx_graph),
                "num_candidates": int(candidate_mask_np.sum()),
                "trajectory": nav_result.get("trajectory", []),  # 添加轨迹字段
                "safety": nav_result.get("safety", {}),
                "target_position": target_position,  # 目标车辆位置（用于可视化）
            }
            if extra_meta:
                result.update(extra_meta)
            result["controller_profile"] = str(controller_profile)
            result["planner_mode"] = str(planner_mode or "")
            
            results.append(result)
            
            _emit(result)
            
            # 打印进度
            status_4m = "✓" if nav_result["success_4m"] else "✗"
            status_8m = "✓" if nav_result["success_8m"] else "✗"
            random_status = "✓" if random_correct else "✗"
            print(f"  [OK] Start={start_dist:.1f}m → Min={nav_result['min_dist']:.2f}m → Final={nav_result['final_dist']:.2f}m, "
                  f"4m={status_4m}, 8m={status_8m}, Random={random_status}")
            
        except Exception as e:
            error_msg = f"Unexpected exception: {str(e)}"
            print(f"  [ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            failed_scenarios.append({
                "id": scen_id,
                "reason": error_msg
            })
            # 写入失败记录，确保N=150
            fail_result = {
                "episode": int(episode_id if episode_id is not None else episode_idx),
                "mode": str(MODE),
                "scen_id": int(scen_id),
                "valid": False,
                "error_type": "unexpected_exception",
                "error_msg": error_msg[:300],  # 限制长度
                "success_4m": False,
                "success_8m": False,
                "min_dist": None,
                "final_dist": None,
                "steps": 0,
                "stuck": False,
                "timeout": False,
                "collision": False,
                "trajectory": [],
                "phase_results": {},
            }
            if extra_meta:
                fail_result.update(extra_meta)
            _emit(fail_result)
            continue
        
        print(f"[Progress] {episode_idx + 1}/{len(scenarios)} episodes completed")
    
    # 总结
    print(f"\n[Summary]")
    print(f"  Success: {len(results)}/{len(scenarios)}")
    print(f"  Failed: {len(failed_scenarios)}/{len(scenarios)}")
    
    if failed_scenarios:
        print(f"\n[Failed Scenarios]")
        for fs in failed_scenarios:
            print(f"  scen_id={fs['id']}: {fs['reason']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Random Protocol-Driven Online Evaluation")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="data/online_multi/multi_seen_v3.json",
        help="Path to scenario JSON file"
    )
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="scripts/online_multi/protocol_graphs",
        help="Directory containing protocol graphs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs_v2/proto_main_eval/random_proto_driven.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--num-npcs",
        type=int,
        default=10,
        help="Number of NPC vehicles"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max navigation steps"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to run"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=30,
        help="Maximum number of scenes to load before repeated runs; use 0 for all scenes."
    )
    parser.add_argument(
        "--seed-config",
        type=str,
        default=None,
        help="Path to configs/random_seeds.json (list or {configs: [...]})"
    )
    parser.add_argument(
        "--num-runs-per-scene",
        type=int,
        default=5,
        help="Number of repeated runs per scene (R)."
    )
    parser.add_argument(
        "--run-id-offset",
        type=int,
        default=0,
        help="Run id offset (added to 1..R when looking up seed configs)."
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
        help="Traffic Manager port"
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="localhost",
        help="CARLA RPC host"
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA RPC port"
    )
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
        help="Fixed delta seconds for synchronous mode"
    )
    parser.add_argument(
        "--no-rendering",
        action="store_true",
        help="Enable CARLA no_rendering_mode for speed (optional)"
    )
    parser.add_argument(
        "--only-scene-id",
        type=int,
        default=None,
        help="Only run episodes for this scene_id (for isolation testing)"
    )
    parser.add_argument(
        "--only-run-id",
        type=int,
        default=None,
        help="Only run episodes for this run_id (for isolation testing, requires --only-scene-id)"
    )
    parser.add_argument(
        "--controller-profile",
        type=str,
        default="paper_fixed",
        choices=["paper_fixed", "improved_v1", "improved_v2", "improved_v3", "improved_v4", "improved_v5", "improved_v6", "improved_v7", "improved_v8", "improved_v9", "improved_v10", "improved_v11", "improved_v12", "improved_v13", "improved_v14", "improved_v15", "improved_v16", "improved_v17", "improved_v18", "improved_v19", "improved_v20", "improved_v21", "grp_diagnostic"],
        help="Controller profile to use for this evaluation."
    )
    parser.add_argument(
        "--planner-mode",
        type=str,
        default=None,
        choices=["simple", "auto", "global"],
        help="Override the route planner mode from the selected controller profile."
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Random Protocol-Driven Online Evaluation")
    print("=" * 80)
    print(f"[Config] Scenarios: {args.scenarios}")
    print(f"[Config] Protocol dir: {args.protocol_dir}")
    print(f"[Config] Output: {args.output}")
    print(f"[Config] Seed config: {args.seed_config}")
    print(f"[Config] Runs per scene: {args.num_runs_per_scene}")
    print(f"[Config] CARLA: {args.carla_host}:{args.carla_port}")
    print(f"[Config] TM port: {args.tm_port}, fixed_delta={args.fixed_delta_seconds}")
    
    # 加载场景
    with open(args.scenarios, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    print(f"[Load] Loaded {len(scenarios)} scenarios")
    
    # 限制场景数量
    if args.max_scenes and args.max_scenes > 0 and len(scenarios) > args.max_scenes:
        scenarios = scenarios[:args.max_scenes]
        print(f"[Config] Limited to first {args.max_scenes} scenarios")
    
    # 连接CARLA
    print("\n[CARLA] Connecting to CARLA...")
    try:
        client = carla.Client(args.carla_host, int(args.carla_port))
        client.set_timeout(30.0)
        
        first_scenario = scenarios[0]
        town = first_scenario.get("town", "Town10HD_Opt")
        weather = first_scenario.get("weather", "clear")
        
        world = setup_world(client, town, weather)

        # 准备输出文件
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果指定了隔离运行（only-scene-id），使用追加模式，不清空文件
        # 否则清空文件（批量运行模式）
        if args.only_scene_id is None:
            if out_path.exists():
                out_path.unlink()
        else:
            print(f"[Isolation] Using append mode for output file (will not clear existing file)")

        seed_items = load_seed_items(args.seed_config)
        R = int(args.num_runs_per_scene)
        global_ep = 0
        all_results = []

        scenes_to_run = scenarios
        if args.episodes is not None and args.episodes > 0:
            scenes_to_run = scenarios[:args.episodes]
        
        # 隔离运行：只运行指定的scene_id和run_id
        if args.only_scene_id is not None:
            scenes_to_run = [s for s in scenarios if s.get("id", scenarios.index(s)) == args.only_scene_id]
            if not scenes_to_run:
                print(f"[ERROR] scene_id={args.only_scene_id} not found in scenarios")
                return
            print(f"[Isolation] Only running scene_id={args.only_scene_id}")

        print(f"\n[Batch] Running scenes={len(scenes_to_run)} with R={R} (total={len(scenes_to_run)*R})")

        for i, scen in enumerate(scenes_to_run):
            scen_id = scen.get("id", i)
            for r in range(R):
                run_id_1based = (r + 1 + int(args.run_id_offset))
                
                # 隔离运行：只运行指定的run_id
                if args.only_run_id is not None and run_id_1based != args.only_run_id:
                    continue
                
                seed_index = i * R + r

                seed_item = None
                if seed_items is not None:
                    # 优先 (scene_id, run_id) 查找
                    found = None
                    for it in seed_items:
                        if isinstance(it, dict) and it.get("scene_id") == scen_id and it.get("run_id") == run_id_1based:
                            found = it
                            break
                    seed_item = normalize_seed_item(found if found is not None else seed_items[seed_index], seed_index)
                    apply_all_seeds(seed_item)

                tm_seed = int(seed_item.get("traffic_manager_seed", 0)) if seed_item else 0
                carla_seed = int(seed_item.get("carla_seed", 0)) if seed_item else 0
                tm = setup_world_determinism(
                    client, world,
                    tm_port=args.tm_port,
                    tm_seed=tm_seed,
                    carla_seed=carla_seed,
                    fixed_delta_seconds=args.fixed_delta_seconds,
                    no_rendering=args.no_rendering,
                )

                # 每轮重建actors，避免状态污染
                initial_npcs = max(args.num_npcs, 10)
                ego_vehicle = target_vehicle = None
                npc_vehicles = []
                spawn_error = None
                for spawn_attempt in range(3):
                    try:
                        if spawn_attempt > 0:
                            print(f"  [Spawn Retry {spawn_attempt}/2] Reloading world after spawn failure...")
                            try:
                                client.reload_world()
                                world = client.get_world()
                                settings = world.get_settings()
                                settings.synchronous_mode = True
                                settings.fixed_delta_seconds = args.fixed_delta_seconds
                                if args.no_rendering:
                                    settings.no_rendering_mode = True
                                world.apply_settings(settings)
                                tm = client.get_trafficmanager(args.tm_port)
                                tm.set_synchronous_mode(True)
                                try:
                                    tm.set_random_device_seed(tm_seed)
                                except Exception:
                                    pass
                            except Exception as reload_error:
                                print(f"  [Spawn Retry] World reload warning: {reload_error}")
                        ego_vehicle, target_vehicle, npc_vehicles, _tm_local = setup_ego_and_target_vehicles(
                            client, world, num_npcs=initial_npcs
                        )
                        spawn_error = None
                        break
                    except Exception as e:
                        spawn_error = str(e)
                        print(f"  [Spawn Retry {spawn_attempt}] Failed: {spawn_error}")
                        try:
                            destroy_actors_safe(ego_vehicle, target_vehicle, npc_vehicles, world)
                            for _ in range(10):
                                world.tick()
                        except Exception:
                            pass
                if spawn_error is not None:
                    failure_record = {
                        "episode": int(global_ep),
                        "mode": "random_proto_driven",
                        "scen_id": int(scen_id),
                        "scene_id": int(scen_id),
                        "run_id": int(run_id_1based),
                        "seed_index": int(seed_index),
                        "valid": False,
                        "error": f"spawn_failed: {spawn_error}",
                        "target_idx_gt": None,
                        "chosen_idx_graph": None,
                        "gs_correct": False,
                        "controller_profile": str(args.controller_profile),
                        "planner_mode": str(args.planner_mode or ""),
                    }
                    if seed_item:
                        failure_record.update({
                            "seed_id": int(seed_item.get("seed_id", seed_index)),
                            "carla_seed": int(seed_item.get("carla_seed", 0)),
                            "traffic_manager_seed": int(seed_item.get("traffic_manager_seed", 0)),
                            "python_seed": int(seed_item.get("python_seed", 0)),
                            "numpy_seed": int(seed_item.get("numpy_seed", 0)),
                            "torch_seed": int(seed_item.get("torch_seed", 0)),
                        })
                    ensure_safety_fields(failure_record)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                    all_results.append(failure_record)
                    global_ep += 1
                    continue
                # 确保NPC走TM同步
                try:
                    for npc in npc_vehicles:
                        if npc.is_alive:
                            npc.set_autopilot(True, tm.get_port())
                except Exception:
                    pass

                # 稳定世界
                for _ in range(5):
                    world.tick()

                meta = {
                    "scene_id": int(scen_id),
                    "run_id": int(run_id_1based),
                    "seed_index": int(seed_index),
                }
                if seed_item:
                    meta.update({
                        "seed_id": int(seed_item.get("seed_id", seed_index)),
                        "carla_seed": int(seed_item.get("carla_seed", 0)),
                        "traffic_manager_seed": int(seed_item.get("traffic_manager_seed", 0)),
                        "python_seed": int(seed_item.get("python_seed", 0)),
                        "numpy_seed": int(seed_item.get("numpy_seed", 0)),
                        "torch_seed": int(seed_item.get("torch_seed", 0)),
                    })

                try:
                    ep_results = run_evaluation(
                        scenarios=[scen],
                        world=world,
                        ego_vehicle=ego_vehicle,
                        target_vehicle=target_vehicle,
                        npc_vehicles=npc_vehicles,
                        max_steps=args.max_steps,
                        output_path=str(out_path),
                        protocol_dir=args.protocol_dir,
                        max_episodes=1,
                        episode_id=global_ep,
                        extra_meta=meta,
                        controller_profile=str(args.controller_profile),
                        planner_mode=args.planner_mode,
                    )
                    all_results.extend(ep_results)
                finally:
                    destroy_actors_safe(ego_vehicle, target_vehicle, npc_vehicles, world)

                global_ep += 1

        print(f"\n[Done] Wrote {len(all_results)} episodes to {out_path}")
        results = all_results
        
        print(f"\n✅ 评估完成！结果保存到 {args.output}")
        
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n[Cleanup] Destroying actors...")
        try:
            if "world" in locals():
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()

