#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协议驱动的GS模式在线评估

关键区别：
- 不再在CARLA里临时构图决定候选
- 候选集合、target_idx_gt全部来自protocol_graphs/scen_xxx.npz
- 在线只做：从协议图决定目标车 + 用G1/G2/G3控制器开车
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
_script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/online_multi
_online_multi_dir = os.path.dirname(_script_dir)  # scripts
_project_root = os.path.dirname(_online_multi_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 在导入carla之前，先处理CARLA Python API路径
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
    setup_world, load_gs_model,
    load_protocol_graph,
)
from scripts.online.oracle_behavior_demo import setup_ego_and_target_vehicles
from scripts.online_multi.proto_driven_utils import (
    graph_index_to_actor,
    setup_world_for_protocol_scenario,
)
from scripts.online_multi.safety_fields import ensure_safety_fields
from scripts.online_multi.execution_cache import ExecutionCache
from lc_org_nav.online_nav.hierarchical_controller import HierarchicalNavController
from scripts.revision.train_eval_candidate_mlp_baseline import CandidateMLP

######################################################################
# Seed handling & determinism helpers
######################################################################


def load_seed_config(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Load seed items from JSON; return None if path is None.
    支持两种格式：
    - list[dict]
    - { "configs": [...] }（当前工程生成的 random_seeds.json）
    """
    if path is None:
        return None
    seed_path = Path(path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed config not found: {seed_path}")
    data = json.load(open(seed_path, "r", encoding="utf-8"))
    items = data["configs"] if isinstance(data, dict) and "configs" in data else data
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("Seed config must be a non-empty list (or {configs: [...]})")
    return items


def apply_all_seeds(seed_item: Optional[Dict[str, Any]]):
    """Apply Python / NumPy / Torch seeds."""
    if not seed_item:
        return
    py_seed = int(seed_item.get("python_seed", seed_item.get("python_random_seed", 0)))
    np_seed = int(seed_item.get("numpy_seed", seed_item.get("np_seed", 0)))
    torch_seed = int(seed_item.get("torch_seed", 0))
    random.seed(py_seed)
    np.random.seed(np_seed)
    try:
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_relation_mask_from_node_feats(node_feats_np: np.ndarray, rel_type: int) -> np.ndarray:
    """Reconstruct a relation-consistent candidate mask from anchor-relative features."""
    d_a = node_feats_np[:, 7]
    cos_a = node_feats_np[:, 8]
    sin_a = node_feats_np[:, 9]
    is_anchor = node_feats_np[:, 11] if node_feats_np.shape[1] > 11 else np.zeros_like(d_a)
    theta = np.arctan2(sin_a, cos_a)
    near_d = 50.0
    far_min_d = 5.0
    far_max_d = 40.0
    front_deg = np.deg2rad(30.0)
    side_min = np.deg2rad(60.0)
    side_max = np.deg2rad(120.0)
    if rel_type == 0:
        ok = d_a < near_d
    elif rel_type == 1:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (np.abs(theta) < front_deg)
    elif rel_type == 2:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (np.abs(np.abs(theta) - np.pi) < front_deg)
    elif rel_type == 3:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (theta > side_min) & (theta < side_max)
    elif rel_type == 4:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (theta > -side_max) & (theta < -side_min)
    else:
        ok = np.zeros_like(d_a, dtype=bool)
    return ok.astype(bool) & (is_anchor < 0.5)


def query_policy_from_lang(lang_feat_np: np.ndarray) -> int:
    """Decode hard-confusable policy bits from the 32-d query vector."""
    if lang_feat_np.shape[0] >= 9 and float(np.max(lang_feat_np[5:9])) > 0.5:
        return int(np.argmax(lang_feat_np[5:9]))
    return 0


def fine_policy_score(node_feats_np: np.ndarray, idx: int, policy_id: int) -> float:
    """Task-specific policy score used only for the disclosed rule ceiling."""
    d_a = float(node_feats_np[int(idx), 7])
    sin_a = float(node_feats_np[int(idx), 9])
    lateral = d_a * sin_a
    if policy_id == 0:  # nearest
        return -d_a
    if policy_id == 1:  # farthest
        return d_a
    if policy_id == 2:  # leftmost
        return lateral
    if policy_id == 3:  # rightmost
        return -lateral
    return -d_a


def choose_target_index(
    logits: torch.Tensor,
    candidate_mask_np: np.ndarray,
    node_feats_np: np.ndarray,
    lang_feat_np: np.ndarray,
    selector_profile: str,
) -> Dict[str, Any]:
    cand_indices = np.nonzero(candidate_mask_np)[0]
    logits_cpu = logits.detach().cpu()
    rel_type = int(np.argmax(lang_feat_np[:5]))
    if selector_profile == "fine_rank_rule":
        relation_mask = build_relation_mask_from_node_feats(node_feats_np, rel_type)
        gated_mask = candidate_mask_np & relation_mask
        gated_indices = np.nonzero(gated_mask)[0]
        pool = gated_indices if len(gated_indices) > 0 else cand_indices
        policy_id = query_policy_from_lang(lang_feat_np)
        chosen_idx = int(max(pool, key=lambda idx: fine_policy_score(node_feats_np, int(idx), policy_id)))
        return {
            "chosen_idx": chosen_idx,
            "rel_type": rel_type,
            "relation_gate_candidates": [int(x) for x in gated_indices.tolist()],
            "relation_gate_fallback": bool(len(gated_indices) == 0),
            "query_policy_id": int(policy_id),
        }
    if selector_profile == "rule_semantic":
        relation_mask = build_relation_mask_from_node_feats(node_feats_np, rel_type)
        gated_mask = candidate_mask_np & relation_mask
        gated_indices = np.nonzero(gated_mask)[0]
        if len(gated_indices) == 1:
            return {
                "chosen_idx": int(gated_indices[0]),
                "rel_type": rel_type,
                "relation_gate_candidates": [int(gated_indices[0])],
                "relation_gate_fallback": False,
            }
        if len(gated_indices) > 1:
            distances = node_feats_np[gated_indices, 7]
            k_best = int(np.argmin(distances))
            return {
                "chosen_idx": int(gated_indices[k_best]),
                "rel_type": rel_type,
                "relation_gate_candidates": [int(x) for x in gated_indices.tolist()],
                "relation_gate_fallback": False,
            }
        distances = node_feats_np[cand_indices, 7]
        k_best = int(np.argmin(distances))
        return {
            "chosen_idx": int(cand_indices[k_best]),
            "rel_type": rel_type,
            "relation_gate_candidates": [],
            "relation_gate_fallback": True,
        }
    if selector_profile == "relation_gated":
        relation_mask = build_relation_mask_from_node_feats(node_feats_np, rel_type)
        gated_mask = candidate_mask_np & relation_mask
        gated_indices = np.nonzero(gated_mask)[0]
        if len(gated_indices) > 0:
            logits_gated = logits_cpu[torch.from_numpy(gated_mask)]
            k_best = int(torch.argmax(logits_gated).item())
            return {
                "chosen_idx": int(gated_indices[k_best]),
                "rel_type": rel_type,
                "relation_gate_candidates": [int(x) for x in gated_indices.tolist()],
                "relation_gate_fallback": False,
            }
        logits_cands = logits_cpu[torch.from_numpy(candidate_mask_np)]
        k_best = int(torch.argmax(logits_cands).item())
        return {
            "chosen_idx": int(cand_indices[k_best]),
            "rel_type": rel_type,
            "relation_gate_candidates": [],
            "relation_gate_fallback": True,
        }

    logits_cands = logits_cpu[torch.from_numpy(candidate_mask_np)]
    k_best = int(torch.argmax(logits_cands).item())
    return {
        "chosen_idx": int(cand_indices[k_best]),
        "rel_type": rel_type,
        "relation_gate_candidates": [],
        "relation_gate_fallback": False,
    }


def setup_world_determinism(
    client: "carla.Client",
    world: "carla.World",
    tm_port: int,
    tm_seed: int,
    carla_seed: int,
    fixed_delta_seconds: float = 0.05,
):
    """Set synchronous mode, fixed delta, TM seed; tick once to apply."""
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    # Best-effort CARLA seed (pedestrians)
    try:
        world.set_pedestrians_seed(int(carla_seed))
    except Exception:
        pass

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    try:
        tm.set_random_device_seed(int(tm_seed))
    except Exception:
        pass

    world.tick()
    return tm


def select_seed(
    seed_list: Optional[List[Dict[str, Any]]],
    scen_id: int,
    run_id_1based: int,
    scene_idx: int,
    run_local: int,
    runs_per_scene: int,
) -> Optional[Dict[str, Any]]:
    """Choose seed by (scen_id, run_id_1based) if available; else fallback sequential by (scene_idx, run_local)."""
    if seed_list is None:
        return None
    # If seed has scene_id/run_id, try exact match
    for item in seed_list:
        if "scene_id" in item and "run_id" in item:
            if int(item["scene_id"]) == int(scen_id) and int(item["run_id"]) == int(run_id_1based):
                return item
    # Fallback: sequential
    idx = scene_idx * runs_per_scene + run_local
    if idx >= len(seed_list):
        raise IndexError(f"Seed index {idx} out of range (len={len(seed_list)})")
    return seed_list[idx]


MODE = "gs_proto_driven"


def build_candidate_mlp_features(
    node_feats_np: np.ndarray,
    lang_feat_np: np.ndarray,
    candidate_mask_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the same candidate-wise MLP+pooling features used in offline training."""
    cand_indices = np.nonzero(candidate_mask_np)[0].astype(np.int64)
    if len(cand_indices) == 0:
        dim = int(node_feats_np.shape[1] * 3 + lang_feat_np.shape[0])
        return np.zeros((0, dim), dtype=np.float32), cand_indices
    node_feats = node_feats_np.astype(np.float32)
    lang_feat = lang_feat_np.astype(np.float32)
    mean_pool = node_feats.mean(axis=0)
    max_pool = node_feats.max(axis=0)
    rows = [
        np.concatenate([node_feats[int(idx)], mean_pool, max_pool, lang_feat], axis=0)
        for idx in cand_indices
    ]
    return np.stack(rows).astype(np.float32), cand_indices


def load_candidate_mlp_model(
    checkpoint_path: str,
    protocol_dir: str,
    device: torch.device,
    hidden_dim: int = 128,
) -> CandidateMLP:
    """Load the query-aware MLP baseline from its checkpoint."""
    sample_paths = sorted(Path(protocol_dir).glob("scen_*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No protocol graph files found in {protocol_dir}")
    sample = np.load(str(sample_paths[0]), allow_pickle=False)
    x_sample, _ = build_candidate_mlp_features(
        node_feats_np=sample["node_feats"],
        lang_feat_np=sample["lang_feat"],
        candidate_mask_np=sample["candidate_mask"].astype(bool),
    )
    if len(x_sample) == 0:
        raise RuntimeError(f"Protocol graph has no candidates: {sample_paths[0]}")
    model = CandidateMLP(input_dim=int(x_sample.shape[1]), hidden_dim=int(hidden_dim)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_config' in state and 'state_dict' in state:
        from r3_selectors.pipeline import load_checkpoint
        from lc_org_nav.online_nav.gs_loader import _R3CheckpointAdapter
        bundle = load_checkpoint(checkpoint_path, device=str(device))
        adapter = _R3CheckpointAdapter(bundle)
        adapter.eval()
        return adapter
    model.load_state_dict(state)
    model.eval()
    return model


def run_evaluation(
    client: carla.Client,
    scenarios: List[Dict[str, Any]],
    world: carla.World,
    model,
    device: torch.device,
    max_steps: int,
    output_path: str,
    protocol_dir: str,
    max_episodes: Optional[int] = None,
    carla_python_api_path: Optional[str] = None,
    seed_list: Optional[List[Dict[str, Any]]] = None,
    num_runs_per_scene: int = 1,
    run_id_offset: int = 0,
    tm_port: int = 8000,
    fixed_delta_seconds: float = 0.05,
    no_rendering: bool = False,
    num_npcs: int = 10,
    max_spawn_retries: int = 3,
    controller_profile: str = "paper_fixed",
    planner_mode: Optional[str] = None,
    selector_profile: str = "neural",
    model_type: str = "gnn",
    execution_cache: Optional[ExecutionCache] = None,
) -> List[Dict[str, Any]]:
    """
    运行协议驱动的GS模式评估
    
    关键：候选集合、target_idx_gt全部来自protocol_graphs/scen_xxx.npz
    """
    protocol_dir = Path(protocol_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 清空输出文件
    if output_path.exists():
        output_path.unlink()
    
    if max_episodes is not None and max_episodes > 0:
        scenarios = scenarios[:max_episodes]
        print(f"[Config] Limited to first {max_episodes} episodes")
    
    results = []
    failed_scenarios = []
    total_runs = len(scenarios) * num_runs_per_scene
    completed = 0

    def _emit(result: Dict[str, Any]) -> None:
        """每个(scene, run)写一行JSONL（成功/失败都写），用于严格对齐30×R统计。"""
        ensure_safety_fields(result)
        with open(output_path, 'a', encoding='utf-8') as f:
            json_str = json.dumps(result, ensure_ascii=False, separators=(',', ':'), indent=None)
            f.write(json_str + '\n')
            f.flush()

    carla_map = world.get_map()
    nav_controller = HierarchicalNavController(
        world=world,
        carla_map=carla_map,
        success_radius_4m=4.0,
        success_radius_8m=8.0,
        record_trajectory=True,
        traj_subsample=2,
        carla_python_api_path=carla_python_api_path,
        controller_profile=controller_profile,
        planner_mode=planner_mode,
    )

    print(f"\n[Evaluation] Running {total_runs} episodes in {MODE.upper()} mode...")
    print(f"[Progress] 0/{total_runs} episodes completed")

    for scene_idx, scen in enumerate(scenarios):
        scen_id = scen.get("id", scene_idx)
        print(f"\n[Scene {scene_idx+1}/{len(scenarios)}] scen_id={scen_id}")

        for run_local in range(num_runs_per_scene):
            run_id_1based = (run_local + 1 + int(run_id_offset))
            print(f"  [Run {run_local+1}/{num_runs_per_scene}]")
            episode_index = scene_idx * num_runs_per_scene + run_local

            # 选择并应用种子
            seed_item = select_seed(
                seed_list=seed_list,
                scen_id=int(scen_id),
                run_id_1based=int(run_id_1based),
                scene_idx=scene_idx,
                run_local=run_local,
                runs_per_scene=num_runs_per_scene,
            ) if seed_list else None
            apply_all_seeds(seed_item)

            # 设置同步模式 & TM seed
            tm_seed = seed_item.get("traffic_manager_seed", seed_item.get("tm_seed", 0)) if seed_item else 0
            carla_seed = seed_item.get("carla_seed", 0) if seed_item else 0
            tm = setup_world_determinism(
                client=client,
                world=world,
                tm_port=tm_port,
                tm_seed=tm_seed,
                carla_seed=carla_seed,
                fixed_delta_seconds=float(fixed_delta_seconds),
            )
            try:
                if no_rendering:
                    settings = world.get_settings()
                    settings.no_rendering_mode = True
                    world.apply_settings(settings)
            except Exception:
                pass

            # 每轮重建actor
            ego_vehicle = None
            target_vehicle = None
            npc_vehicles = []
            spawn_error = None
            for spawn_attempt in range(max(1, int(max_spawn_retries))):
                try:
                    if spawn_attempt > 0:
                        print(f"    [Spawn Retry {spawn_attempt}/{max_spawn_retries-1}] Reloading world after spawn failure...")
                        try:
                            client.reload_world()
                            world = client.get_world()
                            for _ in range(10):
                                world.tick()
                        except Exception:
                            pass
                        tm = setup_world_determinism(
                            client=client,
                            world=world,
                            tm_port=tm_port,
                            tm_seed=tm_seed,
                            carla_seed=carla_seed,
                            fixed_delta_seconds=float(fixed_delta_seconds),
                        )
                    ego_vehicle, target_vehicle, npc_vehicles, _tm = setup_ego_and_target_vehicles(
                        client, world, num_npcs=max(int(num_npcs), 0)
                    )
                    spawn_error = None
                    break
                except Exception as e:
                    spawn_error = str(e)
                    print(f"    [Spawn Retry {spawn_attempt}] Failed: {spawn_error}")
                    try:
                        destroy_actors_safe(ego_vehicle, target_vehicle, npc_vehicles, world)
                    except Exception:
                        pass
                    ego_vehicle = None
                    target_vehicle = None
                    npc_vehicles = []

            if spawn_error is not None:
                print(f"    [ERROR] Spawn vehicles failed: {spawn_error}")
                failed_scenarios.append({"id": scen_id, "reason": f"spawn_failed: {spawn_error}"})
                fail_result = {
                    "episode": int(episode_index),
                    "mode": str(MODE),
                    "scen_id": int(scen_id),
                    "scene_id": int(scen_id),
                    "run_id": int(run_id_1based),
                    "seed_index": int(episode_index),
                    "valid": False,
                    "error": f"spawn_failed: {spawn_error}",
                }
                fail_result["controller_profile"] = str(controller_profile)
                fail_result["planner_mode"] = str(planner_mode or "")

                if seed_item:
                    fail_result.update({
                        "seed_id": seed_item.get("seed_id", episode_index),
                        "carla_seed": seed_item.get("carla_seed"),
                        "traffic_manager_seed": seed_item.get("traffic_manager_seed", seed_item.get("tm_seed", None)),
                        "python_seed": seed_item.get("python_seed", seed_item.get("python_random_seed", None)),
                        "numpy_seed": seed_item.get("numpy_seed", seed_item.get("np_seed", None)),
                        "torch_seed": seed_item.get("torch_seed", None),
                    })
                _emit(fail_result)
                completed += 1
                print(f"[Progress] {completed}/{total_runs} episodes completed")
                continue

            try:
                # 1. 加载协议图
                protocol_graph_path = protocol_dir / f"scen_{scen_id:03d}.npz"
                if not protocol_graph_path.exists():
                    print(f"    [FAIL] Protocol graph not found: {protocol_graph_path}")
                    failed_scenarios.append({
                        "id": scen_id,
                        "reason": "Protocol graph not found"
                    })
                    fail_result = {
                        "episode": int(episode_index),
                        "mode": str(MODE),
                        "scen_id": int(scen_id),
                        "scene_id": int(scen_id),
                        "run_id": int(run_id_1based),
                        "seed_index": int(episode_index),
                        "valid": False,
                        "error": "Protocol graph not found",
                    }
                    if seed_item:
                        fail_result.update({
                            "seed_id": seed_item.get("seed_id", episode_index),
                            "carla_seed": seed_item.get("carla_seed"),
                            "traffic_manager_seed": seed_item.get("traffic_manager_seed", seed_item.get("tm_seed", None)),
                            "python_seed": seed_item.get("python_seed", seed_item.get("python_random_seed", None)),
                            "numpy_seed": seed_item.get("numpy_seed", seed_item.get("np_seed", None)),
                            "torch_seed": seed_item.get("torch_seed", None),
                        })
                    _emit(fail_result)
                    completed += 1
                    print(f"[Progress] {completed}/{total_runs} episodes completed")
                    raise RuntimeError("protocol graph missing")

                proto_data = np.load(str(protocol_graph_path), allow_pickle=False)

                # 提取协议图信息
                node_feats_np = proto_data["node_feats"]
                edge_index_np = proto_data["edge_index"]
                edge_attr_np = proto_data["edge_attr"]
                lang_feat_np = proto_data["lang_feat"]
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

                # 3. 目标选择（GS模式）- 使用协议图
                node_feats = torch.from_numpy(node_feats_np).float().to(device)
                edge_index = torch.from_numpy(edge_index_np).long().to(device)
                edge_attr = torch.from_numpy(edge_attr_np).float().to(device)
                lang_feat = torch.from_numpy(lang_feat_np).float().to(device)
                candidate_mask = torch.from_numpy(candidate_mask_np).bool().to(device)

                if model_type == "rule_semantic":
                    goal_logits = torch.zeros((len(candidate_mask_np),), dtype=torch.float32)
                else:
                    model.eval()
                    with torch.no_grad():
                        if model_type == "candidate_mlp":
                            if hasattr(model, "forward_single"):
                                outputs = model.forward_single(
                                    node_feats=node_feats,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    lang_feat=lang_feat,
                                    candidate_mask=candidate_mask,
                                )
                                goal_logits = outputs["goal_logits"]
                            else:
                                x_mlp, cand_indices_mlp = build_candidate_mlp_features(
                                    node_feats_np=node_feats_np,
                                    lang_feat_np=lang_feat_np,
                                    candidate_mask_np=candidate_mask_np,
                                )
                                if len(cand_indices_mlp) == 0:
                                    raise RuntimeError("candidate MLP selector received an empty candidate mask")
                                cand_logits = model(torch.from_numpy(x_mlp).float().to(device)).detach().cpu()
                                goal_logits = torch.full((len(candidate_mask_np),), -1e9, dtype=torch.float32)
                                goal_logits[torch.from_numpy(cand_indices_mlp)] = cand_logits
                        else:
                            outputs = model.forward_single(
                                node_feats=node_feats,
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                lang_feat=lang_feat,
                                candidate_mask=candidate_mask,
                            )
                            goal_logits = outputs["goal_logits"]  # (N,)

                # 只在候选集合上取argmax
                logits = goal_logits.view(-1)
                choice = choose_target_index(
                    logits=logits,
                    candidate_mask_np=candidate_mask_np,
                    node_feats_np=node_feats_np.astype(np.float32),
                    lang_feat_np=lang_feat_np.astype(np.float32),
                    selector_profile=selector_profile,
                )
                chosen_idx_graph = int(choice["chosen_idx"])

                gs_correct = (chosen_idx_graph == target_idx_gt)

                print(f"    [Protocol] chosen_idx_graph={chosen_idx_graph}, target_idx_gt={target_idx_gt}, gs_correct={'✓' if gs_correct else '✗'}")

                # 4. 把chosen_idx_graph映射到具体车辆
                chosen_vehicle = graph_index_to_actor(
                    chosen_idx_graph, world, ego_vehicle, target_vehicle, npc_vehicles,
                    protocol_graph, scen, max_distance=5.0
                )

                if chosen_vehicle is None:
                    print(f"    [FAIL] Failed to find vehicle for chosen_idx_graph={chosen_idx_graph}")
                    failed_scenarios.append({
                        "id": scen_id,
                        "reason": f"Failed to find vehicle for chosen_idx_graph={chosen_idx_graph}"
                    })
                    fail_result = {
                        "episode": int(episode_index),
                        "mode": str(MODE),
                        "scen_id": int(scen_id),
                        "scene_id": int(scen_id),
                        "run_id": int(run_id_1based),
                        "seed_index": int(episode_index),
                        "valid": False,
                        "error": "Failed to find vehicle for chosen_idx_graph",
                        "target_idx_gt": int(target_idx_gt),
                        "chosen_idx_graph": int(chosen_idx_graph),
                        "num_candidates": int(candidate_mask_np.sum()),
                        "model_type": str(model_type),
                        "selector_profile": str(selector_profile),
                        "rel_type": int(choice["rel_type"]),
                        "relation_gate_candidates": choice["relation_gate_candidates"],
                        "relation_gate_fallback": bool(choice["relation_gate_fallback"]),
                    }
                    if seed_item:
                        fail_result.update({
                            "seed_id": seed_item.get("seed_id", episode_index),
                            "carla_seed": seed_item.get("carla_seed"),
                            "traffic_manager_seed": seed_item.get("traffic_manager_seed", seed_item.get("tm_seed", None)),
                            "python_seed": seed_item.get("python_seed", seed_item.get("python_random_seed", None)),
                            "numpy_seed": seed_item.get("numpy_seed", seed_item.get("np_seed", None)),
                            "torch_seed": seed_item.get("torch_seed", None),
                        })
                    _emit(fail_result)
                    completed += 1
                    print(f"[Progress] {completed}/{total_runs} episodes completed")
                    raise RuntimeError("chosen vehicle missing")

                # 5. 共享命令执行复用：同一 (scene, seed, 选中身份) 只消耗一次物理实现
                # 契约：给定冻结布局、冻结种子与冻结控制器配置，物理结果只是命令的函数，
                # 因此发出相同命令的选择器必须复用同一次实现，而不是各自新跑一遍。
                cache_key = None
                execution_source = "executed"
                reused_payload = None
                if execution_cache is not None:
                    cache_key = execution_cache.make_key(
                        scen_id=int(scen_id),
                        carla_seed=(seed_item or {}).get("carla_seed"),
                        chosen_idx_graph=int(chosen_idx_graph),
                        controller_profile=controller_profile,
                        planner_mode=planner_mode or "",
                    )
                    reused_payload = execution_cache.get(cache_key)

                # 6. 调用统一控制栈（G1/G2/G3）
                try:
                    if reused_payload is not None:
                        execution_source = "reused"
                        nav_result = dict(reused_payload.get("nav_result", {}))
                        print(f"    [Cache] reused physical realization for {cache_key}")
                    else:
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
                    print(f"    [ERROR] Navigation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_scenarios.append({
                        "id": scen_id,
                        "reason": f"Navigation failed: {str(e)}"
                    })
                    fail_result = {
                        "episode": int(episode_index),
                        "mode": str(MODE),
                        "scen_id": int(scen_id),
                        "scene_id": int(scen_id),
                        "run_id": int(run_id_1based),
                        "seed_index": int(episode_index),
                        "valid": False,
                        "error": f"Navigation failed: {str(e)}",
                        "target_idx_gt": int(target_idx_gt),
                        "chosen_idx_graph": int(chosen_idx_graph),
                        "num_candidates": int(candidate_mask_np.sum()),
                        "model_type": str(model_type),
                        "selector_profile": str(selector_profile),
                        "rel_type": int(choice["rel_type"]),
                        "relation_gate_candidates": choice["relation_gate_candidates"],
                        "relation_gate_fallback": bool(choice["relation_gate_fallback"]),
                    }
                    if seed_item:
                        fail_result.update({
                            "seed_id": seed_item.get("seed_id", episode_index),
                            "carla_seed": seed_item.get("carla_seed"),
                            "traffic_manager_seed": seed_item.get("traffic_manager_seed", seed_item.get("tm_seed", None)),
                            "python_seed": seed_item.get("python_seed", seed_item.get("python_random_seed", None)),
                            "numpy_seed": seed_item.get("numpy_seed", seed_item.get("np_seed", None)),
                            "torch_seed": seed_item.get("torch_seed", None),
                        })
                    _emit(fail_result)
                    completed += 1
                    print(f"[Progress] {completed}/{total_runs} episodes completed")
                    raise

                # 7. 仅在真正执行时写入复用库（复用命中不产生新实现）
                if reused_payload is None and execution_cache is not None and cache_key is not None:
                    execution_cache.put(
                        cache_key,
                        nav_result=nav_result,
                        actor={
                            "actor_id": getattr(chosen_vehicle, "id", None),
                            "type_id": getattr(chosen_vehicle, "type_id", None),
                        },
                        source={
                            "scen_id": int(scen_id),
                            "carla_seed": (seed_item or {}).get("carla_seed"),
                            "chosen_idx_graph": int(chosen_idx_graph),
                            "model_type": str(model_type),
                            "selector_profile": str(selector_profile),
                        },
                    )

                # 8. 记录结果
                result = {
                    "episode": int(episode_index),
                    "mode": str(MODE),
                    "scen_id": int(scen_id),
                    "scene_id": int(scen_id),
                    "run_id": int(run_id_1based),
                    "seed_index": int(scene_idx * num_runs_per_scene + run_local),
                    "valid": True,
                    "gs_correct": bool(gs_correct),
                    "min_dist": float(nav_result["min_dist"]),
                    "success_4m": bool(nav_result["success_4m"]),
                    "success_8m": bool(nav_result["success_8m"]),
                    "start_dist": float(start_dist),
                    "final_dist": float(nav_result["final_dist"]),
                    "steps": int(nav_result["steps"]),
                    "stuck": bool(nav_result.get("stuck", False)),
                    "target_idx_gt": int(target_idx_gt),
                    "chosen_idx_graph": int(chosen_idx_graph),  # 协议图索引
                    "num_candidates": int(candidate_mask_np.sum()),
                    "model_type": str(model_type),
                    "selector_profile": str(selector_profile),
                    "rel_type": int(choice["rel_type"]),
                    "relation_gate_candidates": choice["relation_gate_candidates"],
                    "relation_gate_fallback": bool(choice["relation_gate_fallback"]),
                    "trajectory": nav_result.get("trajectory", []),  # 添加轨迹字段
                    "safety": nav_result.get("safety", {}),
                    "target_position": target_position,  # 目标车辆位置（用于可视化）
                }

                if seed_item:
                    result.update({
                        "seed_id": seed_item.get("seed_id", scene_idx * num_runs_per_scene + run_local),
                        "carla_seed": seed_item.get("carla_seed"),
                        "traffic_manager_seed": seed_item.get("traffic_manager_seed", seed_item.get("tm_seed", None)),
                        "python_seed": seed_item.get("python_seed", seed_item.get("python_random_seed", None)),
                        "numpy_seed": seed_item.get("numpy_seed", seed_item.get("np_seed", None)),
                        "torch_seed": seed_item.get("torch_seed", None),
                    })
                result["controller_profile"] = str(controller_profile)
                result["planner_mode"] = str(planner_mode or "")
                result["execution_source"] = str(execution_source)
                result["cache_key"] = cache_key

                results.append(result)
                _emit(result)

                # 打印进度
                status_4m = "✓" if nav_result["success_4m"] else "✗"
                status_8m = "✓" if nav_result["success_8m"] else "✗"
                gs_status = "✓" if gs_correct else "✗"
                print(f"    [OK] Start={start_dist:.1f}m → Min={nav_result['min_dist']:.2f}m → Final={nav_result['final_dist']:.2f}m, "
                      f"4m={status_4m}, 8m={status_8m}, GS={gs_status}")

                completed += 1
                print(f"[Progress] {completed}/{total_runs} episodes completed")

            except Exception as e:
                # 已在上面记录failed_scenarios
                pass
            finally:
                # 清理actors，避免状态污染
                try:
                    if 'ego_vehicle' in locals() and ego_vehicle.is_alive:
                        ego_vehicle.destroy()
                    if 'target_vehicle' in locals() and target_vehicle.is_alive:
                        target_vehicle.destroy()
                    for npc in locals().get("npc_vehicles", []):
                        if npc.is_alive:
                            npc.destroy()
                except Exception:
                    pass
                try:
                    world.tick()
                except Exception:
                    pass

    # 总结
    print(f"\n[Summary]")
    print(f"  Success: {len(results)}/{total_runs}")
    print(f"  Failed: {len(failed_scenarios)}/{total_runs}")

    if failed_scenarios:
        print(f"\n[Failed Scenarios]")
        for fs in failed_scenarios:
            print(f"  scen_id={fs['id']}: {fs['reason']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GS Protocol-Driven Online Evaluation")
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
        "--checkpoint",
        type=str,
        default="",
        help="Path to selector model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs_v2/proto_main_eval/gs_proto_driven.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
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
        "--carla-python-api-path",
        type=str,
        default=None,
        help="Path to CARLA Python API"
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
        "--max-spawn-retries",
        type=int,
        default=3,
        help="Number of CARLA spawn retries after collision or residual actor failures."
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
    parser.add_argument(
        "--selector-profile",
        type=str,
        default="neural",
        choices=["neural", "relation_gated", "rule_semantic", "fine_rank_rule"],
        help="Target selector profile; default preserves the original neural-only selector."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gnn",
        choices=["gnn", "candidate_mlp", "rule_semantic"],
        help="Selector model implementation. Use candidate_mlp for the query-aware MLP+pooling baseline or rule_semantic for the deterministic relation rule."
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the CandidateMLP baseline checkpoint."
    )
    
    parser.add_argument(
        "--execution-cache",
        type=str,
        default=None,
        help="Path to a shared-command execution reuse library (JSONL). Commands that "
             "share (scene, rollout seed, selected identity, controller) reuse one "
             "physical realization instead of spending a new CARLA episode."
    )
    parser.add_argument(
        "--execution-cache-readonly",
        action="store_true",
        help="Reuse-only mode: never append a new realization to the library."
    )
    parser.add_argument(
        "--execution-cache-stats",
        type=str,
        default=None,
        help="Optional path for a JSON hit/miss/write statistics file of the library."
    )

    args = parser.parse_args()
    
    print("=" * 80)
    print("GS Protocol-Driven Online Evaluation")
    print("=" * 80)
    print(f"[Config] Scenarios: {args.scenarios}")
    print(f"[Config] Protocol dir: {args.protocol_dir}")
    print(f"[Config] Checkpoint: {args.checkpoint}")
    print(f"[Config] Output: {args.output}")
    print(f"[Config] Device: {args.device}")
    print(f"[Config] Seed config: {args.seed_config}")
    print(f"[Config] Runs per scene: {args.num_runs_per_scene}")
    print(f"[Config] CARLA: {args.carla_host}:{args.carla_port}")
    print(f"[Config] TM port: {args.tm_port}, fixed_delta={args.fixed_delta_seconds}")
    print(f"[Config] Model type: {args.model_type}")
    print(f"[Config] Selector profile: {args.selector_profile}")
    
    # 加载场景
    with open(args.scenarios, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    print(f"[Load] Loaded {len(scenarios)} scenarios")
    
    # 限制场景数量
    if args.max_scenes and args.max_scenes > 0 and len(scenarios) > args.max_scenes:
        scenarios = scenarios[:args.max_scenes]
        print(f"[Config] Limited to first {args.max_scenes} scenarios")
    
    # 加载模型
    device = torch.device(args.device)
    print(f"\n[Model] Loading selector model ({args.model_type})...")
    if args.model_type == "rule_semantic":
        model = None
        if args.selector_profile == "neural":
            args.selector_profile = "rule_semantic"
        print(f"[Model] Rule selector profile={args.selector_profile}; no neural checkpoint is loaded.")
    elif args.model_type == "candidate_mlp":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --model-type candidate_mlp")
        model = load_candidate_mlp_model(
            args.checkpoint,
            args.protocol_dir,
            device,
            hidden_dim=int(args.mlp_hidden_dim),
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --model-type gnn")
        model = load_gs_model(args.checkpoint, device)
    
    # 连接CARLA
    print("\n[CARLA] Connecting to CARLA...")
    try:
        client = carla.Client(args.carla_host, int(args.carla_port))
        client.set_timeout(30.0)
        
        first_scenario = scenarios[0]
        town = first_scenario.get("town", "Town10HD_Opt")
        weather = first_scenario.get("weather", "clear")
        
        world = setup_world(client, town, weather)
        
        # 载入seed config（可选）
        seed_list = load_seed_config(args.seed_config)

        # 共享命令执行复用库：给定冻结布局/种子/控制器，物理结果只是命令的函数
        execution_cache = None
        if args.execution_cache:
            execution_cache = ExecutionCache(
                args.execution_cache,
                readonly=bool(args.execution_cache_readonly),
            )
            stats = execution_cache.stats()
            print(f"[Cache] library={stats['path']} entries={stats['entries']} "
                  f"readonly={stats['readonly']}")

        # 运行评估（内部每轮会spawn/destroy actors）
        results = run_evaluation(
            scenarios=scenarios,
            client=client,
            world=world,
            model=model,
            device=device,
            max_steps=args.max_steps,
            output_path=args.output,
            protocol_dir=args.protocol_dir,
            max_episodes=args.episodes,
            carla_python_api_path=args.carla_python_api_path,
            seed_list=seed_list,
            num_runs_per_scene=int(args.num_runs_per_scene),
            run_id_offset=int(args.run_id_offset),
            tm_port=int(args.tm_port),
            fixed_delta_seconds=float(args.fixed_delta_seconds),
            no_rendering=bool(args.no_rendering),
            num_npcs=int(args.num_npcs),
            max_spawn_retries=int(args.max_spawn_retries),
            controller_profile=str(args.controller_profile),
            planner_mode=args.planner_mode,
            selector_profile=str(args.selector_profile),
            model_type=str(args.model_type),
            execution_cache=execution_cache,
        )

        if execution_cache is not None:
            cache_stats = execution_cache.stats()
            cache_stats.update({
                "scenarios": len(scenarios),
                "selector_profile": str(args.selector_profile),
                "model_type": str(args.model_type),
                "output": str(args.output),
            })
            print(f"[Cache] hits={cache_stats['hits']} misses={cache_stats['misses']} "
                  f"writes={cache_stats['writes']} entries={cache_stats['entries']}")
            if args.execution_cache_stats:
                execution_cache.write_stats(args.execution_cache_stats, extra={
                    "scenarios": len(scenarios),
                    "selector_profile": str(args.selector_profile),
                    "model_type": str(args.model_type),
                    "output": str(args.output),
                })
        
        print(f"\n✅ 评估完成！结果保存到 {args.output}")
        
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n[Cleanup] Destroying actors...")
        try:
            if 'ego_vehicle' in locals() and ego_vehicle.is_alive:
                ego_vehicle.destroy()
            if 'target_vehicle' in locals() and target_vehicle.is_alive:
                target_vehicle.destroy()
            for npc in npc_vehicles:
                if npc.is_alive:
                    npc.destroy()
        except:
            pass


if __name__ == "__main__":
    main()

