#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate larger protocol-scenario JSON files for CARLA closed-loop reruns."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _add_carla_python_api(carla_root: str) -> None:
    paths = [
        os.environ.get("CARLA_PYTHON_API_PATH"),
        str(Path(carla_root) / "WindowsNoEditor" / "PythonAPI"),
        str(Path(carla_root) / "PythonAPI"),
    ]
    for path in paths:
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
    dist_dir = Path(carla_root) / "WindowsNoEditor" / "PythonAPI" / "carla" / "dist"
    if dist_dir.exists():
        eggs = sorted(dist_dir.glob("carla-*.egg"))
        for egg in eggs:
            egg_s = str(egg)
            if egg_s not in sys.path:
                sys.path.insert(0, egg_s)


def _dist_xy(a: Any, b: Any) -> float:
    return math.hypot(a.location.x - b.location.x, a.location.y - b.location.y)


def _make_seed_config(num_scenes: int, runs_per_scene: int, base_seed: int) -> Dict[str, Any]:
    configs = []
    for scene_id in range(num_scenes):
        for run_id in range(1, runs_per_scene + 1):
            base = int(base_seed) + int(scene_id) * 1000 + int(run_id) * 100
            configs.append(
                {
                    "scene_id": int(scene_id),
                    "run_id": int(run_id),
                    "carla_seed": base + 1,
                    "traffic_manager_seed": base + 2,
                    "python_random_seed": base + 3,
                    "numpy_seed": base + 4,
                    "torch_seed": base + 5,
                }
            )
    return {
        "base_seed": int(base_seed),
        "num_runs_per_scene": int(runs_per_scene),
        "configs": configs,
    }


def _candidate_pool(
    spawn_points: Sequence[Any],
    ego_idx: int,
    min_dist: float,
    max_dist: float,
) -> List[Tuple[int, float]]:
    ego_tf = spawn_points[ego_idx]
    out: List[Tuple[int, float]] = []
    for idx, tf in enumerate(spawn_points):
        if idx == ego_idx:
            continue
        d = _dist_xy(ego_tf, tf)
        if min_dist <= d <= max_dist:
            out.append((idx, d))
    return out


def _generate_new_scenario(
    scen_id: int,
    spawn_points: Sequence[Any],
    rng: random.Random,
    town: str,
    weather: str,
    min_dist: float,
    max_dist: float,
    target_order: str,
    used_triplets: set,
) -> Dict[str, Any]:
    all_ego = list(range(len(spawn_points)))
    rng.shuffle(all_ego)

    for ego_idx in all_ego:
        pool = _candidate_pool(spawn_points, ego_idx, min_dist, max_dist)
        if len(pool) < 2:
            continue
        rng.shuffle(pool)

        for target_idx, target_dist in pool:
            distractors = [(idx, d) for idx, d in pool if idx != target_idx]
            if not distractors:
                continue
            # Prefer a distractor close enough to make the target-selection problem non-trivial.
            distractors.sort(key=lambda item: abs(item[1] - target_dist))
            top_k = distractors[: min(8, len(distractors))]
            distractor_idx, distractor_dist = rng.choice(top_k)

            triplet = (ego_idx, target_idx, distractor_idx)
            if triplet in used_triplets:
                continue
            used_triplets.add(triplet)

            if target_order == "compatible":
                node_spawn_indices = [ego_idx, target_idx, distractor_idx]
                target_index_gt = 1
            elif target_order == "balanced":
                target_slot = 1 if (scen_id % 2 == 0) else 2
                if target_slot == 1:
                    node_spawn_indices = [ego_idx, target_idx, distractor_idx]
                    target_index_gt = 1
                else:
                    node_spawn_indices = [ego_idx, distractor_idx, target_idx]
                    target_index_gt = 2
            else:
                raise ValueError(f"Unsupported target_order: {target_order}")

            candidate_indices = [1, 2]
            candidate_spawn_indices = [ego_idx, node_spawn_indices[1], node_spawn_indices[2]]
            candidate_dists = [
                0.0,
                round(_dist_xy(spawn_points[ego_idx], spawn_points[node_spawn_indices[1]]), 2),
                round(_dist_xy(spawn_points[ego_idx], spawn_points[node_spawn_indices[2]]), 2),
            ]

            return {
                "id": int(scen_id),
                "town": town,
                "weather": weather,
                "ego_spawn_idx": int(ego_idx),
                "anchor_spawn_idx": int(ego_idx),
                "target_spawn_idx": int(target_idx),
                "target_idx_gt": int(target_index_gt),
                "candidate_spawn_indices": [int(x) for x in candidate_spawn_indices],
                "candidate_dists": candidate_dists,
                "num_candidates": 3,
                "start_dist": round(float(target_dist), 2),
                "rel_type": 0,
                "seed": int(scen_id * 1000 + 42),
                "node_spawn_indices": [int(x) for x in node_spawn_indices],
                "candidate_indices": candidate_indices,
                "target_index_gt": int(target_index_gt),
            }

    raise RuntimeError("Could not generate a valid scenario with the requested constraints")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", default="data/online_multi/multi_seen_v3.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-config-output", default=None)
    parser.add_argument("--total-scenes", type=int, default=60)
    parser.add_argument("--runs-per-scene", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--target-order", choices=["compatible", "balanced"], default="compatible")
    parser.add_argument("--min-dist", type=float, default=10.0)
    parser.add_argument("--max-dist", type=float, default=50.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "E:/carla"))
    args = parser.parse_args()

    _add_carla_python_api(args.carla_root)
    import carla  # type: ignore

    existing_path = Path(args.existing)
    existing: List[Dict[str, Any]] = []
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))

    scenarios = []
    for i, scen in enumerate(existing[: args.total_scenes]):
        item = dict(scen)
        item["id"] = int(i)
        scenarios.append(item)

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(30.0)
    world = client.load_world(args.town)
    spawn_points = world.get_map().get_spawn_points()

    rng = random.Random(int(args.base_seed))
    used_triplets = {
        (
            int(s.get("ego_spawn_idx", -1)),
            int(s.get("target_spawn_idx", -1)),
            int(s.get("node_spawn_indices", [None, None, None])[2]),
        )
        for s in scenarios
        if len(s.get("node_spawn_indices", [])) >= 3
    }

    next_id = len(scenarios)
    while len(scenarios) < args.total_scenes:
        scen = _generate_new_scenario(
            scen_id=next_id,
            spawn_points=spawn_points,
            rng=rng,
            town=args.town,
            weather=args.weather,
            min_dist=float(args.min_dist),
            max_dist=float(args.max_dist),
            target_order=args.target_order,
            used_triplets=used_triplets,
        )
        scenarios.append(scen)
        next_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.seed_config_output:
        seed_cfg = _make_seed_config(len(scenarios), int(args.runs_per_scene), int(args.base_seed))
        seed_path = Path(args.seed_config_output)
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(json.dumps(seed_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] wrote scenarios: {out_path} ({len(scenarios)} scenes)")
    if args.seed_config_output:
        print(f"[OK] wrote seed config: {args.seed_config_output}")
    print(f"[Summary] target_order={args.target_order}, town={args.town}, distance=[{args.min_dist}, {args.max_dist}]")


if __name__ == "__main__":
    main()
