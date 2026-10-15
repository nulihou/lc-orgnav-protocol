#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate true 3-candidate protocol scenarios for CARLA closed-loop supplements.

Each generated item contains one ego node plus three candidate vehicle nodes:
``node_spawn_indices = [ego, cand1, cand2, cand3]`` and
``candidate_indices = [1, 2, 3]``.  The target slot is balanced across the
three candidate positions, and scenes are generated across configurable
distance bins so the supplement is not only a larger copy of the main 2-candidate
near-query protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _add_carla_python_api(carla_root: str) -> None:
    roots = [Path(carla_root), Path(carla_root) / "WindowsNoEditor"]
    for root in roots:
        api = root / "PythonAPI"
        if api.exists() and str(api) not in sys.path:
            sys.path.insert(0, str(api))
        dist = api / "carla" / "dist"
        if dist.exists():
            for egg in sorted(dist.glob("carla-*.egg")):
                egg_s = str(egg)
                if egg_s not in sys.path:
                    sys.path.insert(0, egg_s)


def _dist_xy(a: Any, b: Any) -> float:
    return math.hypot(a.location.x - b.location.x, a.location.y - b.location.y)


def _bearing_gap_deg(ego_tf: Any, a_tf: Any, b_tf: Any) -> float:
    yaw = math.radians(float(ego_tf.rotation.yaw))
    ax = a_tf.location.x - ego_tf.location.x
    ay = a_tf.location.y - ego_tf.location.y
    bx = b_tf.location.x - ego_tf.location.x
    by = b_tf.location.y - ego_tf.location.y
    aa = math.atan2(ay, ax) - yaw
    bb = math.atan2(by, bx) - yaw
    diff = abs((aa - bb + math.pi) % (2 * math.pi) - math.pi)
    return math.degrees(diff)


def _seed_config(num_scenes: int, runs_per_scene: int, base_seed: int) -> Dict[str, Any]:
    configs: List[Dict[str, int]] = []
    for scene_id in range(num_scenes):
        for run_id in range(1, runs_per_scene + 1):
            base = int(base_seed) + scene_id * 1000 + run_id * 100
            configs.append(
                {
                    "scene_id": scene_id,
                    "run_id": run_id,
                    "carla_seed": base + 1,
                    "traffic_manager_seed": base + 2,
                    "python_random_seed": base + 3,
                    "numpy_seed": base + 4,
                    "torch_seed": base + 5,
                }
            )
    return {"base_seed": int(base_seed), "num_runs_per_scene": int(runs_per_scene), "configs": configs}


def _pool_for_bin(
    spawn_points: Sequence[Any],
    ego_idx: int,
    min_dist: float,
    max_dist: float,
) -> List[Tuple[int, float]]:
    ego_tf = spawn_points[ego_idx]
    pool: List[Tuple[int, float]] = []
    for idx, tf in enumerate(spawn_points):
        if idx == ego_idx:
            continue
        d = _dist_xy(ego_tf, tf)
        if min_dist <= d <= max_dist:
            pool.append((idx, d))
    return pool


def _make_scene(
    scen_id: int,
    spawn_points: Sequence[Any],
    rng: random.Random,
    town: str,
    weather: str,
    bin_name: str,
    min_dist: float,
    max_dist: float,
    target_slot: int,
    used: set[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    ego_indices = list(range(len(spawn_points)))
    rng.shuffle(ego_indices)

    for ego_idx in ego_indices:
        pool = _pool_for_bin(spawn_points, ego_idx, min_dist, max_dist)
        if len(pool) < 3:
            continue
        rng.shuffle(pool)

        # Prefer compact but not identical candidate distances so the selector
        # faces three plausible candidates rather than one target plus two trivial decoys.
        for target_idx, target_dist in pool:
            others = [(idx, d) for idx, d in pool if idx != target_idx]
            others.sort(key=lambda item: abs(item[1] - target_dist))
            decoy_candidates = others[: min(12, len(others))]
            decoy_pairs = []
            for i in range(len(decoy_candidates)):
                for j in range(i + 1, len(decoy_candidates)):
                    d1 = decoy_candidates[i]
                    d2 = decoy_candidates[j]
                    if d1[0] == d2[0]:
                        continue
                    # Keep at least a small bearing separation to reduce spawn overlap risk.
                    gap = _bearing_gap_deg(spawn_points[ego_idx], spawn_points[d1[0]], spawn_points[d2[0]])
                    if gap >= 2.0:
                        decoy_pairs.append((d1, d2))
            rng.shuffle(decoy_pairs)

            for (decoy1_idx, _), (decoy2_idx, _) in decoy_pairs:
                key = (ego_idx, target_idx, decoy1_idx, decoy2_idx)
                if key in used:
                    continue
                used.add(key)

                candidate_spawn_nodes = [target_idx, decoy1_idx, decoy2_idx]
                # target_slot is 1-based graph node index in [1, 3].
                target_pos = target_slot - 1
                candidate_spawn_nodes[0], candidate_spawn_nodes[target_pos] = (
                    candidate_spawn_nodes[target_pos],
                    candidate_spawn_nodes[0],
                )
                node_spawn_indices = [ego_idx] + candidate_spawn_nodes
                candidate_indices = [1, 2, 3]
                candidate_spawn_indices = [ego_idx] + candidate_spawn_nodes
                candidate_dists = [
                    0.0,
                    *[
                        round(_dist_xy(spawn_points[ego_idx], spawn_points[sp_idx]), 2)
                        for sp_idx in candidate_spawn_nodes
                    ],
                ]

                return {
                    "id": int(scen_id),
                    "town": town,
                    "weather": weather,
                    "scenario_type": f"three_candidate_{bin_name}",
                    "distance_bin": bin_name,
                    "ego_spawn_idx": int(ego_idx),
                    "anchor_spawn_idx": int(ego_idx),
                    "target_spawn_idx": int(target_idx),
                    "target_idx_gt": int(target_slot),
                    "target_index_gt": int(target_slot),
                    "candidate_spawn_indices": [int(x) for x in candidate_spawn_indices],
                    "candidate_dists": candidate_dists,
                    "num_candidates": 3,
                    "start_dist": round(float(target_dist), 2),
                    "rel_type": 0,
                    "seed": int(scen_id * 1000 + 42),
                    "node_spawn_indices": [int(x) for x in node_spawn_indices],
                    "candidate_indices": candidate_indices,
                }

    raise RuntimeError(f"Could not generate scene for bin={bin_name}, target_slot={target_slot}")


def _parse_bins(raw: str) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    for chunk in raw.split(","):
        name, lo, hi = chunk.split(":")
        out.append((name.strip(), float(lo), float(hi)))
    if not out:
        raise ValueError("At least one distance bin is required")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-config-output", default=None)
    parser.add_argument("--scenes-per-bin", type=int, default=6)
    parser.add_argument("--runs-per-scene", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=73000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--distance-bins", default="near:10:25,mid:25:40,far:40:55")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "E:/carla"))
    args = parser.parse_args()

    _add_carla_python_api(args.carla_root)
    import carla  # type: ignore

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(30.0)
    world = client.load_world(args.town)
    spawn_points = world.get_map().get_spawn_points()

    bins = _parse_bins(args.distance_bins)
    rng = random.Random(int(args.base_seed))
    used: set[Tuple[int, int, int, int]] = set()
    scenarios: List[Dict[str, Any]] = []

    scen_id = 0
    for bin_name, lo, hi in bins:
        for local_idx in range(int(args.scenes_per_bin)):
            target_slot = (local_idx % 3) + 1
            scen = _make_scene(
                scen_id=scen_id,
                spawn_points=spawn_points,
                rng=rng,
                town=args.town,
                weather=args.weather,
                bin_name=bin_name,
                min_dist=lo,
                max_dist=hi,
                target_slot=target_slot,
                used=used,
            )
            scenarios.append(scen)
            scen_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.seed_config_output:
        seed_path = Path(args.seed_config_output)
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            json.dumps(_seed_config(len(scenarios), int(args.runs_per_scene), int(args.base_seed)), indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"[OK] wrote true 3-candidate scenarios: {out_path} ({len(scenarios)} scenes)")
    print(f"[OK] bins: {', '.join(f'{n}=[{lo},{hi}]' for n, lo, hi in bins)}")
    print("[OK] target slots are balanced by construction across candidate indices 1, 2, and 3")
    if args.seed_config_output:
        print(f"[OK] wrote seed config: {args.seed_config_output}")


if __name__ == "__main__":
    main()
