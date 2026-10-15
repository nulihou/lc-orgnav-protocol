#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate relation-constrained two-candidate CARLA protocol scenarios.

The historical expanded generator samples by distance only.  This generator is
for supplemental evidence: the target must satisfy the requested relation
relative to the ego/anchor, while the decoy must not satisfy that relation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REL_NAMES = {
    1: "front",
    2: "behind",
    3: "left",
    4: "right",
}


def add_carla_python_api(carla_root: str) -> None:
    root = Path(carla_root)
    pyapi = root / "WindowsNoEditor" / "PythonAPI"
    egg_dir = pyapi / "carla" / "dist"
    if str(pyapi) not in sys.path:
        sys.path.insert(0, str(pyapi))
    if egg_dir.exists():
        for egg in egg_dir.glob("carla-*-py3.7-win-amd64.egg"):
            if str(egg) not in sys.path:
                sys.path.insert(0, str(egg))


def distance(a, b) -> float:
    return math.hypot(a.location.x - b.location.x, a.location.y - b.location.y)


def relative_bearing_deg(anchor_tf, target_tf) -> float:
    dx = target_tf.location.x - anchor_tf.location.x
    dy = target_tf.location.y - anchor_tf.location.y
    angle = math.degrees(math.atan2(dy, dx)) - float(anchor_tf.rotation.yaw)
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def satisfies_relation(anchor_tf, target_tf, rel_type: int) -> bool:
    d = distance(anchor_tf, target_tf)
    theta = relative_bearing_deg(anchor_tf, target_tf)
    if not (5.0 < d < 40.0):
        return False
    if rel_type == 1:
        return abs(theta) < 30.0
    if rel_type == 2:
        return abs(abs(theta) - 180.0) < 30.0
    if rel_type == 3:
        return 60.0 < theta < 120.0
    if rel_type == 4:
        return -120.0 < theta < -60.0
    return False


def build_seed_config(num_scenes: int, runs_per_scene: int, base_seed: int) -> Dict[str, object]:
    configs = []
    for scene_id in range(num_scenes):
        for run_id in range(runs_per_scene):
            seed_id = scene_id * runs_per_scene + run_id
            configs.append(
                {
                    "seed_id": seed_id,
                    "scene_id": scene_id,
                    "run_id": run_id + 1,
                    "carla_seed": base_seed + seed_id * 17,
                    "traffic_manager_seed": base_seed + seed_id * 17 + 1,
                    "python_seed": base_seed + seed_id * 17 + 2,
                    "numpy_seed": base_seed + seed_id * 17 + 3,
                }
            )
    return {"base_seed": base_seed, "num_runs_per_scene": runs_per_scene, "configs": configs}


def parse_relations(raw: str) -> List[int]:
    out: List[int] = []
    aliases = {name: rel for rel, name in REL_NAMES.items()}
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in aliases:
            out.append(aliases[token])
        else:
            out.append(int(token))
    bad = [rel for rel in out if rel not in REL_NAMES]
    if bad:
        raise ValueError(f"Unsupported relation types: {bad}; supported={REL_NAMES}")
    return out


def generate_for_relation(
    spawn_points: Sequence[object],
    rel_type: int,
    scenes_per_relation: int,
    rng: random.Random,
    used_triplets: set,
    start_id: int,
    town: str,
    weather: str,
) -> List[Dict[str, object]]:
    rel_name = REL_NAMES[rel_type]
    anchors = list(range(len(spawn_points)))
    rng.shuffle(anchors)
    scenes: List[Dict[str, object]] = []

    for ego_idx in anchors:
        ego_tf = spawn_points[ego_idx]
        target_pool: List[Tuple[int, float, float]] = []
        decoy_pool: List[Tuple[int, float, float]] = []
        for idx, tf in enumerate(spawn_points):
            if idx == ego_idx:
                continue
            d = distance(ego_tf, tf)
            if not (5.0 < d < 40.0):
                continue
            bearing = relative_bearing_deg(ego_tf, tf)
            if satisfies_relation(ego_tf, tf, rel_type):
                target_pool.append((idx, d, bearing))
            else:
                decoy_pool.append((idx, d, bearing))
        rng.shuffle(target_pool)
        for target_idx, target_dist, target_bearing in target_pool:
            candidates = [item for item in decoy_pool if item[0] != target_idx]
            if not candidates:
                continue
            candidates.sort(key=lambda item: abs(item[1] - target_dist))
            decoy_idx, decoy_dist, decoy_bearing = rng.choice(candidates[: min(10, len(candidates))])
            signature = (ego_idx, target_idx, decoy_idx, rel_type)
            if signature in used_triplets:
                continue
            used_triplets.add(signature)

            scen_id = start_id + len(scenes)
            target_slot = 1 if scen_id % 2 == 0 else 2
            if target_slot == 1:
                node_spawn_indices = [ego_idx, target_idx, decoy_idx]
                slot_dists = [0.0, target_dist, decoy_dist]
                slot_bearings = [0.0, target_bearing, decoy_bearing]
                target_index_gt = 1
            else:
                node_spawn_indices = [ego_idx, decoy_idx, target_idx]
                slot_dists = [0.0, decoy_dist, target_dist]
                slot_bearings = [0.0, decoy_bearing, target_bearing]
                target_index_gt = 2

            scenes.append(
                {
                    "id": scen_id,
                    "town": town,
                    "weather": weather,
                    "scenario_type": f"relation_{rel_name}",
                    "query_type": rel_name,
                    "rel_type": rel_type,
                    "ego_spawn_idx": int(ego_idx),
                    "anchor_spawn_idx": int(ego_idx),
                    "target_spawn_idx": int(target_idx),
                    "target_idx_gt": int(target_index_gt),
                    "target_index_gt": int(target_index_gt),
                    "candidate_spawn_indices": [int(x) for x in node_spawn_indices],
                    "candidate_dists": [round(float(x), 2) for x in slot_dists],
                    "candidate_bearings": [round(float(x), 2) for x in slot_bearings],
                    "num_candidates": 2,
                    "start_dist": round(float(target_dist), 2),
                    "node_spawn_indices": [int(x) for x in node_spawn_indices],
                    "candidate_indices": [1, 2],
                }
            )
            if len(scenes) >= scenes_per_relation:
                return scenes

    raise RuntimeError(f"Could only generate {len(scenes)}/{scenes_per_relation} scenes for relation {rel_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate relation-constrained protocol scenes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-config-output", required=True)
    parser.add_argument("--scenes-per-relation", type=int, default=4)
    parser.add_argument("--runs-per-scene", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=91000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--relations", default="front,behind,left,right")
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", ""))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    add_carla_python_api(args.carla_root)
    import carla  # type: ignore

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(args.town)
    spawn_points = world.get_map().get_spawn_points()
    rng = random.Random(args.base_seed)
    relations = parse_relations(args.relations)

    scenes: List[Dict[str, object]] = []
    used_triplets: set = set()
    for rel_type in relations:
        scenes.extend(
            generate_for_relation(
                spawn_points=spawn_points,
                rel_type=rel_type,
                scenes_per_relation=args.scenes_per_relation,
                rng=rng,
                used_triplets=used_triplets,
                start_id=len(scenes),
                town=args.town,
                weather=args.weather,
            )
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    seed_config = build_seed_config(len(scenes), args.runs_per_scene, args.base_seed)
    Path(args.seed_config_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.seed_config_output).write_text(json.dumps(seed_config, indent=2), encoding="utf-8")

    counts: Dict[str, int] = {}
    target_slots: Dict[int, int] = {}
    for scen in scenes:
        counts[str(scen["query_type"])] = counts.get(str(scen["query_type"]), 0) + 1
        slot = int(scen["target_index_gt"])
        target_slots[slot] = target_slots.get(slot, 0) + 1
    print(f"[OK] wrote {len(scenes)} scenarios to {args.output}")
    print(f"[OK] relation counts: {counts}")
    print(f"[OK] target slots: {target_slots}")
    print(f"[OK] wrote seed config to {args.seed_config_output}")


if __name__ == "__main__":
    main()
