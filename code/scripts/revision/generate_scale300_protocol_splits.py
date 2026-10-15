#!/usr/bin/env python3
"""Generate stratified protocol scenarios for the 300-scene closed-loop revision.

The old large generators mostly created distance-only "near" tasks. Those tasks
are not semantically identifiable when multiple candidates are near the ego. This
generator instead creates relation-identifiable targets: the target satisfies the
requested relation and distractors do not. Target slots are balanced so the test
set cannot be solved by a candidate-index shortcut.

Typical use:

  python scripts/revision/generate_scale300_protocol_splits.py \
    --output-root runs/scale300_20260522/scenarios \
    --splits train:3000,val:800,test:300 \
    --runs-per-test-scene 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


RELATIONS = {
    1: "front",
    2: "behind",
    3: "left",
    4: "right",
}


def add_carla_python_api(carla_root: str) -> None:
    root = Path(carla_root)
    candidates = [
        root / "WindowsNoEditor" / "PythonAPI",
        root / "PythonAPI",
    ]
    for api in candidates:
        if api.exists() and str(api) not in sys.path:
            sys.path.insert(0, str(api))
        dist = api / "carla" / "dist"
        if dist.exists():
            for egg in sorted(dist.glob("carla-*.egg")):
                egg_s = str(egg)
                if egg_s not in sys.path:
                    sys.path.insert(0, egg_s)


def dist_xy(a: Any, b: Any) -> float:
    return math.hypot(a.location.x - b.location.x, a.location.y - b.location.y)


def bearing_deg(anchor_tf: Any, target_tf: Any) -> float:
    dx = target_tf.location.x - anchor_tf.location.x
    dy = target_tf.location.y - anchor_tf.location.y
    angle = math.degrees(math.atan2(dy, dx)) - float(anchor_tf.rotation.yaw)
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def satisfies_relation(anchor_tf: Any, target_tf: Any, rel_type: int) -> bool:
    d = dist_xy(anchor_tf, target_tf)
    theta = bearing_deg(anchor_tf, target_tf)
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
    raise ValueError(f"Unsupported rel_type: {rel_type}")


def parse_splits(raw: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for item in raw.split(","):
        name, n_s = item.split(":")
        name = name.strip()
        n = int(n_s)
        if not name or n <= 0:
            raise ValueError(f"Invalid split specification: {item}")
        out.append((name, n))
    return out


def parse_candidate_mix(raw: str) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for item in raw.split(","):
        k_s, w_s = item.split(":")
        k = int(k_s)
        w = float(w_s)
        if k < 2 or w <= 0:
            raise ValueError(f"Invalid candidate mix entry: {item}")
        out.append((k, w))
    return out


def weighted_schedule(values: Sequence[Tuple[int, float]], n: int, rng: random.Random) -> List[int]:
    total_w = sum(w for _, w in values)
    raw_counts = [(value, n * w / total_w) for value, w in values]
    counts = {value: int(math.floor(c)) for value, c in raw_counts}
    remaining = n - sum(counts.values())
    remainders = sorted(((c - math.floor(c), value) for value, c in raw_counts), reverse=True)
    for _, value in remainders[:remaining]:
        counts[value] += 1
    schedule: List[int] = []
    for value, count in counts.items():
        schedule.extend([value] * count)
    rng.shuffle(schedule)
    return schedule


def even_relation_schedule(n: int, rng: random.Random) -> List[int]:
    rels = list(RELATIONS)
    schedule = [rels[i % len(rels)] for i in range(n)]
    rng.shuffle(schedule)
    return schedule


def build_seed_config(num_scenes: int, runs_per_scene: int, base_seed: int) -> Dict[str, Any]:
    configs: List[Dict[str, int]] = []
    for scene_id in range(num_scenes):
        for run_id in range(1, runs_per_scene + 1):
            base = base_seed + scene_id * 1000 + run_id * 100
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
    return {"base_seed": base_seed, "num_runs_per_scene": runs_per_scene, "configs": configs}


def candidate_pool(
    spawn_points: Sequence[Any],
    ego_idx: int,
    rel_type: int,
) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
    ego_tf = spawn_points[ego_idx]
    positives: List[Tuple[int, float, float]] = []
    negatives: List[Tuple[int, float, float]] = []
    for idx, tf in enumerate(spawn_points):
        if idx == ego_idx:
            continue
        d = dist_xy(ego_tf, tf)
        if not (5.0 < d < 45.0):
            continue
        theta = bearing_deg(ego_tf, tf)
        item = (idx, d, theta)
        if satisfies_relation(ego_tf, tf, rel_type):
            positives.append(item)
        else:
            negatives.append(item)
    return positives, negatives


def choose_decoys(
    negatives: Sequence[Tuple[int, float, float]],
    target: Tuple[int, float, float],
    k: int,
    rng: random.Random,
) -> List[Tuple[int, float, float]]:
    needed = k - 1
    if len(negatives) < needed:
        raise RuntimeError("Not enough relation-negative decoys")
    target_dist = target[1]
    ranked = sorted(negatives, key=lambda item: (abs(item[1] - target_dist), abs(item[2] - target[2])))
    hard_pool = ranked[: max(needed * 4, min(24, len(ranked)))]
    rng.shuffle(hard_pool)
    decoys: List[Tuple[int, float, float]] = []
    used = {target[0]}
    for item in hard_pool + ranked:
        if item[0] in used:
            continue
        used.add(item[0])
        decoys.append(item)
        if len(decoys) == needed:
            return decoys
    raise RuntimeError("Could not choose unique decoys")


def make_scene(
    local_id: int,
    global_id: int,
    split: str,
    spawn_points: Sequence[Any],
    rng: random.Random,
    used_signatures: set,
    town: str,
    weather: str,
    rel_type: int,
    k: int,
    target_slot: int,
) -> Dict[str, Any]:
    ego_indices = list(range(len(spawn_points)))
    rng.shuffle(ego_indices)
    for ego_idx in ego_indices:
        positives, negatives = candidate_pool(spawn_points, ego_idx, rel_type)
        rng.shuffle(positives)
        for target in positives:
            try:
                decoys = choose_decoys(negatives, target, k, rng)
            except RuntimeError:
                continue
            signature = (town, ego_idx, target[0], tuple(sorted(d[0] for d in decoys)), rel_type, k)
            if signature in used_signatures:
                continue
            used_signatures.add(signature)

            candidate_items = [target] + decoys
            target_pos = target_slot - 1
            candidate_items[0], candidate_items[target_pos] = candidate_items[target_pos], candidate_items[0]
            node_spawn_indices = [ego_idx] + [item[0] for item in candidate_items]
            candidate_indices = list(range(1, k + 1))
            candidate_dists = [0.0] + [round(float(item[1]), 2) for item in candidate_items]
            candidate_bearings = [0.0] + [round(float(item[2]), 2) for item in candidate_items]

            return {
                "id": int(local_id),
                "global_id": int(global_id),
                "split": split,
                "town": town,
                "weather": weather,
                "scenario_type": "scale300_relation_identifiable",
                "query_type": RELATIONS[rel_type],
                "rel_type": int(rel_type),
                "num_candidates": int(k),
                "ego_spawn_idx": int(ego_idx),
                "anchor_spawn_idx": int(ego_idx),
                "target_spawn_idx": int(target[0]),
                "target_idx_gt": int(target_slot),
                "target_index_gt": int(target_slot),
                "candidate_spawn_indices": [int(x) for x in node_spawn_indices],
                "candidate_dists": candidate_dists,
                "candidate_bearings": candidate_bearings,
                "start_dist": round(float(target[1]), 2),
                "node_spawn_indices": [int(x) for x in node_spawn_indices],
                "candidate_indices": candidate_indices,
                "target_relation_unique": True,
                "decoy_policy": "relation-negative-hard-distance-matched",
            }
    raise RuntimeError(f"Could not generate scene split={split} rel={rel_type} k={k}")


def summarize(scenarios_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"splits": {}}
    for split, scenes in scenarios_by_split.items():
        by_k = Counter(int(s["num_candidates"]) for s in scenes)
        by_rel = Counter(str(s["query_type"]) for s in scenes)
        by_slot = Counter(int(s["target_index_gt"]) for s in scenes)
        by_k_slot = Counter(f"K{int(s['num_candidates'])}/slot{int(s['target_index_gt'])}" for s in scenes)
        summary["splits"][split] = {
            "n": len(scenes),
            "candidate_counts": dict(sorted(by_k.items())),
            "relation_counts": dict(sorted(by_rel.items())),
            "target_slot_counts": dict(sorted(by_slot.items())),
            "target_slot_counts_by_k": dict(sorted(by_k_slot.items())),
            "unique_spawn_signatures": len(
                {
                    (
                        s["town"],
                        s["ego_spawn_idx"],
                        s["target_spawn_idx"],
                        tuple(s["node_spawn_indices"][1:]),
                        s["rel_type"],
                    )
                    for s in scenes
                }
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train:3000,val:800,test:300")
    parser.add_argument("--candidate-mix", default="2:0.60,3:0.30,5:0.10")
    parser.add_argument("--runs-per-test-scene", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=520600)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", ""))
    args = parser.parse_args()

    add_carla_python_api(str(args.carla_root))
    import carla  # type: ignore

    rng = random.Random(int(args.base_seed))
    splits = parse_splits(args.splits)
    candidate_mix = parse_candidate_mix(args.candidate_mix)

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(30.0)
    world = client.load_world(args.town)
    spawn_points = world.get_map().get_spawn_points()

    args.output_root.mkdir(parents=True, exist_ok=True)
    used_signatures: set = set()
    scenarios_by_split: Dict[str, List[Dict[str, Any]]] = {}
    global_id = 0
    for split, n in splits:
        k_schedule = weighted_schedule(candidate_mix, n, rng)
        rel_schedule = even_relation_schedule(n, rng)
        scenes: List[Dict[str, Any]] = []
        slot_counters: Dict[int, int] = defaultdict(int)
        for local_id in range(n):
            k = k_schedule[local_id]
            target_slot = 1 + (slot_counters[k] % k)
            slot_counters[k] += 1
            scene = make_scene(
                local_id=local_id,
                global_id=global_id,
                split=split,
                spawn_points=spawn_points,
                rng=rng,
                used_signatures=used_signatures,
                town=args.town,
                weather=args.weather,
                rel_type=rel_schedule[local_id],
                k=k,
                target_slot=target_slot,
            )
            scenes.append(scene)
            global_id += 1
        scenarios_by_split[split] = scenes
        out_path = args.output_root / f"{split}_scenarios.json"
        out_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {out_path} ({len(scenes)} scenes)")

    if "test" in scenarios_by_split:
        seed_cfg = build_seed_config(
            num_scenes=len(scenarios_by_split["test"]),
            runs_per_scene=int(args.runs_per_test_scene),
            base_seed=int(args.base_seed) + 900000,
        )
        seed_path = args.output_root / "test_seed_config_300x5.json"
        seed_path.write_text(json.dumps(seed_cfg, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {seed_path}")

    summary = summarize(scenarios_by_split)
    summary.update(
        {
            "generator": Path(__file__).as_posix(),
            "base_seed": int(args.base_seed),
            "town": args.town,
            "weather": args.weather,
            "candidate_mix": args.candidate_mix,
            "relation_policy": "front/behind/left/right only; relation-negative hard decoys",
        }
    )
    summary_path = args.output_root / "scenario_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
