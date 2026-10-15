#!/usr/bin/env python3
"""Generate hard-confusable protocol splits for selector-only stress tests.

Scale-300 intentionally made every target uniquely identifiable by the queried
coarse relation. That construction is useful for closed-loop scale, but it also
lets a deterministic relation rule solve the selector-only task. This generator
creates a complementary stress set in which multiple candidates satisfy the same
coarse relation. The target is made unique by a small query-policy bit encoded in
the tail of the existing 32-d language vector.

The generated JSON is consumed by scripts/online_multi/precompute_protocol_graphs.py
with both --honor-scenario-rel-type and --honor-scenario-lang-feat.
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
from typing import Any, Dict, List, Sequence, Tuple


RELATIONS = {
    1: "front",
    2: "behind",
    3: "left",
    4: "right",
}

POLICIES = {
    0: "nearest",
    1: "farthest",
    2: "leftmost",
    3: "rightmost",
}

POLICY_BY_NAME = {v: k for k, v in POLICIES.items()}


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
        if k < 3 or w <= 0:
            raise ValueError(f"Invalid candidate mix entry: {item}")
        out.append((k, w))
    return out


def parse_policies(raw: str) -> List[int]:
    out: List[int] = []
    for item in raw.split(","):
        name = item.strip().lower()
        if name not in POLICY_BY_NAME:
            raise ValueError(f"Unknown query policy: {item}")
        out.append(POLICY_BY_NAME[name])
    if not out:
        raise ValueError("At least one query policy is required")
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


def even_schedule(values: Sequence[int], n: int, rng: random.Random) -> List[int]:
    schedule = [values[i % len(values)] for i in range(n)]
    rng.shuffle(schedule)
    return schedule


def dist_xy(a: Any, b: Any) -> float:
    return math.hypot(a.location.x - b.location.x, a.location.y - b.location.y)


def relation_geometry(anchor_tf: Any, target_tf: Any) -> Tuple[float, float, float, float]:
    dx = target_tf.location.x - anchor_tf.location.x
    dy = target_tf.location.y - anchor_tf.location.y
    yaw = math.radians(float(anchor_tf.rotation.yaw))
    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)
    left_x = -math.sin(yaw)
    left_y = math.cos(yaw)
    longitudinal = dx * forward_x + dy * forward_y
    lateral = dx * left_x + dy * left_y
    d = math.hypot(dx, dy)
    theta = math.degrees(math.atan2(lateral, longitudinal))
    return d, theta, longitudinal, lateral


def satisfies_relation(anchor_tf: Any, target_tf: Any, rel_type: int) -> bool:
    d, theta, _, _ = relation_geometry(anchor_tf, target_tf)
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


def policy_score(item: Dict[str, float], policy_id: int) -> float:
    if policy_id == 0:
        return -float(item["anchor_d"])
    if policy_id == 1:
        return float(item["anchor_d"])
    if policy_id == 2:
        return float(item["anchor_lateral"])
    if policy_id == 3:
        return -float(item["anchor_lateral"])
    raise ValueError(f"Unsupported policy_id: {policy_id}")


def lang_feat(rel_type: int, policy_id: int, context_anchor: bool) -> List[float]:
    vec = [0.0] * 32
    vec[rel_type] = 1.0
    vec[5 + policy_id] = 1.0
    if context_anchor:
        vec[9] = 1.0
    return vec


def candidate_pool(
    spawn_points: Sequence[Any],
    ego_idx: int,
    anchor_idx: int,
    rel_type: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    ego_tf = spawn_points[ego_idx]
    anchor_tf = spawn_points[anchor_idx]
    positives: List[Dict[str, float]] = []
    negatives: List[Dict[str, float]] = []
    for idx, tf in enumerate(spawn_points):
        if idx in {ego_idx, anchor_idx}:
            continue
        ego_d = dist_xy(ego_tf, tf)
        if not (5.0 < ego_d < 50.0):
            continue
        anchor_d, theta, longitudinal, lateral = relation_geometry(anchor_tf, tf)
        if not (5.0 < anchor_d < 45.0):
            continue
        item = {
            "idx": float(idx),
            "ego_d": float(ego_d),
            "anchor_d": float(anchor_d),
            "anchor_theta": float(theta),
            "anchor_longitudinal": float(longitudinal),
            "anchor_lateral": float(lateral),
        }
        if satisfies_relation(anchor_tf, tf, rel_type):
            positives.append(item)
        else:
            negatives.append(item)
    return positives, negatives


def choose_anchor_indices(spawn_points: Sequence[Any], ego_idx: int, rng: random.Random, context_anchor: bool) -> List[int]:
    if not context_anchor:
        return [ego_idx]
    ego_tf = spawn_points[ego_idx]
    anchors: List[Tuple[int, float]] = []
    for idx, tf in enumerate(spawn_points):
        if idx == ego_idx:
            continue
        d = dist_xy(ego_tf, tf)
        if 8.0 < d < 35.0:
            anchors.append((idx, d))
    anchors.sort(key=lambda item: item[1])
    nearby = anchors[: min(24, len(anchors))]
    rng.shuffle(nearby)
    return [idx for idx, _ in nearby]


def choose_decoys(
    positives: Sequence[Dict[str, float]],
    negatives: Sequence[Dict[str, float]],
    target: Dict[str, float],
    k: int,
    min_same_relation_decoys: int,
    rng: random.Random,
) -> List[Dict[str, float]]:
    used = {int(target["idx"])}
    same_rel = [p for p in positives if int(p["idx"]) not in used]
    same_rel = sorted(
        same_rel,
        key=lambda item: (
            abs(float(item["anchor_d"]) - float(target["anchor_d"])),
            abs(float(item["anchor_lateral"]) - float(target["anchor_lateral"])),
        ),
    )
    decoys: List[Dict[str, float]] = []
    for item in same_rel:
        used.add(int(item["idx"]))
        decoys.append(item)
        if len(decoys) >= min_same_relation_decoys:
            break
    if len(decoys) < min_same_relation_decoys:
        raise RuntimeError("Not enough same-relation decoys")

    needed = k - 1
    if len(decoys) >= needed:
        return decoys[:needed]

    ranked_negatives = sorted(
        negatives,
        key=lambda item: (
            abs(float(item["anchor_d"]) - float(target["anchor_d"])),
            abs(float(item["ego_d"]) - float(target["ego_d"])),
        ),
    )
    hard_pool = ranked_negatives[: max((needed - len(decoys)) * 4, min(24, len(ranked_negatives)))]
    rng.shuffle(hard_pool)
    for item in hard_pool + ranked_negatives:
        idx = int(item["idx"])
        if idx in used:
            continue
        used.add(idx)
        decoys.append(item)
        if len(decoys) == needed:
            return decoys
    raise RuntimeError("Could not choose enough decoys")


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
    policy_id: int,
    k: int,
    target_slot: int,
    context_anchor: bool,
    min_same_relation_decoys: int,
) -> Dict[str, Any]:
    ego_indices = list(range(len(spawn_points)))
    rng.shuffle(ego_indices)
    for ego_idx in ego_indices:
        for anchor_idx in choose_anchor_indices(spawn_points, ego_idx, rng, context_anchor):
            positives, negatives = candidate_pool(spawn_points, ego_idx, anchor_idx, rel_type)
            if len(positives) < min_same_relation_decoys + 1:
                continue
            scored = sorted(positives, key=lambda item: policy_score(item, policy_id), reverse=True)
            target = scored[0]
            try:
                decoys = choose_decoys(
                    positives=positives,
                    negatives=negatives,
                    target=target,
                    k=k,
                    min_same_relation_decoys=min_same_relation_decoys,
                    rng=rng,
                )
            except RuntimeError:
                continue

            signature = (
                town,
                ego_idx,
                anchor_idx,
                int(target["idx"]),
                tuple(sorted(int(d["idx"]) for d in decoys)),
                rel_type,
                policy_id,
                k,
            )
            if signature in used_signatures:
                continue
            used_signatures.add(signature)

            candidate_items = [target] + decoys
            target_pos = target_slot - 1
            candidate_items[0], candidate_items[target_pos] = candidate_items[target_pos], candidate_items[0]

            if context_anchor:
                node_spawn_indices = [ego_idx, anchor_idx] + [int(item["idx"]) for item in candidate_items]
                candidate_start = 2
                anchor_node_idx = 1
            else:
                node_spawn_indices = [ego_idx] + [int(item["idx"]) for item in candidate_items]
                candidate_start = 1
                anchor_node_idx = 0
            candidate_indices = list(range(candidate_start, candidate_start + k))
            target_index_gt = candidate_start + target_pos

            candidate_ego_dists = [0.0] * candidate_start + [round(float(item["ego_d"]), 2) for item in candidate_items]
            candidate_anchor_dists = [0.0] * candidate_start + [round(float(item["anchor_d"]), 2) for item in candidate_items]
            candidate_anchor_bearings = [0.0] * candidate_start + [round(float(item["anchor_theta"]), 2) for item in candidate_items]

            return {
                "id": int(local_id),
                "global_id": int(global_id),
                "split": split,
                "town": town,
                "weather": weather,
                "scenario_type": "hard_confusable_rank_policy",
                "query_type": f"{RELATIONS[rel_type]}_{POLICIES[policy_id]}",
                "rel_type": int(rel_type),
                "query_policy": POLICIES[policy_id],
                "query_policy_id": int(policy_id),
                "lang_feat": lang_feat(rel_type, policy_id, context_anchor),
                "num_candidates": int(k),
                "ego_spawn_idx": int(ego_idx),
                "anchor_spawn_idx": int(anchor_idx),
                "anchor_node_idx": int(anchor_node_idx),
                "target_spawn_idx": int(target["idx"]),
                "target_idx_gt": int(target_index_gt),
                "target_index_gt": int(target_index_gt),
                "target_slot": int(target_slot),
                "candidate_spawn_indices": [int(x) for x in node_spawn_indices],
                "candidate_indices": candidate_indices,
                "candidate_dists": candidate_ego_dists,
                "candidate_anchor_dists": candidate_anchor_dists,
                "candidate_anchor_bearings": candidate_anchor_bearings,
                "start_dist": round(float(target["ego_d"]), 2),
                "node_spawn_indices": [int(x) for x in node_spawn_indices],
                "target_relation_unique": False,
                "same_relation_match_count": int(len(positives)),
                "same_relation_decoys_requested": int(min_same_relation_decoys),
                "decoy_policy": "same-relation-confusable-plus-distance-matched-fill",
                "context_anchor": bool(context_anchor),
            }
    raise RuntimeError(
        f"Could not generate scene split={split} rel={rel_type} policy={POLICIES[policy_id]} k={k}"
    )


def summarize(scenarios_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"splits": {}}
    for split, scenes in scenarios_by_split.items():
        by_k = Counter(int(s["num_candidates"]) for s in scenes)
        by_rel = Counter(str(s["query_type"]).split("_")[0] for s in scenes)
        by_policy = Counter(str(s["query_policy"]) for s in scenes)
        by_slot = Counter(int(s["target_slot"]) for s in scenes)
        by_context = Counter(str(bool(s["context_anchor"])) for s in scenes)
        by_same_rel = Counter(int(s["same_relation_match_count"]) for s in scenes)
        summary["splits"][split] = {
            "n": len(scenes),
            "candidate_counts": dict(sorted(by_k.items())),
            "relation_counts": dict(sorted(by_rel.items())),
            "query_policy_counts": dict(sorted(by_policy.items())),
            "target_slot_counts": dict(sorted(by_slot.items())),
            "context_anchor_counts": dict(sorted(by_context.items())),
            "same_relation_match_count": dict(sorted(by_same_rel.items())),
            "unique_spawn_signatures": len(
                {
                    (
                        s["town"],
                        s["ego_spawn_idx"],
                        s["anchor_spawn_idx"],
                        s["target_spawn_idx"],
                        tuple(s["node_spawn_indices"]),
                        s["rel_type"],
                        s["query_policy_id"],
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
    parser.add_argument("--candidate-mix", default="3:0.45,5:0.40,8:0.15")
    parser.add_argument("--policies", default="farthest,leftmost,rightmost")
    parser.add_argument("--min-same-relation-decoys", type=int, default=1)
    parser.add_argument("--context-anchor", action="store_true")
    parser.add_argument("--base-seed", type=int, default=526600)
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
    policies = parse_policies(args.policies)

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
        rel_schedule = even_schedule(list(RELATIONS), n, rng)
        policy_schedule = even_schedule(policies, n, rng)
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
                policy_id=policy_schedule[local_id],
                k=k,
                target_slot=target_slot,
                context_anchor=bool(args.context_anchor),
                min_same_relation_decoys=int(args.min_same_relation_decoys),
            )
            scenes.append(scene)
            global_id += 1
        scenarios_by_split[split] = scenes
        out_path = args.output_root / f"{split}_scenarios.json"
        out_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {out_path} ({len(scenes)} scenes)")

    summary = summarize(scenarios_by_split)
    summary.update(
        {
            "generator": Path(__file__).as_posix(),
            "base_seed": int(args.base_seed),
            "town": args.town,
            "weather": args.weather,
            "candidate_mix": args.candidate_mix,
            "policies": [POLICIES[p] for p in policies],
            "context_anchor": bool(args.context_anchor),
            "relation_policy": "multiple candidates satisfy the coarse relation; query-policy tail bit disambiguates target",
        }
    )
    summary_path = args.output_root / "scenario_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
