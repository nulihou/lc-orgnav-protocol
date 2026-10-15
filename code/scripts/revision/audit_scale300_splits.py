#!/usr/bin/env python3
"""Audit scale-300 scenario JSON files before CARLA runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_json(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def signature(scene: dict) -> Tuple[object, ...]:
    return (
        scene.get("town"),
        int(scene["ego_spawn_idx"]),
        int(scene["target_spawn_idx"]),
        tuple(int(x) for x in scene["node_spawn_indices"][1:]),
        int(scene["rel_type"]),
        int(scene["num_candidates"]),
    )


def audit_split(name: str, scenes: Sequence[dict]) -> Dict[str, object]:
    errors: List[str] = []
    by_k = Counter()
    by_rel = Counter()
    by_slot = Counter()
    by_k_slot = Counter()
    sigs = set()
    for i, scene in enumerate(scenes):
        sid = int(scene.get("id", i))
        k = int(scene["num_candidates"])
        target = int(scene["target_index_gt"])
        candidates = [int(x) for x in scene["candidate_indices"]]
        nodes = [int(x) for x in scene["node_spawn_indices"]]
        by_k[k] += 1
        by_rel[str(scene.get("query_type", scene.get("rel_type")))] += 1
        by_slot[target] += 1
        by_k_slot[f"K{k}/slot{target}"] += 1
        if sid != i:
            errors.append(f"{name}: scene id is not contiguous at row {i}: id={sid}")
        if len(nodes) != k + 1:
            errors.append(f"{name}:{sid}: node_spawn_indices length {len(nodes)} != K+1={k+1}")
        if candidates != list(range(1, k + 1)):
            errors.append(f"{name}:{sid}: candidate_indices={candidates}, expected 1..{k}")
        if target not in candidates:
            errors.append(f"{name}:{sid}: target_index_gt={target} not in candidates={candidates}")
        if len(set(nodes)) != len(nodes):
            errors.append(f"{name}:{sid}: duplicate spawn indices in node_spawn_indices={nodes}")
        sig = signature(scene)
        if sig in sigs:
            errors.append(f"{name}:{sid}: duplicate split-local spawn signature")
        sigs.add(sig)
    return {
        "n": len(scenes),
        "candidate_counts": dict(sorted(by_k.items())),
        "relation_counts": dict(sorted(by_rel.items())),
        "target_slot_counts": dict(sorted(by_slot.items())),
        "target_slot_counts_by_k": dict(sorted(by_k_slot.items())),
        "unique_signatures": len(sigs),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    all_sigs: Dict[Tuple[object, ...], str] = {}
    cross_errors: List[str] = []
    report: Dict[str, object] = {"splits": {}, "cross_split_errors": cross_errors}
    for split in split_names:
        path = args.scenario_root / f"{split}_scenarios.json"
        scenes = load_json(path)
        report["splits"][split] = audit_split(split, scenes)
        for scene in scenes:
            sig = signature(scene)
            owner = all_sigs.get(sig)
            if owner is not None:
                cross_errors.append(f"signature overlap: {owner} and {split}: {sig}")
            all_sigs[sig] = split

    total_errors = len(cross_errors)
    for split_report in report["splits"].values():  # type: ignore[union-attr]
        total_errors += len(split_report["errors"])  # type: ignore[index]
    report["total_errors"] = total_errors
    report["passed"] = total_errors == 0

    out = args.output or (args.scenario_root / "scenario_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if total_errors:
        raise SystemExit(f"Audit failed with {total_errors} errors")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
