#!/usr/bin/env python3
"""Freeze Oracle-feasible closed-loop scenes into a new test split.

The script is intentionally explicit: it preserves each accepted scene's source
ID, records the Oracle gate rows used for selection, reindexes the frozen split
to 0..N-1, and writes fresh paired seed configs and chunk files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_scenarios(paths: Sequence[Path]) -> Dict[int, Dict[str, Any]]:
    scenes: Dict[int, Dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Scenario file must contain a list: {path}")
        for scene in data:
            sid = int(scene["id"])
            if sid not in scenes:
                scenes[sid] = dict(scene)
    return scenes


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def collect_jsonl_paths(files: Sequence[Path], roots: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for file_path in files:
        if file_path.exists():
            out.append(file_path)
        else:
            raise FileNotFoundError(file_path)
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        out.extend(sorted(root.rglob("*.jsonl")))
    unique: List[Path] = []
    seen = set()
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def row_scene_id(row: Dict[str, Any]) -> Optional[int]:
    if row.get("scene_id") is not None:
        return int(row["scene_id"])
    if row.get("scen_id") is not None:
        return int(row["scen_id"])
    return None


def row_run_id(row: Dict[str, Any]) -> Optional[int]:
    if row.get("run_id") is not None:
        return int(row["run_id"])
    if row.get("seed_index") is not None:
        return int(row["seed_index"])
    return None


def short_gate_row(row: Dict[str, Any], success_field: str) -> Dict[str, Any]:
    keys = [
        "episode",
        "scene_id",
        "scen_id",
        "run_id",
        "seed_index",
        "seed_id",
        "carla_seed",
        "traffic_manager_seed",
        "python_seed",
        "python_random_seed",
        "numpy_seed",
        "torch_seed",
        "valid",
        "success_8m",
        "success_4m",
        "min_dist",
        "final_dist",
        "steps",
        "stuck",
        "collision",
        "lane_invasion",
        "chosen_idx_graph",
        "target_idx_gt",
        "num_candidates",
    ]
    compact = {key: row.get(key) for key in keys if key in row}
    compact["gate_success_field"] = success_field
    compact["gate_success"] = bool(row.get(success_field, False))
    return compact


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


def build_preserved_gate_seed_config(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    configs: List[Dict[str, int]] = []
    for scene in scenarios:
        gate = scene.get("oracle_feasibility_gate", {})
        rows = gate.get("rows", [])
        success_rows = [row for row in rows if row.get("gate_success")]
        if not success_rows:
            raise ValueError(f"No successful gate row available for frozen scene {scene.get('id')}")
        row = success_rows[0]
        python_seed = row.get("python_random_seed", row.get("python_seed", 0))
        configs.append(
            {
                "scene_id": int(scene["id"]),
                "run_id": 1,
                "seed_id": int(row.get("seed_id", 0) or 0),
                "carla_seed": int(row.get("carla_seed", 0) or 0),
                "traffic_manager_seed": int(row.get("traffic_manager_seed", 0) or 0),
                "python_random_seed": int(python_seed or 0),
                "numpy_seed": int(row.get("numpy_seed", 0) or 0),
                "torch_seed": int(row.get("torch_seed", 0) or 0),
                "source_scene_id": int(gate.get("source_id", -1)),
                "source_run_id": int(row.get("run_id", 1) or 1),
            }
        )
    return {
        "base_seed": "preserved_from_oracle_gate",
        "num_runs_per_scene": 1,
        "configs": configs,
    }


def write_chunks(scenarios: Sequence[Dict[str, Any]], chunks_dir: Path, chunk_size: int) -> List[str]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for start in range(0, len(scenarios), chunk_size):
        end = min(len(scenarios), start + chunk_size)
        out = chunks_dir / f"test_scenarios_{start:03d}_{end - 1:03d}.json"
        out.write_text(
            json.dumps(list(scenarios[start:end]), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(str(out))
    chunk_manifest = {
        "chunks_dir": str(chunks_dir),
        "chunk_size": int(chunk_size),
        "num_scenarios": len(scenarios),
        "num_chunks": len(written),
        "chunks": written,
    }
    (chunks_dir / "chunk_manifest.json").write_text(
        json.dumps(chunk_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return written


def relation_name(scene: Dict[str, Any]) -> str:
    if scene.get("query_type") is not None:
        return str(scene["query_type"])
    rel_map = {1: "front", 2: "behind", 3: "left", 4: "right"}
    return rel_map.get(int(scene.get("rel_type", -1)), "unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scenarios", type=Path, nargs="+", required=True)
    parser.add_argument("--oracle-jsonl", type=Path, nargs="*", default=[])
    parser.add_argument("--oracle-jsonl-root", type=Path, nargs="*", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--success-field", default="success_8m")
    parser.add_argument("--min-success-runs", type=int, default=1)
    parser.add_argument("--required-run-ids", default="")
    parser.add_argument("--runs-per-scene", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1520600)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--gate-label", default="")
    parser.add_argument("--preserve-gate-seeds", action="store_true")
    parser.add_argument("--exclude-source-ids", default="")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    source_scenes = load_scenarios(args.source_scenarios)
    exclude_source_ids = {
        int(item.strip())
        for item in args.exclude_source_ids.split(",")
        if item.strip()
    }
    jsonl_paths = collect_jsonl_paths(args.oracle_jsonl, args.oracle_jsonl_root)
    if not jsonl_paths:
        raise RuntimeError("No Oracle JSONL files were provided.")

    required_run_ids = {
        int(item.strip())
        for item in args.required_run_ids.split(",")
        if item.strip()
    }
    rows_by_scene: Dict[int, List[Tuple[Path, Dict[str, Any]]]] = defaultdict(list)
    total_rows = 0
    for path in jsonl_paths:
        for row in iter_jsonl(path):
            sid = row_scene_id(row)
            if sid is None:
                continue
            if sid not in source_scenes:
                continue
            rows_by_scene[sid].append((path, row))
            total_rows += 1

    accepted_source_ids: List[int] = []
    rejected: List[Dict[str, Any]] = []
    gate_rows: Dict[int, List[Dict[str, Any]]] = {}
    for sid in sorted(source_scenes):
        if sid in exclude_source_ids:
            rejected.append({"source_id": sid, "reason": "excluded_by_refinement"})
            continue
        rows = rows_by_scene.get(sid, [])
        if not rows:
            rejected.append({"source_id": sid, "reason": "no_oracle_rows"})
            continue
        if required_run_ids:
            by_run = {row_run_id(row): (path, row) for path, row in rows}
            missing = sorted(rid for rid in required_run_ids if rid not in by_run)
            if missing:
                rejected.append({"source_id": sid, "reason": "missing_required_runs", "missing": missing})
                continue
            candidate_rows = [by_run[rid] for rid in sorted(required_run_ids)]
        else:
            candidate_rows = rows
        successes = [
            (path, row)
            for path, row in candidate_rows
            if row.get("valid", True) is not False and bool(row.get(args.success_field, False))
        ]
        if len(successes) < int(args.min_success_runs):
            rejected.append(
                {
                    "source_id": sid,
                    "reason": "insufficient_oracle_success",
                    "successes": len(successes),
                    "required": int(args.min_success_runs),
                }
            )
            continue
        accepted_source_ids.append(sid)
        gate_rows[sid] = [
            {
                "jsonl": str(path),
                **short_gate_row(row, args.success_field),
            }
            for path, row in candidate_rows
        ]
        if len(accepted_source_ids) >= int(args.target_count):
            break

    if len(accepted_source_ids) < int(args.target_count) and not args.allow_partial:
        raise RuntimeError(
            f"Only {len(accepted_source_ids)} Oracle-feasible scenes found; "
            f"target-count={args.target_count}. Re-run with more candidate scenes or --allow-partial."
        )

    frozen: List[Dict[str, Any]] = []
    for new_id, source_id in enumerate(accepted_source_ids):
        scene = dict(source_scenes[source_id])
        scene["source_id"] = int(source_id)
        if "global_id" in scene:
            scene["source_global_id"] = int(scene["global_id"])
        scene["id"] = int(new_id)
        scene["global_id"] = int(new_id)
        scene["split"] = args.split_name
        scene["oracle_feasibility_gate"] = {
            "label": args.gate_label,
            "source_id": int(source_id),
            "success_field": args.success_field,
            "min_success_runs": int(args.min_success_runs),
            "required_run_ids": sorted(required_run_ids),
            "rows": gate_rows[source_id],
        }
        frozen.append(scene)

    scenario_dir = args.output_root / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = scenario_dir / "test_scenarios.json"
    scenario_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.preserve_gate_seeds:
        if int(args.runs_per_scene) != 1:
            raise ValueError("--preserve-gate-seeds requires --runs-per-scene 1")
        seed_config = build_preserved_gate_seed_config(frozen)
    else:
        seed_config = build_seed_config(
            num_scenes=len(frozen),
            runs_per_scene=int(args.runs_per_scene),
            base_seed=int(args.base_seed),
        )
    seed_path = scenario_dir / "test_seed_config_300x5.json"
    seed_path.write_text(json.dumps(seed_config, indent=2) + "\n", encoding="utf-8")
    chunks = write_chunks(frozen, scenario_dir / "chunks", int(args.chunk_size))

    by_k = Counter(int(scene.get("num_candidates", 0)) for scene in frozen)
    by_rel = Counter(relation_name(scene) for scene in frozen)
    manifest = {
        "output_root": str(args.output_root),
        "source_scenarios": [str(path) for path in args.source_scenarios],
        "oracle_jsonl": [str(path) for path in jsonl_paths],
        "target_count": int(args.target_count),
        "accepted_count": len(frozen),
        "source_scene_count": len(source_scenes),
        "oracle_rows_considered": total_rows,
        "success_field": args.success_field,
        "min_success_runs": int(args.min_success_runs),
        "required_run_ids": sorted(required_run_ids),
        "runs_per_scene": int(args.runs_per_scene),
        "preserve_gate_seeds": bool(args.preserve_gate_seeds),
        "exclude_source_ids": sorted(exclude_source_ids),
        "chunk_size": int(args.chunk_size),
        "gate_label": args.gate_label,
        "candidate_counts": dict(sorted(by_k.items())),
        "relation_counts": dict(sorted(by_rel.items())),
        "accepted_source_ids": accepted_source_ids,
        "rejected_preview": rejected[:200],
        "scenario_json": str(scenario_path),
        "seed_config": str(seed_path),
        "chunks": chunks,
    }
    manifest_path = scenario_dir / "oracle_feasibility_gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if len(frozen) < int(args.target_count) and not args.allow_partial:
        sys.exit(2)


if __name__ == "__main__":
    main()
