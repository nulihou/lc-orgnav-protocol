#!/usr/bin/env python3
"""Offset scenario ids and regenerate a matching seed config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_seed_config(scene_ids: List[int], runs_per_scene: int, base_seed: int) -> Dict[str, Any]:
    configs: List[Dict[str, int]] = []
    for ordinal, scene_id in enumerate(scene_ids):
        for run_id in range(1, runs_per_scene + 1):
            base = base_seed + ordinal * 1000 + run_id * 100
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
    return {"base_seed": int(base_seed), "num_runs_per_scene": int(runs_per_scene), "configs": configs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-offset", type=int, required=True)
    parser.add_argument("--seed-config-output", type=Path, default=None)
    parser.add_argument("--runs-per-scene", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=2520600)
    args = parser.parse_args()

    scenarios = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list):
        raise ValueError("--input must contain a list of scenarios")

    out_scenarios: List[Dict[str, Any]] = []
    for scene in scenarios:
        item = dict(scene)
        old_id = int(item["id"])
        item["source_local_id_before_offset"] = old_id
        item["id"] = old_id + int(args.id_offset)
        if "global_id" in item:
            item["global_id_before_offset"] = int(item["global_id"])
        item["global_id"] = int(item["id"])
        out_scenarios.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "id_offset": int(args.id_offset),
        "num_scenarios": len(out_scenarios),
        "first_id": out_scenarios[0]["id"] if out_scenarios else None,
        "last_id": out_scenarios[-1]["id"] if out_scenarios else None,
    }
    manifest_path = args.output.with_suffix(".offset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.seed_config_output is not None:
        cfg = build_seed_config(
            [int(scene["id"]) for scene in out_scenarios],
            runs_per_scene=int(args.runs_per_scene),
            base_seed=int(args.base_seed),
        )
        args.seed_config_output.parent.mkdir(parents=True, exist_ok=True)
        args.seed_config_output.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        manifest["seed_config_output"] = str(args.seed_config_output)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
