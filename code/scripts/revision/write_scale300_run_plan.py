#!/usr/bin/env python3
r"""Write command files for the scale-300 revision experiment.

The generated commands assume CARLA 0.9.14 is installed at CARLA_ROOT (set via environment) and the
Python 3.7 conda env is .venv They are intentionally
split into stages so failed CARLA chunks can be rerun without overwriting other
chunks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def cmd_quote(path: Path) -> str:
    return f'"{str(path)}"'


def write(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_test_scenarios(scenario_path: Path, chunks_dir: Path, chunk_size: int) -> List[Path]:
    scenarios = json.loads(scenario_path.read_text(encoding="utf-8"))
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for start in range(0, len(scenarios), chunk_size):
        end = min(len(scenarios), start + chunk_size)
        chunk = scenarios[start:end]
        path = chunks_dir / f"test_scenarios_{start:03d}_{end - 1:03d}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out.append(path)
    return out


def activated_header(conda_bat: Path, env_path: Path, carla_root: Path) -> List[str]:
    return [
        "@echo off",
        "setlocal",
        f"call {cmd_quote(conda_bat)} {cmd_quote(env_path)}",
        "if \"%SCALE300_DEVICE%\"==\"\" set SCALE300_DEVICE=cpu",
        f"set CARLA_ROOT={carla_root}",
        f"set CARLA_PYTHON_API_PATH={carla_root}\\WindowsNoEditor\\PythonAPI",
        f"set PYTHONPATH={carla_root}\\WindowsNoEditor\\PythonAPI;{carla_root}\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.14-py3.7-win-amd64.egg;%PYTHONPATH%",
        "cd /d %~dp0\\..\\..\\..",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/scale300_20260522"))
    parser.add_argument("--conda-bat", type=Path, default=Path(r"python
    parser.add_argument("--env-path", type=Path, default=Path(r".venv
    parser.add_argument("--train-env-path", type=Path, default=Path(r".venv
    parser.add_argument("--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", "")))
    parser.add_argument("--carla-exe", type=Path, default=Path(os.environ.get("CARLA_UE4_EXE", "")))
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--runs-per-scene", type=int, default=5)
    parser.add_argument("--controller-profile", default="improved_v4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ours-checkpoint",
        type=Path,
        default=Path("runs/scale300_20260522/training/lcorg_full_relgate_goalrank_seed21/full_best.pt"),
    )
    parser.add_argument(
        "--mlp-checkpoint",
        type=Path,
        default=Path("runs/scale300_20260522/training/candidate_mlp_seed7/candidate_mlp_best.pt"),
    )
    args = parser.parse_args()

    run_root = args.run_root
    scenario_root = run_root / "scenarios"
    protocol_root = run_root / "protocol_graphs"
    trainable_root = run_root / "trainable_graphs"
    lists_root = run_root / "lists"
    logs_root = run_root / "closed_loop_chunks"
    cmd_root = run_root / "commands"
    scenario_root.mkdir(parents=True, exist_ok=True)
    cmd_root.mkdir(parents=True, exist_ok=True)

    header = activated_header(args.conda_bat, args.env_path, args.carla_root)

    write(
        cmd_root / "00_start_carla_server.cmd",
        [
            "@echo off",
            "setlocal",
            f"start \"CARLA scale300\" {cmd_quote(args.carla_exe)} -carla-server -quality-level=Low -ResX=640 -ResY=480 -RenderOffScreen",
            "echo Started CARLA. Wait until port 2000 is ready before running stage 01.",
        ],
    )

    write(
        cmd_root / "01_generate_scenarios.cmd",
        header
        + [
            "python scripts\\revision\\generate_scale300_protocol_splits.py ^",
            f"  --output-root {scenario_root} ^",
            "  --splits train:3000,val:800,test:300 ^",
            "  --candidate-mix 2:0.60,3:0.30,5:0.10 ^",
            f"  --runs-per-test-scene {args.runs_per_scene} ^",
            f"  --carla-root {args.carla_root}",
            "",
            f"python scripts\\revision\\audit_scale300_splits.py --scenario-root {scenario_root}",
        ],
    )

    precompute_lines = header[:]
    for split in ("train", "val", "test"):
        precompute_lines.extend(
            [
                f"python scripts\\online_multi\\precompute_protocol_graphs.py ^",
                f"  --scenario_json {scenario_root / (split + '_scenarios.json')} ^",
                f"  --out_dir {protocol_root / split} ^",
                "  --honor-scenario-rel-type",
                "",
                f"python scripts\\revision\\convert_protocol_graphs_for_training.py ^",
                f"  --input-dir {protocol_root / split} ^",
                f"  --output-dir {trainable_root / split} ^",
                f"  --list-output {lists_root / (split + '_graphs.txt')}",
                "",
            ]
        )
    write(cmd_root / "02_precompute_and_convert_graphs.cmd", precompute_lines)

    write(
        cmd_root / "03_train_selectors.cmd",
        activated_header(args.conda_bat, args.train_env_path, args.carla_root)
        + [
            "set SCALE300_DEVICE=cuda",
            "",
            "echo Training optimized relation-gated graph selectors for seeds 42, 7, and 21",
            "for %%S in (42 7 21) do (",
            "python scripts\\revision\\train_eval_lcorg_ablation.py ^",
            "  --variant full ^",
            f"  --train-list {lists_root / 'train_graphs.txt'} ^",
            f"  --val-list {lists_root / 'val_graphs.txt'} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            f"  --output-dir {run_root / 'training' / 'lcorg_full_relgate_goalrank_seed%%S'} ^",
            "  --epochs 80 --batch-size 64 --device %SCALE300_DEVICE% --max-scen-id 299 ^",
            "  --seed %%S --use-rel-gate --lambda-rel 0 --lambda-geom 0 --lambda-rank 0.2",
            ")",
            "",
            "echo Training query-aware MLP+pooling selectors for seeds 42, 7, and 21",
            "for %%S in (42 7 21) do (",
            "python scripts\\revision\\train_eval_candidate_mlp_baseline.py ^",
            f"  --train-list {lists_root / 'train_graphs.txt'} ^",
            f"  --val-list {lists_root / 'val_graphs.txt'} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            f"  --output-dir {run_root / 'training' / 'candidate_mlp_seed%%S'} ^",
            "  --epochs 80 --batch-size 512 --device %SCALE300_DEVICE% --max-scen-id 299 --seed %%S",
            ")",
            "",
            "echo If SCALE300_DEVICE is empty, rerun with: set SCALE300_DEVICE=cpu or cuda",
        ],
    )

    test_scenario = scenario_root / "test_scenarios.json"
    chunk_paths: List[Path] = []
    if test_scenario.exists():
        chunk_paths = split_test_scenarios(test_scenario, scenario_root / "chunks", int(args.chunk_size))

    if not chunk_paths:
        # Write a placeholder list. Running 01 first and re-running this script
        # will create concrete chunk files.
        chunk_paths = [scenario_root / "chunks" / "test_scenarios_000_029.json"]

    ckpt_ours = args.ours_checkpoint
    ckpt_mlp = args.mlp_checkpoint
    methods: Dict[str, List[str]] = {
        "random": [
            "python scripts\\online_multi\\eval_random_proto_driven.py ^",
            "  --scenarios {scenario_chunk} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            "  --output {output} ^",
            f"  --seed-config {scenario_root / 'test_seed_config_300x5.json'} ^",
            f"  --num-runs-per-scene {args.runs_per_scene} --max-scenes 0 ^",
            f"  --controller-profile {args.controller_profile} --no-rendering",
        ],
        "mlp_pooling": [
            "python scripts\\online_multi\\eval_gs_proto_driven.py ^",
            "  --scenarios {scenario_chunk} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            f"  --checkpoint {ckpt_mlp} ^",
            "  --output {output} ^",
            f"  --seed-config {scenario_root / 'test_seed_config_300x5.json'} ^",
            f"  --num-runs-per-scene {args.runs_per_scene} --max-scenes 0 ^",
            f"  --controller-profile {args.controller_profile} --device %SCALE300_DEVICE% --model-type candidate_mlp --no-rendering",
        ],
        "ours": [
            "python scripts\\online_multi\\eval_gs_proto_driven.py ^",
            "  --scenarios {scenario_chunk} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            f"  --checkpoint {ckpt_ours} ^",
            "  --output {output} ^",
            f"  --seed-config {scenario_root / 'test_seed_config_300x5.json'} ^",
            f"  --num-runs-per-scene {args.runs_per_scene} --max-scenes 0 ^",
            f"  --controller-profile {args.controller_profile} --device %SCALE300_DEVICE% --no-rendering",
        ],
        "rule_semantic": [
            "python scripts\\online_multi\\eval_gs_proto_driven.py ^",
            "  --scenarios {scenario_chunk} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            "  --output {output} ^",
            f"  --seed-config {scenario_root / 'test_seed_config_300x5.json'} ^",
            f"  --num-runs-per-scene {args.runs_per_scene} --max-scenes 0 ^",
            f"  --controller-profile {args.controller_profile} --device %SCALE300_DEVICE% --model-type rule_semantic --selector-profile rule_semantic --no-rendering",
        ],
        "oracle": [
            "python scripts\\online_multi\\eval_oracle_proto_driven.py ^",
            "  --scenarios {scenario_chunk} ^",
            f"  --protocol-dir {protocol_root / 'test'} ^",
            "  --output {output} ^",
            f"  --seed-config {scenario_root / 'test_seed_config_300x5.json'} ^",
            f"  --num-runs-per-scene {args.runs_per_scene} --max-scenes 0 ^",
            f"  --controller-profile {args.controller_profile} --no-rendering",
        ],
    }
    for method, command_template in methods.items():
        lines = header[:]
        for chunk in chunk_paths:
            stem = chunk.stem.replace("test_scenarios_", "")
            out = logs_root / method / f"{method}_{stem}.jsonl"
            lines.append(f"echo Running {method} on {chunk.name}")
            for raw in command_template:
                lines.append(raw.format(scenario_chunk=chunk, output=out))
            lines.append("")
        write(cmd_root / f"04_closed_loop_{method}.cmd", lines)

    write(
        cmd_root / "05_merge_and_summarize.cmd",
        header
        + [
            "python scripts\\revision\\merge_scale300_jsonl.py ^",
            f"  --chunks-root {logs_root} ^",
            f"  --output-root {run_root / 'closed_loop'}",
            "",
            "python scripts\\revision\\summarize_scale300_closed_loop.py ^",
            f"  --run-root {run_root}",
        ],
    )

    manifest = {
        "run_root": str(run_root),
        "commands": sorted(str(p) for p in cmd_root.glob("*.cmd")),
        "chunk_size": int(args.chunk_size),
        "runs_per_scene": int(args.runs_per_scene),
        "controller_profile": args.controller_profile,
        "ours_checkpoint": str(ckpt_ours),
        "mlp_checkpoint": str(ckpt_mlp),
        "notes": [
            "Run 00 once, then 01-03 if the scenarios/checkpoints need regeneration. Run each 04 method command after offline checks pass.",
            "Re-run this script after 01_generate_scenarios.cmd to materialize exact chunk command files.",
            "The default Ours checkpoint is the optimized single-model seed-21 relation-gated selector; the default MLP checkpoint is the strongest seed-7 MLP+pooling comparator.",
            "Rule-Semantic is included as a deterministic hand-coded comparator for the relation-identifiable multi-candidate setting.",
        ],
    }
    write(cmd_root / "README_scale300_commands.txt", json.dumps(manifest, indent=2).splitlines())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
