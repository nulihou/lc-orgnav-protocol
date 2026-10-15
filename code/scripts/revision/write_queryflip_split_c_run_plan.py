#!/usr/bin/env python3
"""Write command files for the Query-Flip Split-C held-out family audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List


def cmd_quote(path: Path) -> str:
    return f'"{str(path)}"'


def write(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def header(conda_bat: Path, env_path: Path, carla_root: Path) -> List[str]:
    return [
        "@echo off",
        "setlocal",
        f"call {cmd_quote(conda_bat)} {cmd_quote(env_path)}",
        f"set CARLA_ROOT={carla_root}",
        f"set CARLA_PYTHON_API_PATH={carla_root}\\WindowsNoEditor\\PythonAPI",
        f"set PYTHONPATH={carla_root}\\WindowsNoEditor\\PythonAPI;{carla_root}\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.14-py3.7-win-amd64.egg;%PYTHONPATH%",
        "cd /d %~dp0\\..\\..\\..",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/queryflip_split_c_20260523"))
    parser.add_argument("--conda-bat", type=Path, default=Path(r"python
    parser.add_argument("--env-path", type=Path, default=Path(r".venv
    parser.add_argument("--carla-root", type=Path, default=Path(os.environ.get("CARLA_ROOT", "")))
    parser.add_argument("--carla-exe", type=Path, default=Path(os.environ.get("CARLA_UE4_EXE", "")))
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--base-id-offset", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/submission_v1/paper_artifacts_v1/semantic/queryflip/models/finetune_B/best.pt"),
    )
    args = parser.parse_args()

    cmd_root = args.run_root / "commands"
    data_path = Path("data/semantic_queryflip/scenarios_queryflip_split_c_right.json")
    protocol_dir = Path("data/semantic_queryflip_protocols/protocol_graphs_queryflip_split_c_right")
    result_path = args.run_root / "split_c_right_eval.json"
    noquery_path = args.run_root / "split_c_right_noquery.json"
    queryswap_path = args.run_root / "split_c_right_queryswap.json"
    max_scen_id = args.base_id_offset * 2 + args.num_pairs * 2 - 1

    write(
        cmd_root / "00_start_carla_server.cmd",
        [
            "@echo off",
            "setlocal",
            f"start \"CARLA queryflip split-c\" {cmd_quote(args.carla_exe)} -carla-server -quality-level=Low -ResX=640 -ResY=480 -RenderOffScreen",
            "echo Started CARLA. Wait until port 2000 is ready before running stage 01.",
        ],
    )

    common = header(args.conda_bat, args.env_path, args.carla_root)
    write(
        cmd_root / "01_generate_and_precompute_split_c.cmd",
        common
        + [
            "python experiments\\queryflip_semantic\\scripts\\gen_queryflip_split_c_right.py ^",
            f"  --output {data_path} ^",
            f"  --num-pairs {args.num_pairs} ^",
            "  --avoid-json data\\semantic_queryflip\\scenarios_queryflip_train_group.json ^",
            "  --avoid-json data\\semantic_queryflip\\scenarios_queryflip_val_group.json ^",
            f"  --carla-root {args.carla_root}",
            "",
            "python experiments\\queryflip_semantic\\scripts\\precompute_queryflip_protocols.py ^",
            f"  --scenario_json {data_path} ^",
            f"  --out_dir {protocol_dir}",
        ],
    )

    write(
        cmd_root / "02_eval_split_c.cmd",
        common
        + [
            "python experiments\\queryflip_semantic\\scripts\\eval_query_flip_semcrit.py ^",
            f"  --protocol-dir {protocol_dir} ^",
            f"  --scenarios-json {data_path} ^",
            f"  --checkpoint {args.checkpoint} ^",
            f"  --output {result_path} ^",
            f"  --max-scen-id {max_scen_id} --device {args.device}",
            "",
            "python experiments\\queryflip_semantic\\scripts\\eval_query_flip_semcrit.py ^",
            f"  --protocol-dir {protocol_dir} ^",
            f"  --scenarios-json {data_path} ^",
            f"  --checkpoint {args.checkpoint} ^",
            f"  --output {noquery_path} ^",
            f"  --max-scen-id {max_scen_id} --device {args.device} --no-query",
            "",
            "python experiments\\queryflip_semantic\\scripts\\eval_query_flip_semcrit.py ^",
            f"  --protocol-dir {protocol_dir} ^",
            f"  --scenarios-json {data_path} ^",
            f"  --checkpoint {args.checkpoint} ^",
            f"  --output {queryswap_path} ^",
            f"  --max-scen-id {max_scen_id} --device {args.device} --query-swap",
        ],
    )

    manifest = {
        "run_root": str(args.run_root),
        "commands": sorted(str(p) for p in cmd_root.glob("*.cmd")),
        "split_c_family": "Q_SAME_LANE_FRONT vs Q_RIGHT_ADJ_FRONT",
        "num_pairs": int(args.num_pairs),
        "base_id_offset": int(args.base_id_offset),
        "max_scen_id": int(max_scen_id),
        "device": args.device,
        "scenario_json": str(data_path),
        "protocol_dir": str(protocol_dir),
        "checkpoint": str(args.checkpoint),
        "outputs": [str(result_path), str(noquery_path), str(queryswap_path)],
    }
    write(cmd_root / "README_queryflip_split_c_commands.txt", json.dumps(manifest, indent=2).splitlines())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
