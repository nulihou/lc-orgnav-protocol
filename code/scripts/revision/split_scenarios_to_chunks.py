#!/usr/bin/env python3
"""Split a scenario JSON file into foreground-runner chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-json", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--name-prefix",
        default="test_scenarios",
        help="Output prefix; runner expects test_scenarios by default.",
    )
    args = parser.parse_args()

    scenarios: List[Any] = json.loads(args.scenario_json.read_text(encoding="utf-8"))
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for start in range(0, len(scenarios), int(args.chunk_size)):
        end = min(len(scenarios), start + int(args.chunk_size))
        chunk = scenarios[start:end]
        out = args.chunks_dir / f"{args.name_prefix}_{start:03d}_{end - 1:03d}.json"
        out.write_text(json.dumps(chunk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(str(out))

    manifest = {
        "scenario_json": str(args.scenario_json),
        "chunks_dir": str(args.chunks_dir),
        "chunk_size": int(args.chunk_size),
        "num_scenarios": len(scenarios),
        "num_chunks": len(written),
        "chunks": written,
    }
    manifest_path = args.chunks_dir / "chunk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
