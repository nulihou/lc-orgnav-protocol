#!/usr/bin/env python3
"""Merge scale-300 chunk JSONL files by method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def episode_key(row: dict) -> Tuple[int, int]:
    return int(row.get("scene_id", row.get("scen_id"))), int(row.get("run_id", row.get("seed_index", row.get("episode"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {"methods": {}}
    for method_dir in sorted(p for p in args.chunks_root.iterdir() if p.is_dir()):
        rows: List[dict] = []
        chunk_files = sorted(method_dir.glob("*.jsonl"))
        for path in chunk_files:
            rows.extend(load_jsonl(path))
        rows.sort(key=episode_key)
        seen = set()
        duplicates = []
        for row in rows:
            key = episode_key(row)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        out = args.output_root / f"{method_dir.name}_scale300_300x5.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["methods"][method_dir.name] = {
            "chunks": [str(p) for p in chunk_files],
            "output": str(out),
            "episodes": len(rows),
            "unique_scene_run_keys": len(seen),
            "duplicates": duplicates[:20],
        }
        print(f"[OK] {method_dir.name}: {len(rows)} episodes -> {out}")
    manifest_path = args.output_root / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {manifest_path}")


if __name__ == "__main__":
    main()
