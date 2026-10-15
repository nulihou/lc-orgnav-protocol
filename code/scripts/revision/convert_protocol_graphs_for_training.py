#!/usr/bin/env python3
"""Convert protocol graphs into trainable LC-ORGNav graph samples.

Precomputed protocol graphs use `target_idx_gt` because they are primarily
online-evaluation artifacts. The training dataset expects `target_index` and a
small geometry target. This script writes a mirrored graph directory with both
sets of keys, preserving the protocol fields for auditability.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def geom_from_target(node_feats: np.ndarray, target_index: int) -> np.ndarray:
    if node_feats.ndim >= 3 and node_feats.shape[0] == 1:
        node_feats = node_feats[0]
    d = float(node_feats[target_index, 2])
    theta = math.atan2(float(node_feats[target_index, 4]), float(node_feats[target_index, 3]))
    return np.asarray([d, theta], dtype=np.float32)


def convert_one(src: Path, dst: Path) -> Dict[str, object]:
    data = np.load(src, allow_pickle=False)
    arrays = {key: data[key] for key in data.files}
    target_index = int(arrays.get("target_index", arrays["target_idx_gt"]))
    arrays["target_index"] = np.asarray(target_index, dtype=np.int64)
    arrays["target_idx_gt"] = np.asarray(target_index, dtype=np.int64)
    arrays["geom"] = geom_from_target(arrays["node_feats"], target_index)
    arrays["is_online_canonical"] = np.asarray(True, dtype=np.bool_)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)
    return {
        "source": str(src),
        "output": str(dst),
        "target_index": target_index,
        "num_candidates": int(np.asarray(arrays["candidate_mask"]).astype(bool).sum()),
        "rel_type": int(arrays["rel_type"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--list-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("scen_*.npz"))
    if not paths:
        raise RuntimeError(f"No protocol graphs found in {args.input_dir}")

    records: List[Dict[str, object]] = []
    output_paths: List[str] = []
    for src in paths:
        dst = args.output_dir / src.name
        records.append(convert_one(src, dst))
        output_paths.append(str(dst))

    args.list_output.parent.mkdir(parents=True, exist_ok=True)
    args.list_output.write_text("\n".join(output_paths) + "\n", encoding="utf-8")

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "list_output": str(args.list_output),
        "num_graphs": len(records),
        "records_preview": records[:10],
    }
    summary_path = args.summary_output or (args.output_dir / "convert_protocol_graphs_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {args.list_output}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
