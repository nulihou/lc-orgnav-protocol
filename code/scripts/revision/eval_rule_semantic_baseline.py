#!/usr/bin/env python3
"""Evaluate a deterministic relation-rule semantic selector on NPZ protocol graphs.

The rule uses the same compact relation definitions as the protocol generator:
near/front/behind/left/right relative to the anchor/ego frame.  It is intentionally
reported as a strong hand-coded comparator, not as a learned selector.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


REL_NAMES = {
    0: "near",
    1: "front",
    2: "behind",
    3: "left",
    4: "right",
}


def _squeeze_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def read_paths(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("scen_*.npz"))
    with path.open("r", encoding="utf-8-sig") as f:
        return [Path(line.strip()) for line in f if line.strip() and not line.lstrip().startswith("#")]


def relation_satisfied(node_feat: np.ndarray, rel_type: int) -> bool:
    d = float(node_feat[7]) if len(node_feat) > 7 else float(node_feat[2])
    cos_t = float(node_feat[8]) if len(node_feat) > 8 else float(node_feat[3])
    sin_t = float(node_feat[9]) if len(node_feat) > 9 else float(node_feat[4])
    theta = math.atan2(sin_t, cos_t)
    if rel_type == 0:
        return d < 50.0
    if rel_type == 1:
        return (5.0 < d < 40.0) and abs(theta) < math.radians(30.0)
    if rel_type == 2:
        return (5.0 < d < 40.0) and abs(abs(theta) - math.pi) < math.radians(30.0)
    if rel_type == 3:
        return (5.0 < d < 40.0) and (math.radians(60.0) < theta < math.radians(120.0))
    if rel_type == 4:
        return (5.0 < d < 40.0) and (-math.radians(120.0) < theta < -math.radians(60.0))
    return False


def target_index(data: np.lib.npyio.NpzFile) -> int:
    if "target_idx_gt" in data.files:
        return int(data["target_idx_gt"])
    return int(data["target_index"])


def rel_type_from_data(data: np.lib.npyio.NpzFile) -> int:
    if "rel_type" in data.files:
        return int(data["rel_type"])
    if "lang_feat" in data.files:
        lang = _squeeze_first(data["lang_feat"])
        if len(lang) >= 5:
            return int(np.argmax(lang[:5]))
    return -1


def choose_rule(data: np.lib.npyio.NpzFile) -> Tuple[int, Dict[str, Any]]:
    node_feats = _squeeze_first(data["node_feats"]).astype(np.float32)
    mask = _squeeze_first(data["candidate_mask"]).astype(bool)
    cand_indices = np.nonzero(mask)[0]
    rel_type = rel_type_from_data(data)
    if len(cand_indices) == 0:
        return -1, {"tie_status": "no_candidates", "relation_match_count": 0}

    relation_matches = [int(idx) for idx in cand_indices if relation_satisfied(node_feats[int(idx)], rel_type)]
    if len(relation_matches) == 1:
        return relation_matches[0], {
            "tie_status": "unique_relation_match",
            "relation_match_count": 1,
            "relation_matches": relation_matches,
        }

    if len(relation_matches) > 1:
        pool = relation_matches
        tie_status = "multiple_relation_matches_nearest_tiebreak"
    else:
        pool = [int(idx) for idx in cand_indices]
        tie_status = "no_relation_match_nearest_fallback"

    distances = node_feats[pool, 7] if node_feats.shape[1] > 7 else node_feats[pool, 2]
    chosen = int(pool[int(np.argmin(distances))])
    return chosen, {
        "tie_status": tie_status,
        "relation_match_count": len(relation_matches),
        "relation_matches": relation_matches,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pct(k: int, n: int) -> str:
    if n <= 0:
        return "0/0"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} ({100.0 * k / n:.1f}%, 95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f})"


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if r.get("valid", False)]
    correct = sum(1 for r in valid if r["correct"])
    by_k: Dict[str, Counter[str]] = defaultdict(Counter)
    by_rel: Dict[str, Counter[str]] = defaultdict(Counter)
    tie_status = Counter()
    match_count = Counter()
    for row in valid:
        by_k[str(row["num_candidates"])]["n"] += 1
        by_k[str(row["num_candidates"])]["correct"] += int(row["correct"])
        rel_name = REL_NAMES.get(int(row["rel_type"]), str(row["rel_type"]))
        by_rel[rel_name]["n"] += 1
        by_rel[rel_name]["correct"] += int(row["correct"])
        tie_status[str(row["tie_status"])] += 1
        match_count[str(row["relation_match_count"])] += 1
    return {
        "total": len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else 0.0,
        "by_k": {k: dict(v) for k, v in sorted(by_k.items(), key=lambda kv: int(kv[0]))},
        "by_relation": {k: dict(v) for k, v in sorted(by_rel.items())},
        "tie_status": dict(tie_status),
        "relation_match_count": dict(sorted(match_count.items(), key=lambda kv: int(kv[0]))),
    }


def write_markdown(path: Path, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    lines = [
        "# Rule-Semantic Baseline",
        "",
        f"- Input: `{args.input}`",
        f"- Graphs evaluated: {summary['total']}",
        f"- Overall ProtoAcc@1: {pct(int(summary['correct']), int(summary['total']))}",
        "",
        "This is a deterministic hand-coded relation selector. It first keeps candidates satisfying the queried relation; if the relation is ambiguous it breaks ties by nearest anchor-relative distance; if no candidate satisfies the relation it falls back to nearest.",
        "",
        "## By Candidate Count",
        "",
        "| K | ProtoAcc@1 |",
        "|---:|---:|",
    ]
    for k, row in summary["by_k"].items():
        lines.append(f"| {k} | {pct(int(row['correct']), int(row['n']))} |")
    lines.extend(["", "## By Relation", "", "| Relation | ProtoAcc@1 |", "|---|---:|"])
    for rel, row in summary["by_relation"].items():
        lines.append(f"| {rel} | {pct(int(row['correct']), int(row['n']))} |")
    lines.extend(["", "## Rule Ambiguity Diagnostics", "", "| Status | Episodes |", "|---|---:|"])
    for status, n in summary["tie_status"].items():
        lines.append(f"| {status} | {n} |")
    lines.extend(["", "| # relation-matching candidates | Episodes |", "|---:|---:|"])
    for count, n in summary["relation_match_count"].items():
        lines.append(f"| {count} | {n} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="NPZ directory or text file containing NPZ paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = read_paths(args.input)
    records: List[Dict[str, Any]] = []
    for item_id, path in enumerate(paths):
        try:
            data = np.load(path, allow_pickle=False)
            chosen, extra = choose_rule(data)
            target = target_index(data)
            mask = _squeeze_first(data["candidate_mask"]).astype(bool)
            rel_type = rel_type_from_data(data)
            records.append(
                {
                    "item_id": item_id,
                    "path": str(path),
                    "valid": True,
                    "target_idx": target,
                    "chosen_idx": chosen,
                    "correct": bool(chosen == target),
                    "num_candidates": int(mask.sum()),
                    "rel_type": int(rel_type),
                    **extra,
                }
            )
        except Exception as exc:
            records.append({"item_id": item_id, "path": str(path), "valid": False, "error": str(exc), "correct": False})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    (args.output_dir / "rule_semantic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "rule_semantic_records.jsonl", records)
    write_markdown(args.output_dir / "rule_semantic_table.md", args, summary)
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {args.output_dir / 'rule_semantic_table.md'}")


if __name__ == "__main__":
    main()
