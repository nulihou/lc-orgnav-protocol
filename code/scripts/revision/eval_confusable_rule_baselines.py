#!/usr/bin/env python3
"""Evaluate rule baselines on hard-confusable protocol graphs.

This separates two different claims:

- Coarse Rule-Semantic only sees the queried coarse relation and breaks ties by
  nearest distance, matching the Rule-Semantic comparator already used in R2.
- Fine Rank Rule also sees the query-policy tail bit used by the hard-confusable
  generator. It is a task-specific rule ceiling, not a baseline that the learned
  model must beat.
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

POLICY_NAMES = {
    0: "nearest",
    1: "farthest",
    2: "leftmost",
    3: "rightmost",
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


def target_index(data: np.lib.npyio.NpzFile) -> int:
    if "target_idx_gt" in data.files:
        return int(data["target_idx_gt"])
    return int(data["target_index"])


def rel_type_from_data(data: np.lib.npyio.NpzFile) -> int:
    if "rel_type" in data.files:
        return int(data["rel_type"])
    lang = _squeeze_first(data["lang_feat"])
    if len(lang) >= 5:
        return int(np.argmax(lang[:5]))
    return -1


def query_policy_from_data(data: np.lib.npyio.NpzFile) -> int:
    if "query_policy_id" in data.files:
        return int(data["query_policy_id"])
    if "lang_feat" in data.files:
        lang = _squeeze_first(data["lang_feat"])
        if len(lang) >= 9 and float(np.max(lang[5:9])) > 0.5:
            return int(np.argmax(lang[5:9]))
    return -1


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


def anchor_distance(node_feat: np.ndarray) -> float:
    return float(node_feat[7]) if len(node_feat) > 7 else float(node_feat[2])


def anchor_lateral(node_feat: np.ndarray) -> float:
    d = anchor_distance(node_feat)
    sin_t = float(node_feat[9]) if len(node_feat) > 9 else float(node_feat[4])
    return d * sin_t


def choose_nearest(node_feats: np.ndarray, pool: Sequence[int]) -> int:
    distances = [anchor_distance(node_feats[int(idx)]) for idx in pool]
    return int(pool[int(np.argmin(distances))])


def choose_coarse_rule(data: np.lib.npyio.NpzFile) -> Tuple[int, Dict[str, Any]]:
    node_feats = _squeeze_first(data["node_feats"]).astype(np.float32)
    mask = _squeeze_first(data["candidate_mask"]).astype(bool)
    cand_indices = [int(idx) for idx in np.nonzero(mask)[0]]
    rel_type = rel_type_from_data(data)
    if not cand_indices:
        return -1, {"tie_status": "no_candidates", "relation_match_count": 0}

    relation_matches = [idx for idx in cand_indices if relation_satisfied(node_feats[idx], rel_type)]
    if len(relation_matches) == 1:
        return relation_matches[0], {
            "tie_status": "unique_relation_match",
            "relation_match_count": 1,
            "relation_matches": relation_matches,
        }
    if len(relation_matches) > 1:
        return choose_nearest(node_feats, relation_matches), {
            "tie_status": "multiple_relation_matches_nearest_tiebreak",
            "relation_match_count": len(relation_matches),
            "relation_matches": relation_matches,
        }
    return choose_nearest(node_feats, cand_indices), {
        "tie_status": "no_relation_match_nearest_fallback",
        "relation_match_count": 0,
        "relation_matches": [],
    }


def fine_policy_score(node_feats: np.ndarray, idx: int, policy_id: int) -> float:
    feat = node_feats[int(idx)]
    if policy_id == 0:
        return -anchor_distance(feat)
    if policy_id == 1:
        return anchor_distance(feat)
    if policy_id == 2:
        return anchor_lateral(feat)
    if policy_id == 3:
        return -anchor_lateral(feat)
    return -anchor_distance(feat)


def choose_fine_rank_rule(data: np.lib.npyio.NpzFile) -> Tuple[int, Dict[str, Any]]:
    node_feats = _squeeze_first(data["node_feats"]).astype(np.float32)
    mask = _squeeze_first(data["candidate_mask"]).astype(bool)
    cand_indices = [int(idx) for idx in np.nonzero(mask)[0]]
    rel_type = rel_type_from_data(data)
    policy_id = query_policy_from_data(data)
    if not cand_indices:
        return -1, {"fine_status": "no_candidates", "query_policy_id": policy_id}
    relation_matches = [idx for idx in cand_indices if relation_satisfied(node_feats[idx], rel_type)]
    pool = relation_matches if relation_matches else cand_indices
    chosen = int(max(pool, key=lambda idx: fine_policy_score(node_feats, idx, policy_id)))
    return chosen, {
        "fine_status": "policy_rank_on_relation_matches" if relation_matches else "policy_rank_on_all_candidates",
        "query_policy_id": policy_id,
        "relation_match_count": len(relation_matches),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pct(k: int, n: int) -> str:
    if n <= 0:
        return "0/0"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} ({100.0 * k / n:.1f}%, 95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f})"


def summarize(records: Sequence[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    valid = [r for r in records if r.get("valid", False) and r.get("model") == model_name]
    correct = sum(1 for r in valid if r["correct"])
    by_k: Dict[str, Counter[str]] = defaultdict(Counter)
    by_rel: Dict[str, Counter[str]] = defaultdict(Counter)
    by_policy: Dict[str, Counter[str]] = defaultdict(Counter)
    status = Counter()
    match_count = Counter()
    for row in valid:
        by_k[str(row["num_candidates"])]["n"] += 1
        by_k[str(row["num_candidates"])]["correct"] += int(row["correct"])
        rel_name = REL_NAMES.get(int(row["rel_type"]), str(row["rel_type"]))
        by_rel[rel_name]["n"] += 1
        by_rel[rel_name]["correct"] += int(row["correct"])
        policy_name = POLICY_NAMES.get(int(row.get("query_policy_id", -1)), str(row.get("query_policy_id", -1)))
        by_policy[policy_name]["n"] += 1
        by_policy[policy_name]["correct"] += int(row["correct"])
        status[str(row.get("status", "unknown"))] += 1
        match_count[str(row.get("relation_match_count", -1))] += 1
    return {
        "total": len(valid),
        "correct": correct,
        "accuracy": correct / len(valid) if valid else 0.0,
        "by_k": {k: dict(v) for k, v in sorted(by_k.items(), key=lambda kv: int(kv[0]))},
        "by_relation": {k: dict(v) for k, v in sorted(by_rel.items())},
        "by_query_policy": {k: dict(v) for k, v in sorted(by_policy.items())},
        "status": dict(status),
        "relation_match_count": dict(sorted(match_count.items(), key=lambda kv: int(kv[0]))),
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_markdown(path: Path, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    lines = [
        "# Hard-Confusable Rule Baselines",
        "",
        f"- Input: `{args.input}`",
        f"- Graphs evaluated: {summary['coarse_rule']['total']}",
        "",
        "| Method | ProtoAcc@1 | Interpretation |",
        "|---|---:|---|",
        (
            f"| Coarse Rule-Semantic | {pct(summary['coarse_rule']['correct'], summary['coarse_rule']['total'])} | "
            "Uses only coarse relation and nearest tie-break, matching the existing Rule-Semantic comparator. |"
        ),
        (
            f"| Fine Rank Rule | {pct(summary['fine_rank_rule']['correct'], summary['fine_rank_rule']['total'])} | "
            "Uses the query-policy tail bit; report as a task-specific rule ceiling, not a learned baseline. |"
        ),
        "",
        "## Coarse Rule Ambiguity",
        "",
        "| Status | Episodes |",
        "|---|---:|",
    ]
    for status, n in summary["coarse_rule"]["status"].items():
        lines.append(f"| {status} | {n} |")
    lines.extend(["", "| # relation-matching candidates | Episodes |", "|---:|---:|"])
    for count, n in summary["coarse_rule"]["relation_match_count"].items():
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
            target = target_index(data)
            mask = _squeeze_first(data["candidate_mask"]).astype(bool)
            rel_type = rel_type_from_data(data)
            policy_id = query_policy_from_data(data)

            coarse_chosen, coarse_extra = choose_coarse_rule(data)
            records.append(
                {
                    "item_id": item_id,
                    "path": str(path),
                    "model": "coarse_rule",
                    "valid": True,
                    "target_idx": target,
                    "chosen_idx": coarse_chosen,
                    "correct": bool(coarse_chosen == target),
                    "num_candidates": int(mask.sum()),
                    "rel_type": int(rel_type),
                    "query_policy_id": int(policy_id),
                    "status": coarse_extra.get("tie_status"),
                    "relation_match_count": int(coarse_extra.get("relation_match_count", -1)),
                }
            )

            fine_chosen, fine_extra = choose_fine_rank_rule(data)
            records.append(
                {
                    "item_id": item_id,
                    "path": str(path),
                    "model": "fine_rank_rule",
                    "valid": True,
                    "target_idx": target,
                    "chosen_idx": fine_chosen,
                    "correct": bool(fine_chosen == target),
                    "num_candidates": int(mask.sum()),
                    "rel_type": int(rel_type),
                    "query_policy_id": int(policy_id),
                    "status": fine_extra.get("fine_status"),
                    "relation_match_count": int(fine_extra.get("relation_match_count", -1)),
                }
            )
        except Exception as exc:
            records.append({"item_id": item_id, "path": str(path), "model": "coarse_rule", "valid": False, "error": str(exc), "correct": False})
            records.append({"item_id": item_id, "path": str(path), "model": "fine_rank_rule", "valid": False, "error": str(exc), "correct": False})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(args.input),
        "coarse_rule": summarize(records, "coarse_rule"),
        "fine_rank_rule": summarize(records, "fine_rank_rule"),
    }
    (args.output_dir / "confusable_rule_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "confusable_rule_records.jsonl", records)
    write_markdown(args.output_dir / "confusable_rule_table.md", args, summary)
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {args.output_dir / 'confusable_rule_table.md'}")


if __name__ == "__main__":
    main()
