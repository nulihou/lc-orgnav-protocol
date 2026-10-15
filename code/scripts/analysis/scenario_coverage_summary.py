#!/usr/bin/env python3
"""Summarize scenario coverage and geometric difficulty for revision auditing."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


REL_NAMES = {
    0: "near",
    1: "front",
    2: "behind",
    3: "left",
    4: "right",
}


def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def stats(values: List[float]) -> Dict[str, float]:
    return {
        "min": pct(values, 0),
        "p25": pct(values, 25),
        "median": pct(values, 50),
        "p75": pct(values, 75),
        "max": pct(values, 100),
    }


def angle_deg(x: float, y: float) -> float:
    return float(math.degrees(math.atan2(y, x)))


def angle_gap(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def summarize_protocol(protocol_dir: Path) -> Dict[str, object]:
    rows: List[dict] = []
    for path in sorted(protocol_dir.glob("scen_*.npz")):
        data = np.load(path, allow_pickle=False)
        node_feats = data["node_feats"].astype(float)
        candidate_indices = [int(x) for x in data["candidate_indices"]]
        target_idx = int(data["target_idx_gt"])
        rel_type = int(np.argmax(data["lang_feat"][:5])) if "lang_feat" in data.files else -1
        target_dist = float(node_feats[target_idx, 2])
        target_bearing = angle_deg(float(node_feats[target_idx, 0]), float(node_feats[target_idx, 1]))
        decoys = [idx for idx in candidate_indices if idx != target_idx]
        dist_gap = min(abs(float(node_feats[idx, 2]) - target_dist) for idx in decoys)
        bearing_gap = min(
            angle_gap(angle_deg(float(node_feats[idx, 0]), float(node_feats[idx, 1])), target_bearing)
            for idx in decoys
        )
        rows.append(
            {
                "scenario": path.stem,
                "num_nodes": int(node_feats.shape[0]),
                "num_candidates": len(candidate_indices),
                "target_distance_m": target_dist,
                "min_decoy_distance_gap_m": float(dist_gap),
                "min_decoy_bearing_gap_deg": float(bearing_gap),
                "rel_type": rel_type,
                "rel_name": REL_NAMES.get(rel_type, str(rel_type)),
            }
        )

    return {
        "num_scenarios": len(rows),
        "candidate_count_distribution": dict(Counter(row["num_candidates"] for row in rows)),
        "node_count_distribution": dict(Counter(row["num_nodes"] for row in rows)),
        "relation_distribution": dict(Counter(row["rel_name"] for row in rows)),
        "target_distance_m": stats([row["target_distance_m"] for row in rows]),
        "min_decoy_distance_gap_m": stats([row["min_decoy_distance_gap_m"] for row in rows]),
        "min_decoy_bearing_gap_deg": stats([row["min_decoy_bearing_gap_deg"] for row in rows]),
        "records": rows,
    }


def _base_task_id(row: dict) -> str:
    task_id = str(row.get("task_id", row.get("id", "")))
    return task_id.split("_")[0]


def summarize_queryflip(path: Path) -> Dict[str, object]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    query_counter = Counter()
    candidate_counter = Counter()
    target_dists: List[float] = []
    dist_gaps: List[float] = []
    bearing_gaps: List[float] = []
    for row in rows:
        query_counter[str(row.get("query_type", "unknown"))] += 1
        candidate_counter[int(row["num_candidates"])] += 1
        target_idx = int(row["target_idx_gt"])
        dists = [float(x) for x in row["candidate_dists"]]
        bearings = [float(x) for x in row.get("candidate_bearings", [])]
        target_dists.append(dists[target_idx])
        decoys = [i for i in range(len(dists)) if i != target_idx]
        dist_gaps.append(min(abs(dists[i] - dists[target_idx]) for i in decoys))
        if bearings:
            bearing_gaps.append(min(angle_gap(bearings[i], bearings[target_idx]) for i in decoys))

    return {
        "path": str(path),
        "num_tasks": len(rows),
        "num_base_pairs": len({_base_task_id(row) for row in rows}),
        "query_distribution": dict(query_counter),
        "candidate_count_distribution": dict(candidate_counter),
        "target_distance_m": stats(target_dists),
        "min_decoy_distance_gap_m": stats(dist_gaps),
        "min_decoy_bearing_gap_deg": stats(bearing_gaps),
    }


def fmt_stats(d: Dict[str, float], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"{d['min']:.2f}/{d['median']:.2f}/{d['max']:.2f}{suffix} "
        f"(IQR {d['p25']:.2f}-{d['p75']:.2f}{suffix})"
    )


def write_markdown(summary: Dict[str, object], out_path: Path) -> None:
    protocol = summary["protocol"]
    qf_val = summary["queryflip_val"]
    qf_train = summary["queryflip_train"]

    lines = [
        "# Scenario Coverage and Difficulty Audit",
        "",
        "This audit is descriptive. It records what the current R2 evidence covers and, just as importantly, what it does not claim to cover.",
        "",
        "## Closed-Loop Protocol Graphs",
        "",
        f"- Scenarios: {protocol['num_scenarios']}",
        f"- Candidate count distribution: {protocol['candidate_count_distribution']}",
        f"- Relation/query distribution: {protocol['relation_distribution']}",
        f"- Target distance min/median/max: {fmt_stats(protocol['target_distance_m'], 'm')}",
        f"- Minimum target-vs-decoy distance gap min/median/max: {fmt_stats(protocol['min_decoy_distance_gap_m'], 'm')}",
        f"- Minimum target-vs-decoy bearing gap min/median/max: {fmt_stats(protocol['min_decoy_bearing_gap_deg'], 'deg')}",
        "",
        "Interpretation: the closed-loop CARLA rerun is a controlled target-acquisition protocol, not a broad open-world driving benchmark. It is appropriate for paired controller-bounded semantic target acquisition, while broader query and scene diversity is tested separately.",
        "",
        "## Query-Flip Counterfactual Sets",
        "",
        "| Split | Tasks | Base pairs | Query distribution | Candidate counts | Target distance min/median/max | Distance gap min/median/max | Bearing gap min/median/max |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for label, item in [("val", qf_val), ("train", qf_train)]:
        lines.append(
            f"| {label} | {item['num_tasks']} | {item['num_base_pairs']} | "
            f"{item['query_distribution']} | {item['candidate_count_distribution']} | "
            f"{fmt_stats(item['target_distance_m'], 'm')} | "
            f"{fmt_stats(item['min_decoy_distance_gap_m'], 'm')} | "
            f"{fmt_stats(item['min_decoy_bearing_gap_deg'], 'deg')} |"
        )
    lines.extend(
        [
            "",
            "Recommended manuscript use: cite the closed-loop protocol as 30 two-candidate target-acquisition scenes and cite Query-Flip as the separate same-graph query-dependence audit with tightly matched candidate geometry.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, default=Path("data/protocol_graphs"))
    parser.add_argument("--queryflip-val", type=Path, default=Path("data/semantic_queryflip/scenarios_queryflip_val_group.json"))
    parser.add_argument("--queryflip-train", type=Path, default=Path("data/semantic_queryflip/scenarios_queryflip_train_group.json"))
    parser.add_argument("--out-md", type=Path, default=Path("runs/scaleup_20260520/main_30x10/scenario_coverage_audit.md"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/scaleup_20260520/main_30x10/scenario_coverage_audit.json"))
    args = parser.parse_args()

    summary: Dict[str, object] = {
        "protocol": summarize_protocol(args.protocol_dir),
        "queryflip_val": summarize_queryflip(args.queryflip_val),
        "queryflip_train": summarize_queryflip(args.queryflip_train),
    }
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.out_md)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
