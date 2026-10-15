#!/usr/bin/env python3
"""Summarize the Query-Flip Split-C leave-relation-family stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    switch_count = sum(
        1
        for row in data.get("results", [])
        if row.get("task_a_chosen") != row.get("task_b_chosen")
    )
    n_pairs = int(data.get("num_scenarios", 0))
    return {
        "pairs": n_pairs,
        "tasks": int(data.get("total_tasks", 0)),
        "correct": int(data.get("correct_tasks", 0)),
        "task_accuracy": float(data.get("accuracy", 0.0)),
        "both_correct": int(data.get("both_correct_count", 0)),
        "both_correct_rate": float(data.get("both_correct_rate", 0.0)),
        "switch_count": switch_count,
        "switch_rate": (100.0 * switch_count / n_pairs) if n_pairs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/queryflip_split_c_20260523"))
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    out_md = args.out_md or (args.run_root / "split_c_summary.md")
    out_json = args.out_json or (args.run_root / "split_c_summary.json")

    rows = {
        "Full": load(args.run_root / "split_c_right_eval.json"),
        "No-Query": load(args.run_root / "split_c_right_noquery.json"),
        "Query-Swap": load(args.run_root / "split_c_right_queryswap.json"),
    }
    summary = {
        "split": "Split-C",
        "family": "Q_SAME_LANE_FRONT vs Q_RIGHT_ADJ_FRONT",
        "interpretation": (
            "Leave-relation-family stress test. The current Query-Flip checkpoint "
            "does not support an unseen relation-pair transfer claim."
        ),
        "rows": rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Query-Flip Split-C Stress Test",
        "",
        "Family: `Q_SAME_LANE_FRONT` vs `Q_RIGHT_ADJ_FRONT`.",
        "",
        "| Setting | Pairs | Tasks | Correct tasks | Task acc. | Both-correct | Switch rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['pairs']} | {row['tasks']} | {row['correct']}/{row['tasks']} "
            f"| {row['task_accuracy']:.1f}% | {row['both_correct']}/{row['pairs']} "
            f"({row['both_correct_rate']:.1f}%) | {row['switch_count']}/{row['pairs']} "
            f"({row['switch_rate']:.1f}%) |"
        )
    lines += [
        "",
        "Interpretation: this is a leave-relation-family stress test, not a positive paper claim. "
        "The current Query-Flip fine-tuned checkpoint performs poorly on the unseen right-adjacent "
        "relation-pair family, so the manuscript should not claim unseen relation-family transfer.",
        "",
        "Primary files:",
        f"- Full: `{args.run_root / 'split_c_right_eval.json'}`",
        f"- No-Query: `{args.run_root / 'split_c_right_noquery.json'}`",
        f"- Query-Swap: `{args.run_root / 'split_c_right_queryswap.json'}`",
        f"- Scenarios: `data/semantic_queryflip/scenarios_queryflip_split_c_right.json`",
        f"- Protocol graphs: `data/semantic_queryflip_protocols/protocol_graphs_queryflip_split_c_right/`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
