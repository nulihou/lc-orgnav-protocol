"""Paired exact tests for protocol-level goal-selection baselines."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = {
    "Ours": "artifacts/revision_20260520/proto_eval/gs_protocol_seed42.jsonl",
    "MLP+pooling": "runs/scaleup_20260520/baselines/candidate_mlp_rerun/candidate_mlp_protocol.jsonl",
    "Query-Transformer": "runs/scaleup_20260520/baselines/query_transformer_cuda_matched_rerun/query_transformer_protocol.jsonl",
    "No-FiLM": "runs/full_rerun_20260520/training/core_ablations/no_film_full/no_film_protocol.jsonl",
    "No-ranking": "artifacts/revision_20260520/proto_eval_ablation/no_rank.jsonl",
    "Random": "artifacts/revision_20260520/proto_eval/random_protocol_seed42.jsonl",
    "Nearest": "artifacts/revision_20260520/proto_eval/heuristic_nearest_protocol.jsonl",
}

DEFAULT_COMPARISONS = [
    ("Ours", "MLP+pooling"),
    ("Ours", "Query-Transformer"),
    ("Ours", "No-FiLM"),
    ("Ours", "Random"),
    ("Ours", "Nearest"),
    ("MLP+pooling", "Random"),
    ("Query-Transformer", "Random"),
    ("No-FiLM", "Random"),
    ("Ours", "No-ranking"),
]


def scen_id_from_row(row: dict[str, Any]) -> int:
    for key in ("scen_id", "scene_id", "item_id"):
        if key in row:
            return int(row[key])
    path = str(row.get("path", ""))
    match = re.search(r"scen_(\d+)", path)
    if match:
        return int(match.group(1))
    raise KeyError(f"Cannot infer scenario id from row: {row}")


def load_rows(path: Path) -> dict[int, bool]:
    rows: dict[int, bool] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[scen_id_from_row(row)] = bool(row.get("gs_correct", False))
    return rows


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cdf)


def compare(name_a: str, rows_a: dict[int, bool], name_b: str, rows_b: dict[int, bool]) -> dict[str, Any]:
    keys = sorted(set(rows_a) & set(rows_b))
    a_success = sum(rows_a[k] for k in keys)
    b_success = sum(rows_b[k] for k in keys)
    a_only = sum(rows_a[k] and not rows_b[k] for k in keys)
    b_only = sum(rows_b[k] and not rows_a[k] for k in keys)
    both_true = sum(rows_a[k] and rows_b[k] for k in keys)
    both_false = sum((not rows_a[k]) and (not rows_b[k]) for k in keys)
    return {
        "method_a": name_a,
        "method_b": name_b,
        "paired_n": len(keys),
        "a_success": a_success,
        "b_success": b_success,
        "a_rate": 100.0 * a_success / len(keys) if keys else 0.0,
        "b_rate": 100.0 * b_success / len(keys) if keys else 0.0,
        "a_true_b_false": a_only,
        "a_false_b_true": b_only,
        "both_true": both_true,
        "both_false": both_false,
        "exact_mcnemar_p": exact_mcnemar_p(a_only, b_only),
    }


def fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.3e}"
    return f"{p:.3f}"


def write_report(results: list[dict[str, Any]], out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Protocol-Level Paired Significance Tests",
        "",
        "Two-sided exact McNemar tests are computed on the same 30 protocol graphs.",
        "",
        "| Comparison | A ProtoAcc@1 | B ProtoAcc@1 | A-only | B-only | Exact p | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        a = f"{item['a_success']}/{item['paired_n']} ({item['a_rate']:.1f}%)"
        b = f"{item['b_success']}/{item['paired_n']} ({item['b_rate']:.1f}%)"
        p = float(item["exact_mcnemar_p"])
        interpretation = "directional/small-sample"
        if p < 0.05:
            interpretation = "significant paired difference"
        elif item["method_b"] == "No-ranking":
            interpretation = "no paired evidence of ranking-loss gain on this small protocol"
        lines.append(
            f"| {item['method_a']} vs {item['method_b']} | {a} | {b} | "
            f"{item['a_true_b_false']} | {item['a_false_b_true']} | {fmt_p(p)} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "Use in R2: these tests support the protocol-level baseline/ablation table without changing the closed-loop evaluation protocol. Because the protocol set has only 30 graphs, non-significant rows should be interpreted conservatively.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-md", type=Path, default=Path("runs/scaleup_20260520/main_30x10/protocol_paired_significance.md"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/scaleup_20260520/main_30x10/protocol_paired_significance.json"))
    args = parser.parse_args()

    rows = {name: load_rows(Path(path)) for name, path in DEFAULT_INPUTS.items()}
    results = [compare(a, rows[a], b, rows[b]) for a, b in DEFAULT_COMPARISONS]
    write_report(results, args.out_md, args.out_json)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
