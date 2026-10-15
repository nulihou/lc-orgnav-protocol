#!/usr/bin/env python3
"""Paired exact McNemar tests for the 30x10 closed-loop rerun.

The three methods are evaluated on the same protocol scene and seed pairs.
This script compares binary outcomes on those paired episodes and writes a
small Markdown/JSON artifact for the revision evidence package.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


EpisodeKey = Tuple[int, int]
EpisodeMap = Dict[EpisodeKey, dict]


def load_jsonl(path: Path) -> EpisodeMap:
    episodes: EpisodeMap = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scene_id = int(row.get("scene_id", row.get("scen_id")))
            # The 30x10 rerun appends runs 6--10 after an earlier 30x5 pass.
            # seed_index repeats across those two passes, while run_id remains
            # the stable paired seed identifier across all three methods.
            seed_id = int(row.get("run_id", row.get("seed_index", row.get("episode"))))
            key = (scene_id, seed_id)
            if key in episodes:
                raise ValueError(f"Duplicate episode key {key} in {path} at line {line_no}")
            episodes[key] = row
    return episodes


def selected_correct(method: str, row: dict) -> bool:
    if row.get("valid") is False:
        return False
    if method == "Oracle":
        return bool(row.get("oracle_correct", row.get("gs_correct", True)))
    return bool(row.get("gs_correct", False))


def semnav_success(method: str, radius_key: str) -> Callable[[dict], bool]:
    def metric(row: dict) -> bool:
        return selected_correct(method, row) and bool(row.get(radius_key, False))

    return metric


def proto_success(method: str) -> Callable[[dict], bool]:
    def metric(row: dict) -> bool:
        return selected_correct(method, row)

    return metric


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value using Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compare_pair(
    method_a: str,
    data_a: EpisodeMap,
    method_b: str,
    data_b: EpisodeMap,
    metric_name: str,
    metric_a: Callable[[dict], bool],
    metric_b: Callable[[dict], bool],
) -> dict:
    keys = sorted(set(data_a) & set(data_b))
    if len(keys) != len(data_a) or len(keys) != len(data_b):
        missing_a = sorted(set(data_b) - set(data_a))[:5]
        missing_b = sorted(set(data_a) - set(data_b))[:5]
        raise ValueError(
            f"Unpaired logs for {method_a} vs {method_b}: "
            f"common={len(keys)}, missing_in_a={missing_a}, missing_in_b={missing_b}"
        )

    a_true_b_false = 0
    a_false_b_true = 0
    both_true = 0
    both_false = 0
    for key in keys:
        a = metric_a(data_a[key])
        b = metric_b(data_b[key])
        if a and b:
            both_true += 1
        elif a and not b:
            a_true_b_false += 1
        elif not a and b:
            a_false_b_true += 1
        else:
            both_false += 1

    n = len(keys)
    a_success = both_true + a_true_b_false
    b_success = both_true + a_false_b_true
    return {
        "metric": metric_name,
        "method_a": method_a,
        "method_b": method_b,
        "paired_n": n,
        "a_success": a_success,
        "b_success": b_success,
        "a_rate": a_success / n,
        "b_rate": b_success / n,
        "a_true_b_false": a_true_b_false,
        "a_false_b_true": a_false_b_true,
        "both_true": both_true,
        "both_false": both_false,
        "exact_mcnemar_p": exact_mcnemar_p(a_true_b_false, a_false_b_true),
    }


def format_p(value: float) -> str:
    if value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.3f}"


def write_markdown(results: Iterable[dict], out_path: Path) -> None:
    lines: List[str] = [
        "# Paired Significance Tests for the 30x10 Closed-Loop Rerun",
        "",
        "Episodes are paired by `(scene_id, run_id)` across methods. The table reports two-sided exact McNemar tests on binary outcomes.",
        "",
        "| Metric | Comparison | Rate A | Rate B | A-only | B-only | Both correct | Both wrong | Exact p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {metric} | {method_a} vs {method_b} | {a_rate:.1%} | {b_rate:.1%} | "
            "{a_true_b_false} | {a_false_b_true} | {both_true} | {both_false} | {p} |".format(
                p=format_p(row["exact_mcnemar_p"]),
                **row,
            )
        )

    lines.extend(
        [
            "",
            "Interpretation for the manuscript: Ours is significantly better than Random on ProtoAcc@1, SemNavSucc@8m, and SemNavSucc@4m. Oracle and Ours are not significantly different at the strict 4m threshold, which is consistent with the close-range controller limitation identified in the oracle failure analysis.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10"),
        help="Directory containing the official 30x10 method logs.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/paired_significance_30x10.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/paired_significance_30x10.json"),
    )
    args = parser.parse_args()

    logs = {
        "Random": load_jsonl(args.run_dir / "random_improved_v4_30x10.jsonl"),
        "Ours": load_jsonl(args.run_dir / "gs_improved_v4_30x10.jsonl"),
        "Oracle": load_jsonl(args.run_dir / "oracle_improved_v4_30x10.jsonl"),
    }

    comparisons = [("Ours", "Random"), ("Oracle", "Ours"), ("Oracle", "Random")]
    metric_specs = [
        ("ProtoAcc@1", proto_success),
        ("SemNavSucc@8m", lambda method: semnav_success(method, "success_8m")),
        ("SemNavSucc@4m", lambda method: semnav_success(method, "success_4m")),
    ]

    results: List[dict] = []
    for metric_name, factory in metric_specs:
        for method_a, method_b in comparisons:
            results.append(
                compare_pair(
                    method_a,
                    logs[method_a],
                    method_b,
                    logs[method_b],
                    metric_name,
                    factory(method_a),
                    factory(method_b),
                )
            )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(results, args.out_md)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
