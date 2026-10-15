#!/usr/bin/env python3
"""Summarize hard-confusable selector-only benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pct(correct: int, total: int) -> str:
    if total <= 0:
        return "--"
    lo, hi = wilson_ci(correct, total)
    return f"{correct}/{total} ({100.0 * correct / total:.1f}%, 95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f})"


def extract_seed(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else name


def run_sort_key(run: Dict[str, Any]) -> Tuple[int, str]:
    seed = extract_seed(str(run["name"]))
    return (int(seed) if seed.isdigit() else 10**9, str(run["name"]))


def dedupe_runs_by_seed(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        seed = extract_seed(str(run["name"]))
        previous = by_seed.get(seed)
        if previous is None:
            by_seed[seed] = run
            continue
        current_is_complete = str(run["name"]).endswith("_complete")
        previous_is_complete = str(previous["name"]).endswith("_complete")
        if current_is_complete and not previous_is_complete:
            by_seed[seed] = run
    return sorted(by_seed.values(), key=run_sort_key)


def load_model_runs(root: Path, glob_pattern: str, summary_name: str, records_name: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for directory in sorted(root.glob(glob_pattern)):
        summary_path = directory / summary_name
        records_path = directory / records_name
        if not summary_path.exists() or not records_path.exists():
            continue
        summary = read_json(summary_path)
        metric = summary.get("protocol", summary)
        raw_acc = float(metric["accuracy"])
        acc_fraction = raw_acc / 100.0 if raw_acc > 1.0 else raw_acc
        records = read_jsonl(records_path)
        runs.append(
            {
                "name": directory.name,
                "summary": summary,
                "records": records,
                "correct": int(metric["correct"]),
                "total": int(metric["total"]),
                "accuracy": acc_fraction,
            }
        )
    return runs


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial with p=0.5
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_compare(a_records: Sequence[Dict[str, Any]], b_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id_a = {int(r.get("scen_id", r.get("item_id", -1))): bool(r.get("gs_correct", r.get("correct", False))) for r in a_records}
    by_id_b = {int(r.get("scen_id", r.get("item_id", -1))): bool(r.get("gs_correct", r.get("correct", False))) for r in b_records}
    ids = sorted(set(by_id_a) & set(by_id_b))
    a_only = sum(1 for idx in ids if by_id_a[idx] and not by_id_b[idx])
    b_only = sum(1 for idx in ids if by_id_b[idx] and not by_id_a[idx])
    both = sum(1 for idx in ids if by_id_a[idx] and by_id_b[idx])
    neither = sum(1 for idx in ids if not by_id_a[idx] and not by_id_b[idx])
    return {
        "n": len(ids),
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "mcnemar_p": exact_mcnemar_p(a_only, b_only),
    }


def run_mean_line(name: str, runs: Sequence[Dict[str, Any]]) -> str:
    if not runs:
        return f"| {name} | -- | -- | -- |"
    accs = [100.0 * float(r["accuracy"]) for r in runs]
    cells = [pct(int(r["correct"]), int(r["total"])) for r in runs]
    return f"| {name} | {' / '.join(cells)} | {mean(accs):.1f}% +/- {pstdev(accs):.1f}% | {len(runs)} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/hard_confusable_20260526"))
    parser.add_argument("--output", type=Path, default=Path("runs/hard_confusable_20260526/hard_confusable_selector_summary.md"))
    args = parser.parse_args()

    training = args.root / "training"
    graph_runs = load_model_runs(
        training,
        "lcorg_full_relgate_rankquery_seed*",
        "full_protocol_summary.json",
        "full_protocol.jsonl",
    )
    graph_runs = dedupe_runs_by_seed(graph_runs)
    mlp_runs = load_model_runs(
        training,
        "candidate_mlp_rankquery_seed*_cpu",
        "candidate_mlp_summary.json",
        "candidate_mlp_protocol.jsonl",
    )
    mlp_runs = dedupe_runs_by_seed(mlp_runs)
    rule_summary_path = training / "rule_baselines" / "confusable_rule_summary.json"
    rule_summary = read_json(rule_summary_path) if rule_summary_path.exists() else {}

    lines = [
        "# Hard-Confusable Selector Summary",
        "",
        f"- Root: `{args.root}`",
        f"- Graph runs found: {len(graph_runs)}",
        f"- MLP+pooling runs found: {len(mlp_runs)}",
        "",
        "| Method | Per-seed ProtoAcc@1 | Mean +/- std | Seeds |",
        "|---|---:|---:|---:|",
        run_mean_line("Ours graph+gate", graph_runs),
        run_mean_line("MLP+pooling", mlp_runs),
    ]
    if rule_summary:
        coarse = rule_summary["coarse_rule"]
        fine = rule_summary["fine_rank_rule"]
        lines.append(f"| Coarse Rule-Semantic | {pct(int(coarse['correct']), int(coarse['total']))} | -- | deterministic |")
        lines.append(f"| Fine Rank Rule ceiling | {pct(int(fine['correct']), int(fine['total']))} | -- | deterministic |")

    lines.extend(["", "## Paired Comparisons", "", "| Comparison | N | A-only | B-only | Both | Neither | Exact McNemar p |", "|---|---:|---:|---:|---:|---:|---:|"])

    if graph_runs and mlp_runs:
        best_graph = max(graph_runs, key=lambda r: r["accuracy"])
        best_mlp = max(mlp_runs, key=lambda r: r["accuracy"])
        cmp_best = paired_compare(best_graph["records"], best_mlp["records"])
        lines.append(
            f"| Best graph ({best_graph['name']}) vs best MLP ({best_mlp['name']}) | "
            f"{cmp_best['n']} | {cmp_best['a_only']} | {cmp_best['b_only']} | "
            f"{cmp_best['both']} | {cmp_best['neither']} | {cmp_best['mcnemar_p']:.3g} |"
        )

        # Also compare same seed names when both are present.
        by_seed_graph = {extract_seed(str(r["name"])): r for r in graph_runs}
        by_seed_mlp = {extract_seed(str(r["name"])): r for r in mlp_runs}
        for seed in sorted(set(by_seed_graph) & set(by_seed_mlp), key=int):
            g = by_seed_graph[seed]
            m = by_seed_mlp[seed]
            cmp_seed = paired_compare(g["records"], m["records"])
            lines.append(
                f"| Graph seed {seed} vs MLP seed {seed} | "
                f"{cmp_seed['n']} | {cmp_seed['a_only']} | {cmp_seed['b_only']} | "
                f"{cmp_seed['both']} | {cmp_seed['neither']} | {cmp_seed['mcnemar_p']:.3g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Use this benchmark only as selector-only hard-confusable evidence. Coarse Rule-Semantic is expected to fail because the coarse relation is intentionally ambiguous. Fine Rank Rule is a task-specific rule ceiling because it reads the query-policy bit directly.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
