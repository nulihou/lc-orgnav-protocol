#!/usr/bin/env python3
"""Summarize extra closed-loop runs added for R2 review hardening."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


EpisodeKey = Tuple[int, int]


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_key(row: dict) -> EpisodeKey:
    return int(row.get("scene_id", row.get("scen_id"))), int(row.get("run_id", row.get("seed_index", row.get("episode"))))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def metric_summary(values: Iterable[bool]) -> dict:
    vals = [bool(v) for v in values]
    n = len(vals)
    k = sum(vals)
    lo, hi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "text": f"{k}/{n} ({100*k/n:.1f}%, 95% CI {100*lo:.1f}-{100*hi:.1f})" if n else "0/0",
    }


def is_oracle(method: str) -> bool:
    return method.lower().startswith("oracle")


def selected_correct(method: str, row: dict) -> bool:
    if is_oracle(method):
        return True
    return row.get("chosen_idx_graph") == row.get("target_idx_gt")


def nav(row: dict, radius: int) -> bool:
    return bool(row.get(f"success_{radius}m", False))


def semnav(method: str, row: dict, radius: int) -> bool:
    return selected_correct(method, row) and nav(row, radius)


def valid(row: dict) -> bool:
    return row.get("valid", True) is not False


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def paired_test(
    logs: Dict[str, List[dict]],
    method_a: str,
    method_b: str,
    metric: str,
    fn: Callable[[str, dict], bool],
) -> dict:
    maps = {name: {episode_key(row): row for row in rows} for name, rows in logs.items()}
    keys = sorted(set(maps[method_a]) & set(maps[method_b]))
    a_only = b_only = both = neither = 0
    for key in keys:
        a = fn(method_a, maps[method_a][key])
        b = fn(method_b, maps[method_b][key])
        if a and b:
            both += 1
        elif a:
            a_only += 1
        elif b:
            b_only += 1
        else:
            neither += 1
    return {
        "metric": metric,
        "comparison": f"{method_a} vs {method_b}",
        "n": len(keys),
        "a_rate": (both + a_only) / len(keys) if keys else 0.0,
        "b_rate": (both + b_only) / len(keys) if keys else 0.0,
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "p": exact_mcnemar_p(a_only, b_only),
    }


def summarize_overall(logs: Dict[str, List[dict]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for method, rows in logs.items():
        out[method] = {
            "proto": metric_summary(selected_correct(method, r) for r in rows),
            "nav8": metric_summary(nav(r, 8) for r in rows),
            "nav4": metric_summary(nav(r, 4) for r in rows),
            "sem8": metric_summary(semnav(method, r, 8) for r in rows),
            "sem4": metric_summary(semnav(method, r, 4) for r in rows),
            "valid": metric_summary(valid(r) for r in rows),
        }
    return out


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_p(value: float) -> str:
    return f"{value:.3e}" if value < 1e-3 else f"{value:.3f}"


def write_main_summary(root: Path) -> None:
    out_dir = root / "runs" / "scaleup_20260520" / "main_30x10" / "learnable_baseline_mlp_closed_loop"
    main_dir = root / "runs" / "scaleup_20260520" / "main_30x10"
    logs = {
        "Random": load_jsonl(main_dir / "random_improved_v4_30x10.jsonl"),
        "MLP+pooling": load_jsonl(out_dir / "candidate_mlp_improved_v4_30x10.jsonl"),
        "Ours": load_jsonl(main_dir / "gs_improved_v4_30x10.jsonl"),
        "Oracle selector": load_jsonl(main_dir / "oracle_improved_v4_30x10.jsonl"),
    }
    summary = {"overall": summarize_overall(logs)}
    tests = []
    metric_specs = [
        ("ProtoAcc@1", selected_correct),
        ("SemNavSucc@8m", lambda m, r: semnav(m, r, 8)),
        ("SemNavSucc@4m", lambda m, r: semnav(m, r, 4)),
    ]
    for metric, fn in metric_specs:
        tests.append(paired_test(logs, "Ours", "MLP+pooling", metric, fn))
        tests.append(paired_test(logs, "MLP+pooling", "Random", metric, fn))
    summary["paired_tests"] = tests
    summary["notes"] = [
        "Oracle selector ProtoAcc@1 is reported as 100% by definition; the separate Valid episodes column captures one conservative invalid vehicle-association record.",
        "All invalid or failed episodes remain in the denominator for closed-loop navigation metrics.",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_mlp_closed_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Candidate MLP Closed-Loop Baseline",
        "",
        "Protocol: 30 two-candidate protocol scenes, 10 paired seeds per scene, 300 episodes per method. All methods use the same improved_v4 shared controller.",
        "",
        "## Overall",
        "",
        "| Method | n | ProtoAcc@1 | SemNavSucc@8m | SemNavSucc@4m | NavSucc@8m | Valid episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ["Random", "MLP+pooling", "Ours", "Oracle selector"]:
        row = summary["overall"][method]
        lines.append(
            f"| {method} | {row['proto']['n']} | {row['proto']['text']} | {row['sem8']['text']} | {row['sem4']['text']} | {row['nav8']['text']} | {row['valid']['text']} |"
        )
    lines.extend(["", "## Paired Tests", "", "| Metric | Comparison | Rate A | Rate B | A-only | B-only | Both | Neither | Exact p |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in tests:
        lines.append(
            f"| {row['metric']} | {row['comparison']} | {fmt_pct(row['a_rate'])} | {fmt_pct(row['b_rate'])} | {row['a_only']} | {row['b_only']} | {row['both']} | {row['neither']} | {fmt_p(row['p'])} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    (out_dir / "candidate_mlp_closed_loop_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_relation_summary(root: Path) -> None:
    rel_dir = root / "runs" / "scaleup_20260520" / "supplement_relation_types"
    out_dir = rel_dir / "closed_loop_relation16x3"
    scenarios = json.loads((rel_dir / "relation_types_16.json").read_text(encoding="utf-8"))
    scene_types = {int(s["id"]): str(s.get("query_type", s.get("scenario_type", "unknown"))) for s in scenarios}
    logs = {
        "Random": load_jsonl(out_dir / "random_improved_v4_16x3.jsonl"),
        "Ours (non-gated)": load_jsonl(out_dir / "gs_neural_improved_v4_16x3.jsonl"),
        "Ours-RG": load_jsonl(out_dir / "gs_relation_gated_improved_v4_16x3.jsonl"),
        "Oracle": load_jsonl(out_dir / "oracle_improved_v4_16x3.jsonl"),
    }
    summary: Dict[str, object] = {"overall": summarize_overall(logs)}
    by_relation: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for method, rows in logs.items():
        groups: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            groups[scene_types[int(row.get("scene_id", row.get("scen_id")))]].append(row)
        for rel, group in groups.items():
            by_relation[rel][method] = summarize_overall({method: group})[method]
    summary["by_relation"] = by_relation
    tests = []
    metric_specs = [
        ("ProtoAcc@1", selected_correct),
        ("SemNavSucc@8m", lambda m, r: semnav(m, r, 8)),
        ("SemNavSucc@4m", lambda m, r: semnav(m, r, 4)),
    ]
    for metric, fn in metric_specs:
        tests.append(paired_test(logs, "Ours-RG", "Ours (non-gated)", metric, fn))
        tests.append(paired_test(logs, "Ours (non-gated)", "Random", metric, fn))
    summary["paired_tests"] = tests
    gate_rows = logs["Ours-RG"]
    summary["relation_gate"] = {
        "fallback_count": sum(bool(r.get("relation_gate_fallback", False)) for r in gate_rows),
        "candidate_count_distribution": dict(Counter(len(r.get("relation_gate_candidates", [])) for r in gate_rows)),
    }
    summary["notes"] = [
        "The non-gated row is the unmodified learned selector and should be used to bound claims about relation generalization.",
        "The relation-gated row is a protocol-aided coverage stress test; the gate leaves one candidate in every episode and should not be presented as pure learned multi-candidate selection.",
    ]

    (out_dir / "relation_closed_loop_summary_with_neural.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Relation-Type Closed-Loop Rerun with Non-Gated Control",
        "",
        "Protocol: 16 relation-constrained CARLA scenes (front/behind/left/right, 4 each), 3 paired seeds per scene, 48 episodes per method.",
        "",
        "## Overall",
        "",
        "| Method | n | ProtoAcc@1 | SemNavSucc@8m | SemNavSucc@4m | NavSucc@8m |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ["Random", "Ours (non-gated)", "Ours-RG", "Oracle"]:
        row = summary["overall"][method]  # type: ignore[index]
        lines.append(
            f"| {method} | {row['proto']['n']} | {row['proto']['text']} | {row['sem8']['text']} | {row['sem4']['text']} | {row['nav8']['text']} |"
        )
    lines.extend(["", "## Paired Tests", "", "| Metric | Comparison | Rate A | Rate B | A-only | B-only | Both | Neither | Exact p |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in tests:
        lines.append(
            f"| {row['metric']} | {row['comparison']} | {fmt_pct(row['a_rate'])} | {fmt_pct(row['b_rate'])} | {row['a_only']} | {row['b_only']} | {row['both']} | {row['neither']} | {fmt_p(row['p'])} |"
        )
    gate = summary["relation_gate"]  # type: ignore[assignment]
    lines.extend(["", "## Relation Gate Audit", "", f"- Fallback episodes: {gate['fallback_count']}", f"- Gate candidate-count distribution: {gate['candidate_count_distribution']}", "", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary["notes"])  # type: ignore[index]
    (out_dir / "relation_closed_loop_summary_with_neural.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    write_main_summary(root)
    write_relation_summary(root)
    print("Wrote extra closed-loop summaries")


if __name__ == "__main__":
    main()
