#!/usr/bin/env python3
"""Summarize supplemental relation-type CARLA closed-loop reruns."""

from __future__ import annotations

import argparse
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


def selected_correct(method: str, row: dict) -> bool:
    if row.get("valid") is False:
        return False
    if method == "Oracle":
        return bool(row.get("oracle_correct", row.get("gs_correct", True)))
    return bool(row.get("gs_correct", False))


def semnav(method: str, row: dict, radius: int) -> bool:
    return selected_correct(method, row) and bool(row.get(f"success_{radius}m", False))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
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


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def pair_maps(logs: Dict[str, List[dict]]) -> Dict[str, Dict[EpisodeKey, dict]]:
    out: Dict[str, Dict[EpisodeKey, dict]] = {}
    for name, rows in logs.items():
        mp: Dict[EpisodeKey, dict] = {}
        for row in rows:
            key = episode_key(row)
            if key in mp:
                raise ValueError(f"duplicate episode key {key} for {name}")
            mp[key] = row
        out[name] = mp
    return out


def paired_test(
    maps: Dict[str, Dict[EpisodeKey, dict]],
    method_a: str,
    method_b: str,
    metric_name: str,
    fn: Callable[[str, dict], bool],
) -> dict:
    keys = sorted(set(maps[method_a]) & set(maps[method_b]))
    if len(keys) != len(maps[method_a]) or len(keys) != len(maps[method_b]):
        raise ValueError(f"unpaired logs for {method_a} vs {method_b}: common={len(keys)}")
    a_only = b_only = both = neither = 0
    for key in keys:
        a = fn(method_a, maps[method_a][key])
        b = fn(method_b, maps[method_b][key])
        if a and b:
            both += 1
        elif a and not b:
            a_only += 1
        elif b and not a:
            b_only += 1
        else:
            neither += 1
    n = len(keys)
    return {
        "metric": metric_name,
        "comparison": f"{method_a} vs {method_b}",
        "n": n,
        "a_rate": (both + a_only) / n,
        "b_rate": (both + b_only) / n,
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "p": exact_mcnemar_p(a_only, b_only),
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_p(value: float) -> str:
    return f"{value:.3e}" if value < 1e-3 else f"{value:.3f}"


def summarize(logs: Dict[str, List[dict]], scene_types: Dict[int, str]) -> dict:
    overall: Dict[str, dict] = {}
    by_relation: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for method, rows in logs.items():
        overall[method] = {
            "proto": metric_summary(selected_correct(method, r) for r in rows),
            "sem8": metric_summary(semnav(method, r, 8) for r in rows),
            "sem4": metric_summary(semnav(method, r, 4) for r in rows),
            "valid": metric_summary(bool(r.get("valid", True)) for r in rows),
        }
        rel_rows: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            rel_rows[scene_types[int(row.get("scen_id", row.get("scene_id")))]].append(row)
        for rel, group in sorted(rel_rows.items()):
            by_relation[rel][method] = {
                "proto": metric_summary(selected_correct(method, r) for r in group),
                "sem8": metric_summary(semnav(method, r, 8) for r in group),
                "sem4": metric_summary(semnav(method, r, 4) for r in group),
            }

    maps = pair_maps(logs)
    tests: List[dict] = []
    metric_specs = [
        ("ProtoAcc@1", lambda m, r: selected_correct(m, r)),
        ("SemNavSucc@8m", lambda m, r: semnav(m, r, 8)),
        ("SemNavSucc@4m", lambda m, r: semnav(m, r, 4)),
    ]
    for metric_name, fn in metric_specs:
        for a, b in [("Ours-RG", "Random"), ("Oracle", "Ours-RG"), ("Oracle", "Random")]:
            tests.append(paired_test(maps, a, b, metric_name, fn))

    gate_rows = logs["Ours-RG"]
    gate = {
        "fallback_count": sum(bool(r.get("relation_gate_fallback", False)) for r in gate_rows),
        "candidate_count_distribution": dict(Counter(len(r.get("relation_gate_candidates", [])) for r in gate_rows)),
    }
    return {"overall": overall, "by_relation": by_relation, "paired_tests": tests, "relation_gate": gate}


def write_md(summary: dict, out_md: Path) -> None:
    lines: List[str] = [
        "# Supplemental Relation-Type Closed-Loop Rerun",
        "",
        "Protocol: 16 relation-constrained CARLA scenes (front/behind/left/right, 4 each), 3 paired seeds per scene, 48 episodes per method. Target graph slots are balanced 1/2.",
        "",
        "## Overall",
        "",
        "| Method | n | ProtoAcc@1 | SemNavSucc@8m | SemNavSucc@4m | Valid episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ["Ours-RG", "Random", "Oracle"]:
        row = summary["overall"][method]
        lines.append(
            f"| {method} | {row['proto']['n']} | {row['proto']['text']} | {row['sem8']['text']} | {row['sem4']['text']} | {row['valid']['text']} |"
        )
    lines.extend(["", "## By Relation", ""])
    for rel, methods in summary["by_relation"].items():
        lines.append(f"### {rel}")
        lines.append("")
        lines.append("| Method | ProtoAcc@1 | SemNavSucc@8m | SemNavSucc@4m |")
        lines.append("| --- | ---: | ---: | ---: |")
        for method in ["Ours-RG", "Random", "Oracle"]:
            row = methods[method]
            lines.append(f"| {method} | {row['proto']['text']} | {row['sem8']['text']} | {row['sem4']['text']} |")
        lines.append("")
    lines.extend(
        [
            "## Paired Tests",
            "",
            "| Metric | Comparison | Rate A | Rate B | A-only | B-only | Both | Neither | Exact p |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["paired_tests"]:
        lines.append(
            f"| {row['metric']} | {row['comparison']} | {fmt_pct(row['a_rate'])} | {fmt_pct(row['b_rate'])} | {row['a_only']} | {row['b_only']} | {row['both']} | {row['neither']} | {fmt_p(row['p'])} |"
        )
    gate = summary["relation_gate"]
    lines.extend(
        [
            "",
            "## Relation Gate Audit",
            "",
            f"- Fallback episodes: {gate['fallback_count']}",
            f"- Gate candidate-count distribution: {gate['candidate_count_distribution']}",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    scene_types = {int(s["id"]): str(s.get("query_type", s.get("scenario_type", "unknown"))) for s in scenarios}
    logs = {
        "Ours-RG": load_jsonl(args.ours),
        "Random": load_jsonl(args.random),
        "Oracle": load_jsonl(args.oracle),
    }
    summary = summarize(logs, scene_types)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_md(summary, args.out_md)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
