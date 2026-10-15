#!/usr/bin/env python3
"""Summarize the scale-300 closed-loop experiment for paper tables."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


METHOD_FILES = {
    "Random": "random_scale300_300x5.jsonl",
    "MLP+pooling": "mlp_pooling_scale300_300x5.jsonl",
    "Ours": "ours_scale300_300x5.jsonl",
    "Rule-Semantic": "rule_semantic_scale300_300x5.jsonl",
    "Oracle selector": "oracle_scale300_300x5.jsonl",
}


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4 * n * n))) / denom
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


def selected_correct(method: str, row: dict) -> bool:
    if method.lower().startswith("oracle"):
        return True
    return row.get("chosen_idx_graph") == row.get("target_idx_gt")


def nav(row: dict, radius: int) -> bool:
    return bool(row.get(f"success_{radius}m", False))


def semnav(method: str, row: dict, radius: int) -> bool:
    return selected_correct(method, row) and nav(row, radius)


def valid(row: dict) -> bool:
    return row.get("valid", True) is not False


def scene_id(row: dict) -> int:
    return int(row.get("scene_id", row.get("scen_id")))


def scene_cluster_bootstrap(
    rows: List[dict],
    method: str,
    fn: Callable[[str, dict], bool],
    iterations: int = 10000,
    seed: int = 42,
) -> dict:
    by_scene: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        by_scene[scene_id(row)].append(row)
    scene_ids = sorted(by_scene)
    per_scene = {sid: sum(fn(method, row) for row in group) / len(group) for sid, group in by_scene.items()}
    mean = sum(per_scene.values()) / len(per_scene) if per_scene else 0.0
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(iterations):
        draw = [rng.choice(scene_ids) for _ in scene_ids]
        samples.append(sum(per_scene[sid] for sid in draw) / len(draw))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))] if samples else 0.0
    hi = samples[int(0.975 * (len(samples) - 1))] if samples else 0.0
    return {
        "scene_n": len(scene_ids),
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "text": f"{100*mean:.1f}% [{100*lo:.1f}, {100*hi:.1f}]",
    }


def episode_key(row: dict) -> Tuple[int, int]:
    return scene_id(row), int(row.get("run_id", row.get("seed_index", row.get("episode"))))


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def paired_test(logs: Dict[str, List[dict]], a: str, b: str, metric: str, fn: Callable[[str, dict], bool]) -> dict:
    maps = {name: {episode_key(row): row for row in rows} for name, rows in logs.items()}
    keys = sorted(set(maps[a]) & set(maps[b]))
    a_only = b_only = both = neither = 0
    for key in keys:
        av = fn(a, maps[a][key])
        bv = fn(b, maps[b][key])
        if av and bv:
            both += 1
        elif av:
            a_only += 1
        elif bv:
            b_only += 1
        else:
            neither += 1
    return {
        "metric": metric,
        "comparison": f"{a} vs {b}",
        "n": len(keys),
        "a_rate": (both + a_only) / len(keys) if keys else 0.0,
        "b_rate": (both + b_only) / len(keys) if keys else 0.0,
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "p": exact_mcnemar_p(a_only, b_only),
    }


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def fmt_p(x: float) -> str:
    return f"{x:.3e}" if x < 1e-3 else f"{x:.3f}"


def load_scenario_meta(run_root: Path) -> Dict[int, dict]:
    path = run_root / "scenarios" / "test_scenarios.json"
    if not path.exists():
        return {}
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["id"]): item for item in scenarios}


def row_meta_value(row: dict, scenario_meta: Dict[int, dict], key: str):
    if row.get(key) is not None:
        return row.get(key)
    meta = scenario_meta.get(scene_id(row), {})
    return meta.get(key)


def stratified_metrics(
    logs: Dict[str, List[dict]],
    scenario_meta: Dict[int, dict],
    group_key: str,
    metric_fns: Dict[str, Callable[[str, dict], bool]],
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    out: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for method, rows in logs.items():
        groups: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            value = row_meta_value(row, scenario_meta, group_key)
            groups[str(value if value is not None else "unknown")].append(row)
        out[method] = {}
        for value, group_rows in sorted(groups.items(), key=lambda kv: kv[0]):
            out[method][value] = {
                metric: metric_summary(fn(method, row) for row in group_rows)
                for metric, fn in metric_fns.items()
                if metric in ("ProtoAcc@1", "SemNavSucc@8m", "SemNavSucc@4m")
            }
    return out


def append_stratified_section(
    lines: List[str],
    title: str,
    stratified: Dict[str, Dict[str, Dict[str, dict]]],
    value_label: str,
    value_names: Optional[Dict[str, str]] = None,
) -> None:
    lines.extend(["", f"## {title}", ""])
    values = sorted({value for method_rows in stratified.values() for value in method_rows})
    for value in values:
        label = value_names.get(value, value) if value_names else value
        lines.extend(
            [
                f"### {value_label} = {label}",
                "",
                "| Method | n | ProtoAcc@1 | SemNavSucc@8m | SemNavSucc@4m |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in ("Random", "MLP+pooling", "Ours", "Rule-Semantic", "Oracle selector"):
            row = stratified.get(method, {}).get(value)
            if not row:
                continue
            n = row["ProtoAcc@1"]["n"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        str(n),
                        row["ProtoAcc@1"]["text"],
                        row["SemNavSucc@8m"]["text"],
                        row["SemNavSucc@4m"]["text"],
                    ]
                )
                + " |"
            )
        lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--closed-root",
        type=Path,
        default=None,
        help="Directory containing merged *_scale300_300x5.jsonl files. Defaults to RUN_ROOT/closed_loop.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    closed_root = args.closed_root if args.closed_root is not None else args.run_root / "closed_loop"
    logs: Dict[str, List[dict]] = {}
    missing = []
    for method, filename in METHOD_FILES.items():
        path = closed_root / filename
        if not path.exists():
            missing.append(str(path))
        else:
            logs[method] = load_jsonl(path)
    if missing:
        raise RuntimeError("Missing closed-loop JSONL files:\n" + "\n".join(missing))

    metric_fns = {
        "ProtoAcc@1": selected_correct,
        "SemNavSucc@8m": lambda m, r: semnav(m, r, 8),
        "SemNavSucc@4m": lambda m, r: semnav(m, r, 4),
        "NavSucc@8m": lambda m, r: nav(r, 8),
        "Valid episodes": lambda m, r: valid(r),
    }
    overall: Dict[str, dict] = {}
    for method, rows in logs.items():
        overall[method] = {}
        for metric, fn in metric_fns.items():
            overall[method][metric] = {
                "episode": metric_summary(fn(method, row) for row in rows),
                "scene_clustered": scene_cluster_bootstrap(
                    rows,
                    method,
                    fn,
                    iterations=int(args.bootstrap_iterations),
                ),
            }

    scenario_meta = load_scenario_meta(args.run_root)
    stratified = {
        "by_num_candidates": stratified_metrics(logs, scenario_meta, "num_candidates", metric_fns),
        "by_relation": stratified_metrics(logs, scenario_meta, "rel_type", metric_fns),
    }

    tests = []
    for metric in ("ProtoAcc@1", "SemNavSucc@8m", "SemNavSucc@4m"):
        tests.append(paired_test(logs, "Ours", "MLP+pooling", metric, metric_fns[metric]))
        tests.append(paired_test(logs, "Ours", "Rule-Semantic", metric, metric_fns[metric]))
        tests.append(paired_test(logs, "Ours", "Random", metric, metric_fns[metric]))
        tests.append(paired_test(logs, "MLP+pooling", "Random", metric, metric_fns[metric]))

    summary = {
        "run_root": str(args.run_root),
        "closed_root": str(closed_root),
        "protocol": "300 independent test scenes x 5 paired seeds x 5 selectors",
        "overall": overall,
        "stratified": stratified,
        "paired_tests": tests,
    }
    out_json = closed_root / "scale300_closed_loop_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Scale-300 Closed-Loop Summary",
        "",
        "Protocol: 300 independent relation-identifiable CARLA protocol scenes, 5 paired seeds per scene, 1500 episodes per method.",
        "",
        "## Main Table Candidate",
        "",
        "| Method | n | ProtoAcc@1 | SemNavSucc@8m | SemNavSucc@4m | Valid episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("Random", "MLP+pooling", "Ours", "Rule-Semantic", "Oracle selector"):
        row = overall[method]
        n = row["ProtoAcc@1"]["episode"]["n"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(n),
                    row["ProtoAcc@1"]["scene_clustered"]["text"],
                    row["SemNavSucc@8m"]["scene_clustered"]["text"],
                    row["SemNavSucc@4m"]["scene_clustered"]["text"],
                    row["Valid episodes"]["episode"]["text"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Scene-clustered bootstrap CIs are shown for the main performance metrics. Episode-level Wilson CIs remain in the JSON summary.",
            "",
            "## Paired Tests",
            "",
            "| Metric | Comparison | Rate A | Rate B | A-only | B-only | Both | Neither | Exact p |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tests:
        lines.append(
            f"| {row['metric']} | {row['comparison']} | {fmt_pct(row['a_rate'])} | {fmt_pct(row['b_rate'])} | "
            f"{row['a_only']} | {row['b_only']} | {row['both']} | {row['neither']} | {fmt_p(row['p'])} |"
        )
    append_stratified_section(lines, "Candidate-Count Stratification", stratified["by_num_candidates"], "K")
    append_stratified_section(
        lines,
        "Relation-Type Stratification",
        stratified["by_relation"],
        "rel_type",
        {"0": "near", "1": "front", "2": "behind", "3": "left", "4": "right"},
    )
    out_md = closed_root / "scale300_closed_loop_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
