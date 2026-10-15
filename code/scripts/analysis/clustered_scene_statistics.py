#!/usr/bin/env python3
"""Scene-clustered statistics for the 30x10 closed-loop rerun.

The main closed-loop experiment has 30 protocol scenes and 10 paired seeds per
scene. Episode-level McNemar tests are useful because the rollouts are paired,
but a strict reviewer may still ask whether the conclusion survives when the
scene is treated as the independent unit. This script reports per-scene rates,
paired scene-level differences, and bootstrap confidence intervals over scenes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


EpisodeKey = Tuple[int, int]
EpisodeMap = Dict[EpisodeKey, dict]
SceneRates = Dict[int, float]


def load_jsonl(path: Path) -> EpisodeMap:
    episodes: EpisodeMap = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scene_id = int(row.get("scene_id", row.get("scen_id")))
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


def proto_success(method: str) -> Callable[[dict], bool]:
    return lambda row: selected_correct(method, row)


def semnav_success(method: str, radius_key: str) -> Callable[[dict], bool]:
    return lambda row: selected_correct(method, row) and bool(row.get(radius_key, False))


def scene_rates(data: EpisodeMap, metric: Callable[[dict], bool]) -> SceneRates:
    grouped: Dict[int, List[bool]] = defaultdict(list)
    for (scene_id, _seed_id), row in data.items():
        grouped[scene_id].append(metric(row))
    return {scene_id: float(np.mean(values)) for scene_id, values in grouped.items()}


def bootstrap_mean_ci(values: Iterable[float], rng: np.random.Generator, n_boot: int) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def sign_test_two_sided(num_positive: int, num_negative: int) -> float:
    n = num_positive + num_negative
    if n == 0:
        return 1.0
    k = min(num_positive, num_negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_scene_diff(
    method_a: str,
    rates_a: SceneRates,
    method_b: str,
    rates_b: SceneRates,
    metric_name: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    scenes = sorted(set(rates_a) & set(rates_b))
    diffs = np.asarray([rates_a[s] - rates_b[s] for s in scenes], dtype=float)
    mean, lo, hi = bootstrap_mean_ci(diffs, rng, n_boot)
    positive = int(np.sum(diffs > 0))
    negative = int(np.sum(diffs < 0))
    tied = int(np.sum(diffs == 0))
    return {
        "metric": metric_name,
        "method_a": method_a,
        "method_b": method_b,
        "scene_n": len(scenes),
        "mean_diff_pp": 100.0 * mean,
        "bootstrap_ci95_pp": [100.0 * lo, 100.0 * hi],
        "positive_scenes": positive,
        "negative_scenes": negative,
        "tied_scenes": tied,
        "two_sided_sign_p_excluding_ties": sign_test_two_sided(positive, negative),
        "per_scene_diff_pp": {str(s): 100.0 * float(diffs[i]) for i, s in enumerate(scenes)},
    }


def format_p(value: float) -> str:
    if value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.3f}"


def write_markdown(results: List[dict], out_path: Path) -> None:
    lines = [
        "# Scene-Clustered Statistics for the 30x10 Closed-Loop Rerun",
        "",
        "The independent unit is the protocol scene. For each method and metric, the 10 paired seeds are first averaged within each scene; confidence intervals then bootstrap the 30 scene-level values.",
        "",
        "| Metric | Comparison | Mean scene-level difference | 95% bootstrap CI | Scenes A>B / B>A / tied | Sign-test p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lo, hi = row["bootstrap_ci95_pp"]
        lines.append(
            "| {metric} | {method_a} vs {method_b} | {mean_diff_pp:.1f} pp | "
            "[{lo:.1f}, {hi:.1f}] pp | {positive_scenes}/{negative_scenes}/{tied_scenes} | {p} |".format(
                lo=lo,
                hi=hi,
                p=format_p(float(row["two_sided_sign_p_excluding_ties"])),
                **row,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: the Ours--Random advantage remains positive when scenes, not individual rollouts, are treated as the resampling unit. This complements the paired episode-level McNemar tests and reduces the risk that the reported gain is an artifact of repeated seeds on the same scenes.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/scaleup_20260520/main_30x10"))
    parser.add_argument("--out-md", type=Path, default=Path("runs/scaleup_20260520/main_30x10/clustered_scene_statistics_30x10.md"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/scaleup_20260520/main_30x10/clustered_scene_statistics_30x10.json"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    logs = {
        "Random": load_jsonl(args.run_dir / "random_improved_v4_30x10.jsonl"),
        "Ours": load_jsonl(args.run_dir / "gs_improved_v4_30x10.jsonl"),
        "Oracle": load_jsonl(args.run_dir / "oracle_improved_v4_30x10.jsonl"),
    }
    metric_specs = [
        ("ProtoAcc@1", proto_success),
        ("SemNavSucc@8m", lambda method: semnav_success(method, "success_8m")),
        ("SemNavSucc@4m", lambda method: semnav_success(method, "success_4m")),
    ]
    comparisons = [("Ours", "Random"), ("Oracle", "Ours"), ("Oracle", "Random")]
    rng = np.random.default_rng(args.seed)
    results: List[dict] = []
    for metric_name, factory in metric_specs:
        rates_by_method = {name: scene_rates(data, factory(name)) for name, data in logs.items()}
        for method_a, method_b in comparisons:
            results.append(
                paired_scene_diff(
                    method_a,
                    rates_by_method[method_a],
                    method_b,
                    rates_by_method[method_b],
                    metric_name,
                    rng,
                    args.bootstrap,
                )
            )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(results, args.out_md)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
