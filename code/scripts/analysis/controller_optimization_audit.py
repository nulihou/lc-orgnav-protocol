#!/usr/bin/env python3
"""Compare pre-optimization and revised shared-controller Oracle logs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.unified_episode_evaluator import evaluate_episode  # type: ignore


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def _episode_key(row: Dict[str, Any]) -> Tuple[int, int]:
    scen = int(row.get("scen_id", row.get("scene_id", -1)))
    seed = row.get("seed_id", row.get("seed_index", row.get("run_id", row.get("episode", -1))))
    return scen, int(seed)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z / denom * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, center - margin), min(1.0, center + margin)


def _stats(flags: Iterable[bool]) -> Dict[str, Any]:
    vals = list(flags)
    total = len(vals)
    successes = sum(1 for v in vals if v)
    lo, hi = _wilson(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
        "ci_low": lo,
        "ci_high": hi,
    }


def _fmt_stat(s: Dict[str, Any]) -> str:
    return (
        f"{100.0 * s['rate']:.1f}% "
        f"({s['successes']}/{s['total']}, "
        f"95% CI {100.0 * s['ci_low']:.1f}-{100.0 * s['ci_high']:.1f})"
    )


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [evaluate_episode(row) for row in rows]
    nav8 = [bool(e["navsucc8"]) for e in evaluated]
    nav4 = [bool(e["navsucc4"]) for e in evaluated]
    timeout = [bool(e.get("timeout", False)) for e in evaluated]
    valid = [bool(row.get("valid", True)) for row in rows]
    dmins = [
        float(e["d_min_to_selected"])
        for e in evaluated
        if e.get("d_min_to_selected") is not None
    ]
    return {
        "n": len(rows),
        "navsucc8": _stats(nav8),
        "navsucc4": _stats(nav4),
        "timeout": _stats(timeout),
        "valid": _stats(valid),
        "median_min_dist": sorted(dmins)[len(dmins) // 2] if dmins else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("runs/full_rerun_20260520/carla/oracle_proto_driven_30x5.jsonl"),
    )
    parser.add_argument(
        "--improved",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/oracle_improved_v4_30x10.jsonl"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/controller_optimization_audit.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/controller_optimization_audit.json"),
    )
    args = parser.parse_args()

    baseline_rows = _load_jsonl(args.baseline)
    improved_rows_all = _load_jsonl(args.improved)
    improved_by_key = {_episode_key(row): row for row in improved_rows_all}
    matched_pairs = [
        (row, improved_by_key[_episode_key(row)])
        for row in baseline_rows
        if _episode_key(row) in improved_by_key
    ]
    matched_baseline = [a for a, _ in matched_pairs]
    matched_improved = [b for _, b in matched_pairs]

    baseline_summary = _summarize(matched_baseline)
    improved_summary = _summarize(matched_improved)

    base_eval = [evaluate_episode(row) for row in matched_baseline]
    imp_eval = [evaluate_episode(row) for row in matched_improved]
    paired_gain8 = sum(
        int(not bool(b["navsucc8"]) and bool(i["navsucc8"]))
        for b, i in zip(base_eval, imp_eval)
    )
    paired_loss8 = sum(
        int(bool(b["navsucc8"]) and not bool(i["navsucc8"]))
        for b, i in zip(base_eval, imp_eval)
    )
    paired_gain4 = sum(
        int(not bool(b["navsucc4"]) and bool(i["navsucc4"]))
        for b, i in zip(base_eval, imp_eval)
    )
    paired_loss4 = sum(
        int(bool(b["navsucc4"]) and not bool(i["navsucc4"]))
        for b, i in zip(base_eval, imp_eval)
    )

    payload = {
        "baseline": str(args.baseline),
        "improved": str(args.improved),
        "matched_episode_count": len(matched_pairs),
        "baseline_summary": baseline_summary,
        "improved_summary": improved_summary,
        "paired_gain8": paired_gain8,
        "paired_loss8": paired_loss8,
        "paired_gain4": paired_gain4,
        "paired_loss4": paired_loss4,
        "baseline_controller_profiles": sorted(
            {str(r.get("controller_profile", "pre-optimization/default")) for r in matched_baseline}
        ),
        "improved_controller_profiles": sorted(
            {str(r.get("controller_profile", "unknown")) for r in matched_improved}
        ),
        "improved_planner_modes": sorted({str(r.get("planner_mode", "")) for r in matched_improved}),
    }

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Controller Optimization Audit",
        "",
        "This paired audit compares the archived pre-optimization Oracle controller log with the revised shared controller on the same scene-seed keys. It is controller-facing evidence only; the final R2 method comparison still uses the same revised controller for Ours, Random, and Oracle.",
        "",
        f"- Baseline log: `{args.baseline.as_posix()}`",
        f"- Revised log: `{args.improved.as_posix()}`",
        f"- Matched scene-seed episodes: {len(matched_pairs)}",
        f"- Baseline profile labels: {', '.join(payload['baseline_controller_profiles'])}",
        f"- Revised profile labels: {', '.join(payload['improved_controller_profiles'])}",
        "",
        "| Oracle controller setting | NavSucc@8m | NavSucc@4m | Timeout flag | Valid episodes | Median min distance |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Pre-optimization archived fixed controller | {_fmt_stat(baseline_summary['navsucc8'])} | "
            f"{_fmt_stat(baseline_summary['navsucc4'])} | {_fmt_stat(baseline_summary['timeout'])} | "
            f"{_fmt_stat(baseline_summary['valid'])} | {baseline_summary['median_min_dist']:.2f} m |"
        ),
        (
            f"| Revised shared controller (`improved_v4`) | {_fmt_stat(improved_summary['navsucc8'])} | "
            f"{_fmt_stat(improved_summary['navsucc4'])} | {_fmt_stat(improved_summary['timeout'])} | "
            f"{_fmt_stat(improved_summary['valid'])} | {improved_summary['median_min_dist']:.2f} m |"
        ),
        "",
        f"- Paired 8 m changes: {paired_gain8} baseline failures became successes; {paired_loss8} baseline successes became failures.",
        f"- Paired 4 m changes: {paired_gain4} baseline failures became successes; {paired_loss4} baseline successes became failures.",
        "",
        "Interpretation: the revised controller materially improves the controller-bounded Oracle target-acquisition ceiling on matched episodes, but it is still imperfect. The manuscript should therefore keep the controller as a shared evaluation stack, report Oracle failures transparently, and avoid claiming a solved low-level control contribution.",
        "",
    ]
    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] wrote {args.out_md}")
    print(f"[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
