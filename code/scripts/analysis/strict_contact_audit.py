#!/usr/bin/env python3
"""Produce a stricter contact audit from unified CARLA safety logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DYNAMIC_PREFIXES = ("vehicle.", "walker.")
STATIC_PREFIXES = ("static.", "traffic.", "road.", "sidewalk.", "vegetation.")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def parse_log_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use METHOD=path for each --log")
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def pct(k: int, n: int) -> str:
    if n <= 0:
        return "NA"
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def actor_types(row: Dict[str, Any]) -> List[str]:
    raw = row.get("collision_actor_type_ids") or []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def has_dynamic_contact(row: Dict[str, Any]) -> bool:
    return any(t.startswith(DYNAMIC_PREFIXES) for t in actor_types(row))


def has_static_only_contact(row: Dict[str, Any]) -> bool:
    types = actor_types(row)
    return bool(row.get("collision")) and bool(types) and not has_dynamic_contact(row)


def is_severe(row: Dict[str, Any], threshold: float) -> bool:
    value = row.get("collision_max_normal_impulse")
    return isinstance(value, (int, float)) and float(value) >= threshold


def summarize(name: str, path: Path, severe_threshold: float) -> Dict[str, Any]:
    rows = read_jsonl(path)
    valid = [r for r in rows if r.get("valid", True) is not False]
    logged = [r for r in valid if r.get("safety_logger_version")]
    raw_contact = [r for r in logged if bool(r.get("collision"))]
    dynamic = [r for r in logged if has_dynamic_contact(r)]
    static_only = [r for r in logged if has_static_only_contact(r)]
    severe = [r for r in logged if is_severe(r, severe_threshold)]
    dynamic_or_severe = [r for r in logged if has_dynamic_contact(r) or is_severe(r, severe_threshold)]
    lane = [r for r in logged if bool(r.get("lane_invasion"))]
    unknown_contact = [
        r
        for r in raw_contact
        if not actor_types(r)
    ]
    return {
        "method": name,
        "path": str(path),
        "episodes": len(rows),
        "valid": len(valid),
        "logged": len(logged),
        "raw_contact": len(raw_contact),
        "dynamic_actor_contact": len(dynamic),
        "static_only_contact": len(static_only),
        "severe_contact": len(severe),
        "dynamic_or_severe_contact": len(dynamic_or_severe),
        "unknown_actor_contact": len(unknown_contact),
        "lane_invasion": len(lane),
    }


def write_md(rows: List[Dict[str, Any]], out_path: Path, severe_threshold: float) -> None:
    lines = [
        "# Strict Contact Audit",
        "",
        f"Severe contact threshold: `collision_max_normal_impulse >= {severe_threshold:g}`.",
        "",
        "This table separates raw CARLA collision-sensor triggers from contacts involving dynamic actors. It is an audit table, not evidence that Ours has a collision-rate advantage.",
        "",
        "| Method | Valid | Logged | Raw contact | Dynamic actor contact | Static-only contact | Severe contact | Dynamic-or-severe contact | Unknown contact actor | Lane invasion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        n = int(row["logged"])
        lines.append(
            "| {method} | {valid} | {logged} | {raw} | {dynamic} | {static_only} | {severe} | {dynamic_or_severe} | {unknown} | {lane} |".format(
                method=row["method"],
                valid=row["valid"],
                logged=row["logged"],
                raw=pct(int(row["raw_contact"]), n),
                dynamic=pct(int(row["dynamic_actor_contact"]), n),
                static_only=pct(int(row["static_only_contact"]), n),
                severe=pct(int(row["severe_contact"]), n),
                dynamic_or_severe=pct(int(row["dynamic_or_severe_contact"]), n),
                unknown=pct(int(row["unknown_actor_contact"]), n),
                lane=pct(int(row["lane_invasion"]), n),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: the unified logger closes the reporting gap, but raw contact rates remain high for all selectors under the shared controller. The paper should only claim a transparent contact audit unless a future dynamic/severe-contact metric shows a method-level benefit with paired statistics.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", type=parse_log_arg, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--severe-impulse-threshold", type=float, default=500.0)
    args = parser.parse_args()

    rows = [summarize(name, path, float(args.severe_impulse_threshold)) for name, path in args.log]
    write_md(rows, args.out_md, float(args.severe_impulse_threshold))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.out_md}")


if __name__ == "__main__":
    main()
