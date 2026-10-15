#!/usr/bin/env python3
"""Summarize unified safety fields from protocol-driven CARLA JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return records


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "NA"
    return f"{100.0 * numerator / denominator:.1f}%"


def finite_values(records: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize(name: str, path: Path) -> Dict[str, Any]:
    records = read_jsonl(path)
    valid = [r for r in records if bool(r.get("valid", True))]
    logged = [r for r in valid if r.get("safety_logger_version")]
    collision_records = [r for r in logged if bool(r.get("collision"))]
    lane_records = [r for r in logged if bool(r.get("lane_invasion"))]
    ttc_values = finite_values(logged, "min_ttc_s")
    min_distance_values = finite_values(logged, "min_distance_to_vehicle_m")

    return {
        "method": name,
        "path": str(path),
        "episodes": len(records),
        "valid": len(valid),
        "safety_logged": len(logged),
        "collision_count": len(collision_records),
        "lane_invasion_count": len(lane_records),
        "min_ttc_s": min(ttc_values) if ttc_values else None,
        "min_distance_to_vehicle_m": min(min_distance_values) if min_distance_values else None,
    }


def write_markdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Unified Safety Metrics",
        "",
        "| Method | Episodes | Valid | Safety logged | Collision episodes | Lane-invasion episodes | Min TTC (s) | Min vehicle distance (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        logged = int(row["safety_logged"])
        min_ttc = row["min_ttc_s"]
        min_dist = row["min_distance_to_vehicle_m"]
        lines.append(
            "| {method} | {episodes} | {valid} | {safety_logged} | "
            "{collision_count} ({collision_pct}) | {lane_invasion_count} ({lane_pct}) | "
            "{min_ttc} | {min_dist} |".format(
                method=row["method"],
                episodes=row["episodes"],
                valid=row["valid"],
                safety_logged=row["safety_logged"],
                collision_count=row["collision_count"],
                collision_pct=pct(int(row["collision_count"]), logged),
                lane_invasion_count=row["lane_invasion_count"],
                lane_pct=pct(int(row["lane_invasion_count"]), logged),
                min_ttc=f"{min_ttc:.2f}" if min_ttc is not None else "NA",
                min_dist=f"{min_dist:.2f}" if min_dist is not None else "NA",
            )
        )
    lines.extend([
        "",
        "Only episodes with `safety_logger_version` are included in collision, lane-invasion, and TTC rates.",
        "Older JSONL files without unified safety logging should not be used for method-level collision claims.",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_log_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use METHOD=path for each --log argument")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", type=parse_log_arg, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize(name, path) for name, path in args.log]
    write_markdown(rows, args.out_md)


if __name__ == "__main__":
    main()
