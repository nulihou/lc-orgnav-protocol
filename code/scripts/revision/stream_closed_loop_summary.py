#!/usr/bin/env python3
"""Stream closed-loop JSONL files into reviewer-facing metric tables.

The online runners store full trajectories, so merged JSONL files can be
gigabyte-sized.  This script keeps only aggregate and per-scene counters.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def proto_correct(row: Dict[str, Any]) -> bool:
    if "selected_correct" in row:
        return truthy(row.get("selected_correct"))
    return truthy(row.get("gs_correct"))


def success(row: Dict[str, Any], radius: int) -> bool:
    return truthy(row.get(f"success_{radius}m"))


def sem_success(row: Dict[str, Any], radius: int) -> bool:
    key = f"semnav_{radius}m"
    if key in row:
        return truthy(row.get(key))
    return proto_correct(row) and success(row, radius)


def valid(row: Dict[str, Any]) -> bool:
    if "valid" in row:
        return truthy(row.get("valid"))
    return True


def scene_id(row: Dict[str, Any]) -> int:
    return int(row.get("scene_id", row.get("scen_id", -1)))


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def add_counter(c: Dict[str, int], row: Dict[str, Any]) -> None:
    c["n"] += 1
    c["proto"] += int(proto_correct(row))
    c["selected8"] += int(success(row, 8))
    c["sem8"] += int(sem_success(row, 8))
    c["sem4"] += int(sem_success(row, 4))
    c["valid"] += int(valid(row))
    c["collision"] += int(truthy(row.get("collision")))
    c["lane_invasion"] += int(truthy(row.get("lane_invasion")))


def summarize_counter(c: Dict[str, int], oracle_rate: float | None = None) -> Dict[str, float]:
    n = c["n"]
    sem8_rate = c["sem8"] / n if n else 0.0
    out = {
        "n": float(n),
        "ProtoAcc@1": pct(c["proto"], n),
        "SelectedTargetAcq@8m": pct(c["selected8"], n),
        "SemTargetAcq@8m": pct(c["sem8"], n),
        "SemTargetAcq@4m": pct(c["sem4"], n),
        "WrongTargetReach@8m": pct(c["selected8"] - c["sem8"], n),
        "Valid": pct(c["valid"], n),
        "Collision": pct(c["collision"], n),
        "LaneInvasion": pct(c["lane_invasion"], n),
    }
    if oracle_rate is not None:
        out["OracleNorm@8m"] = 100.0 * sem8_rate / oracle_rate if oracle_rate else 0.0
    return out


def normal_ci(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, mean
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    half = 1.96 * math.sqrt(var / len(vals))
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", nargs="+", required=True, help="METHOD=PATH inputs")
    parser.add_argument("--oracle-name", default="Oracle selector")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_scene: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for spec in args.jsonl:
        if "=" not in spec:
            raise SystemExit(f"Expected METHOD=PATH, got {spec!r}")
        method, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                add_counter(totals[method], row)
                add_counter(by_scene[method][scene_id(row)], row)

    if args.oracle_name not in totals:
        raise SystemExit(f"Oracle method {args.oracle_name!r} missing; got {sorted(totals)}")
    oracle_rate = totals[args.oracle_name]["sem8"] / totals[args.oracle_name]["n"]

    summary = {m: summarize_counter(c, oracle_rate) for m, c in totals.items()}
    for method, scene_counters in by_scene.items():
        sem8_scene = [
            summarize_counter(c)["SemTargetAcq@8m"] for _, c in sorted(scene_counters.items())
        ]
        proto_scene = [
            summarize_counter(c)["ProtoAcc@1"] for _, c in sorted(scene_counters.items())
        ]
        lo, hi = normal_ci(sem8_scene)
        plo, phi = normal_ci(proto_scene)
        summary[method]["SceneMeanSemTargetAcq@8m"] = sum(sem8_scene) / len(sem8_scene)
        summary[method]["SceneCI95SemTargetAcq@8mLow"] = lo
        summary[method]["SceneCI95SemTargetAcq@8mHigh"] = hi
        summary[method]["SceneMeanProtoAcc@1"] = sum(proto_scene) / len(proto_scene)
        summary[method]["SceneCI95ProtoAcc@1Low"] = plo
        summary[method]["SceneCI95ProtoAcc@1High"] = phi
        summary[method]["SceneCount"] = float(len(scene_counters))

    order = [
        "Random",
        "MLP+pooling",
        "Graph+Relation-Gate",
        "Coarse Rule-Semantic",
        "Fine Rank Rule",
        "Oracle selector",
    ]
    methods = [m for m in order if m in summary] + [m for m in summary if m not in order]
    lines = [
        "| Method | n | ProtoAcc@1 | SemTargetAcq@8m | SemTargetAcq@4m | SelectedTargetAcq@8m | WrongTargetReach@8m | OracleNorm@8m | Valid | Scene-clustered SemTargetAcq@8m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        s = summary[method]
        lines.append(
            "| {method} | {n:.0f} | {proto:.1f} | {sem8:.1f} | {sem4:.1f} | {sel8:.1f} | {wrong:.1f} | {onorm:.1f} | {valid:.1f} | {smean:.1f} [{lo:.1f}, {hi:.1f}] |".format(
                method=method,
                n=s["n"],
                proto=s["ProtoAcc@1"],
                sem8=s["SemTargetAcq@8m"],
                sem4=s["SemTargetAcq@4m"],
                sel8=s["SelectedTargetAcq@8m"],
                wrong=s["WrongTargetReach@8m"],
                onorm=s["OracleNorm@8m"],
                valid=s["Valid"],
                smean=s["SceneMeanSemTargetAcq@8m"],
                lo=s["SceneCI95SemTargetAcq@8mLow"],
                hi=s["SceneCI95SemTargetAcq@8mHigh"],
            )
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.out_md)


if __name__ == "__main__":
    main()
