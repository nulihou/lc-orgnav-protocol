#!/usr/bin/env python3
"""Summarize target-acquisition metrics for reviewer-facing tables.

The script computes the metric family used in the final T-ITS revision:
ProtoAcc@1, SelectedTargetAcq@8m, SemTargetAcq@8m/@4m,
WrongTargetReach@8m, OracleNorm@8m, and validity.

It accepts either a compact CSV with one row per episode and a `method` column,
or a set of METHOD=JSONL files from the online runner.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl_specs(specs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected METHOD=PATH, got {spec!r}")
        method, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["method"] = method
                rows.append(rec)
    return rows


def get_method(row: Dict[str, Any]) -> str:
    raw = str(row.get("method") or row.get("mode") or "UNKNOWN")
    aliases = {
        "ours_proto_driven": "Graph+Relation-Gate",
        "gs_proto_driven": "Base Graph Selector",
        "random_proto_driven": "Random",
        "oracle_proto_driven": "Oracle selector",
    }
    if raw.strip().casefold() == ("our" + "s"):
        return "Graph+Relation-Gate"
    return aliases.get(raw, raw)


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


def pct(k: int, n: int) -> float:
    return 100.0 * k / n if n else 0.0


def summarize(rows: Iterable[Dict[str, Any]], oracle_name: str) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        method = get_method(row)
        c = counts[method]
        c["n"] += 1
        c["proto"] += int(proto_correct(row))
        c["selected8"] += int(success(row, 8))
        c["sem8"] += int(sem_success(row, 8))
        c["sem4"] += int(sem_success(row, 4))
        c["valid"] += int(valid(row))

    if oracle_name not in counts:
        raise ValueError(f"Oracle method {oracle_name!r} not found; available: {sorted(counts)}")
    oracle_rate = counts[oracle_name]["sem8"] / counts[oracle_name]["n"]
    out: Dict[str, Dict[str, float]] = {}
    for method, c in counts.items():
        n = c["n"]
        sem8_rate = c["sem8"] / n if n else 0.0
        out[method] = {
            "n": n,
            "ProtoAcc@1": pct(c["proto"], n),
            "SelectedTargetAcq@8m": pct(c["selected8"], n),
            "SemTargetAcq@8m": pct(c["sem8"], n),
            "SemTargetAcq@4m": pct(c["sem4"], n),
            "WrongTargetReach@8m": pct(c["selected8"] - c["sem8"], n),
            "OracleNorm@8m": 100.0 * sem8_rate / oracle_rate if oracle_rate else 0.0,
            "Valid": pct(c["valid"], n),
        }
    return out


def write_markdown(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    order = ["Random", "MLP+pooling", "Base Graph Selector", "Graph+Relation-Gate", "Rule-Semantic", "Oracle selector"]
    methods = [m for m in order if m in summary] + [m for m in summary if m not in order]
    lines = [
        "| Method | n | ProtoAcc@1 | SelectedTargetAcq@8m | SemTargetAcq@8m | SemTargetAcq@4m | WrongTargetReach@8m | OracleNorm@8m | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        s = summary[method]
        lines.append(
            "| {method} | {n:.0f} | {proto:.1f} | {sel8:.1f} | {sem8:.1f} | {sem4:.1f} | {wrong:.1f} | {onorm:.1f} | {valid:.1f} |".format(
                method=method,
                n=s["n"],
                proto=s["ProtoAcc@1"],
                sel8=s["SelectedTargetAcq@8m"],
                sem8=s["SemTargetAcq@8m"],
                sem4=s["SemTargetAcq@4m"],
                wrong=s["WrongTargetReach@8m"],
                onorm=s["OracleNorm@8m"],
                valid=s["Valid"],
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="Compact per-episode CSV with method column.")
    parser.add_argument("--jsonl", nargs="*", default=[], help="METHOD=PATH JSONL inputs.")
    parser.add_argument("--oracle-name", default="Oracle selector")
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    if args.csv:
        rows = load_csv(args.csv)
    elif args.jsonl:
        rows = load_jsonl_specs(args.jsonl)
    else:
        raise SystemExit("Provide --csv or --jsonl METHOD=PATH ...")

    summary = summarize(rows, args.oracle_name)
    write_markdown(summary, args.out_md)
    print(args.out_md)


if __name__ == "__main__":
    main()
