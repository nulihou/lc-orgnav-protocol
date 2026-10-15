"""Build a compact case-level evidence grid from the 30x10 closed-loop logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHOD_FILES = {
    "ours": "gs_improved_v4_30x10.jsonl",
    "random": "random_improved_v4_30x10.jsonl",
    "oracle": "oracle_improved_v4_30x10.jsonl",
}


def read_jsonl(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("trajectory", None)
            scene_id = int(row.get("scene_id", row.get("scen_id")))
            run_id = int(row["run_id"])
            records[(scene_id, run_id)] = row
    return records


def read_oracle_tags(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (int(row["scene_id"]), int(row["run_id"])): row
            for row in reader
        }


def is_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("gs_correct", row.get("oracle_correct", False)))


def sem8(row: dict[str, Any]) -> bool:
    return is_correct(row) and bool(row.get("success_8m", False))


def sem4(row: dict[str, Any]) -> bool:
    return is_correct(row) and bool(row.get("success_4m", False))


def status(row: dict[str, Any]) -> str:
    return (
        f"choose {row.get('chosen_idx_graph')} / gt {row.get('target_idx_gt')}; "
        f"Sem8={'Y' if sem8(row) else 'N'}, Sem4={'Y' if sem4(row) else 'N'}, "
        f"min={float(row.get('min_dist', 0.0)):.2f} m"
    )


def first_case(keys: list[tuple[int, int]], predicate) -> tuple[int, int] | None:
    for key in keys:
        if predicate(key):
            return key
    return None


def build_grid(log_dir: Path, output: Path) -> None:
    logs = {name: read_jsonl(log_dir / filename) for name, filename in METHOD_FILES.items()}
    tags = read_oracle_tags(log_dir / "oracle_failure_tags_30x10.csv")
    keys = sorted(set.intersection(*(set(records) for records in logs.values())))

    selectors: list[tuple[str, str, Any]] = [
        (
            "Semantic gain over Random",
            "Ours reaches the semantic target while Random reaches a selected decoy.",
            lambda k: sem8(logs["ours"][k])
            and bool(logs["random"][k].get("success_8m", False))
            and not sem8(logs["random"][k]),
        ),
        (
            "Residual selector miss",
            "Oracle succeeds but Ours selects or acquires incorrectly.",
            lambda k: sem8(logs["oracle"][k]) and not sem8(logs["ours"][k]),
        ),
        (
            "Random lucky episode",
            "Random succeeds while Ours fails, kept as a guard against cherry-picking.",
            lambda k: sem8(logs["random"][k]) and not sem8(logs["ours"][k]),
        ),
        (
            "Oracle controller ceiling at 8 m",
            "Even target-index oracle fails at 8 m, exposing fixed-controller limitations.",
            lambda k: is_correct(logs["oracle"][k])
            and not bool(logs["oracle"][k].get("success_8m", False)),
        ),
        (
            "Strict 4 m docking failure",
            "Oracle acquires the target at 8 m but fails the stricter close-range criterion.",
            lambda k: sem8(logs["oracle"][k]) and not sem4(logs["oracle"][k]),
        ),
    ]

    rows: list[tuple[str, tuple[int, int], str, str, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for label, interpretation, predicate in selectors:
        key = first_case([k for k in keys if k not in seen], predicate)
        if key is None:
            key = first_case(keys, predicate)
        if key is None:
            continue
        seen.add(key)
        tag = tags.get(key, {})
        if label == "Oracle controller ceiling at 8 m":
            failure = tag.get("failure_tag_8m") or "not tagged"
            interpretation = f"{interpretation} Failure tag: {failure}."
        elif label == "Strict 4 m docking failure":
            interpretation = f"{interpretation} Failure tag: lane_misalignment after 8 m acquisition."
        rows.append(
            (
                label,
                key,
                status(logs["ours"][key]),
                status(logs["random"][key]),
                status(logs["oracle"][key]),
                interpretation,
            )
        )

    lines = [
        "# Case-Level Evidence Grid",
        "",
        "This grid is generated from the official 30x10 closed-loop logs. It is intended for reviewer-response auditing rather than as a new headline result.",
        "",
        "| Case type | Scene/run | Ours | Random | Oracle | Interpretation |",
        "|---|---:|---|---|---|---|",
    ]
    for label, key, ours, random, oracle, interpretation in rows:
        lines.append(
            f"| {label} | scen {key[0]:03d}, run {key[1]} | {ours} | {random} | {oracle} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "Use in R2: cite this only as an audit trail. The paper-facing numerical claims remain the aggregate 30x10 tables, paired McNemar tests, and selection-control decomposition.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_grid(args.log_dir, args.output)


if __name__ == "__main__":
    main()
