#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the frozen Hard matched-selector main-table row (no retrain).

Usage (from repository root):
  python code/scripts/tools/print_main_table_hard_row.py

Reads results_meta/offline/matched_hard_row.json (extracted from SUMMARY.md).
Optionally checks that experiments/P3/PREDICTION_HASHES.json exists.
Does not invent numbers; does not push or mint DOIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # code_doi_staging/
ROW = ROOT / "results_meta" / "offline" / "matched_hard_row.json"
P3_HASHES = ROOT / "experiments" / "P3" / "PREDICTION_HASHES.json"
SUMMARY = ROOT / "results_meta" / "offline" / "SUMMARY.md"


def main() -> int:
    if not ROW.is_file():
        print(f"ERROR: missing {ROW}", file=sys.stderr)
        return 1
    data = json.loads(ROW.read_text(encoding="utf-8"))
    print("Hard-confusable matched selectors (seeds 7/21/42) — frozen companion row")
    print(f"source: {data.get('source')}")
    print("-" * 72)
    print(f"{'Selector':40s} {'7':>8s} {'21':>8s} {'42':>8s} {'mean':>8s} {'SD':>6s}")
    for r in data["rows"]:
        c = r["seed_correct_over_300"]
        print(
            f"{r['paper_name']:40s} "
            f"{c[0]:3d}/300 {c[1]:3d}/300 {c[2]:3d}/300 "
            f"{r['mean_pct']:7.2f} {r['sd_pp']:6.2f}"
        )
    lo, hi = data["range_mean_pct"]
    print("-" * 72)
    print(f"Paper abstract/body Hard range (3-seed means): {lo:.2f}--{hi:.2f}%")
    print(f"SUMMARY present: {SUMMARY.is_file()}")
    print(f"P3 PREDICTION_HASHES present: {P3_HASHES.is_file()}")
    if P3_HASHES.is_file():
        hashes = json.loads(P3_HASHES.read_text(encoding="utf-8"))
        print(f"P3 hash entries: {len(hashes)}")
    print("OK — numbers are frozen JSON only; verify SHA-256 before any rescoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
