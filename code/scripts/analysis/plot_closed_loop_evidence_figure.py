#!/usr/bin/env python3
"""Create the closed-loop evidence figure for the R2 manuscript.

The figure uses only paper-facing R2 logs and summary files. It visualizes the
same quantities already reported in tables: main semantic acquisition results,
threshold sensitivity, paired scene-seed outcomes, and selection/control
decomposition.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METHODS = ["Random", "Base Graph", "Oracle"]
METHOD_COLORS = {
    "Random": "#D9822B",
    "Base Graph": "#2F6FA3",
    "Oracle": "#5B6770",
}


def sem_success(rec: dict, radius: int = 8) -> bool:
    return bool(rec.get("gs_correct", False)) and bool(rec.get(f"success_{radius}m", False))


def key_for(rec: dict) -> tuple[int, int]:
    scene = rec.get("scene_id", rec.get("scen_id"))
    run = rec.get("run_id", rec.get("seed_index"))
    return int(scene), int(run)


def load_jsonl(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[key_for(rec)] = rec
    return out


def wilson_ci(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = success / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def make_main_panel(ax):
    counts_8 = {"Random": 109, "Base Graph": 207, "Oracle": 221}
    counts_4 = {"Random": 37, "Base Graph": 68, "Oracle": 70}
    total = 300
    x = np.arange(len(METHODS))
    width = 0.34
    for offset, counts, label, color, hatch in [
        (-width / 2, counts_8, "SemTargetAcq@8m", "#4C78A8", ""),
        (width / 2, counts_4, "SemTargetAcq@4m", "#F58518", "///"),
    ]:
        vals = np.array([counts[m] / total * 100 for m in METHODS])
        err_low = []
        err_high = []
        for m in METHODS:
            lo, hi = wilson_ci(counts[m], total)
            val = counts[m] / total
            err_low.append((val - lo) * 100)
            err_high.append((hi - val) * 100)
        bars = ax.bar(
            x + offset,
            vals,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
            hatch=hatch,
            yerr=[err_low, err_high],
            capsize=2,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
        )
        for bar, val, upper in zip(bars, vals, err_high):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper + 1.4,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "boxstyle": "round,pad=0.08",
                    "alpha": 0.85,
                },
            )
    ax.set_xticks(x, METHODS)
    ax.set_ylabel("Semantic acquisition (%)")
    ax.set_ylim(0, 88)
    ax.legend(loc="upper left", fontsize=6, ncols=1, handlelength=1.5)
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.5)
    ax.set_axisbelow(True)


def make_threshold_panel(ax):
    radii = np.array([4, 6, 8, 10])
    vals = {
        "Random": [12.3, 33.3, 36.3, 36.7],
        "Base Graph": [22.7, 63.7, 69.0, 69.0],
        "Oracle": [23.3, 67.3, 73.7, 74.0],
    }
    markers = {"Random": "s", "Base Graph": "o", "Oracle": "^"}
    for method in METHODS:
        ax.plot(
            radii,
            vals[method],
            marker=markers[method],
            color=METHOD_COLORS[method],
            linewidth=1.4,
            markersize=4,
            label=method,
        )
    ax.set_xticks(radii)
    ax.set_xlabel("Success radius (m)")
    ax.set_ylabel("SemTargetAcq (%)")
    ax.set_ylim(0, 82)
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=6, handlelength=1.8)


def make_heatmap_panel(ax, random_log, ours_log, oracle_log, out_csv: Path):
    keys = sorted(set(random_log) & set(ours_log) & set(oracle_log))
    scenes = sorted({k[0] for k in keys})
    runs = sorted({k[1] for k in keys})
    scene_to_x = {s: i for i, s in enumerate(scenes)}
    run_to_y = {r: i for i, r in enumerate(runs)}

    matrix = np.full((len(runs), len(scenes)), np.nan)
    oracle_fail_x = []
    oracle_fail_y = []
    counts = Counter()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "run_id",
                "random_sem8",
                "ours_sem8",
                "oracle_sem8",
                "category",
            ],
        )
        writer.writeheader()
        for scene, run in keys:
            r_ok = sem_success(random_log[(scene, run)], 8)
            o_ok = sem_success(ours_log[(scene, run)], 8)
            oracle_ok = sem_success(oracle_log[(scene, run)], 8)
            if o_ok and r_ok:
                code, category = 3, "both"
            elif o_ok and not r_ok:
                code, category = 2, "ours_only"
            elif r_ok and not o_ok:
                code, category = 1, "random_only"
            else:
                code, category = 0, "neither"
            matrix[run_to_y[run], scene_to_x[scene]] = code
            counts[category] += 1
            if not oracle_ok:
                oracle_fail_x.append(scene_to_x[scene])
                oracle_fail_y.append(run_to_y[run])
            writer.writerow(
                {
                    "scene_id": scene,
                    "run_id": run,
                    "random_sem8": int(r_ok),
                    "ours_sem8": int(o_ok),
                    "oracle_sem8": int(oracle_ok),
                    "category": category,
                }
            )

    cmap = ListedColormap(["#ECEFF1", "#D9822B", "#2F6FA3", "#5AA469"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
    ax.scatter(
        oracle_fail_x,
        oracle_fail_y,
        marker="x",
        s=8,
        color="#202124",
        linewidths=0.45,
        alpha=0.75,
    )
    ax.set_xlabel("Protocol scene")
    ax.set_ylabel("Seed/run")
    ax.set_xticks(np.arange(0, len(scenes), 5), [str(s) for s in scenes[::5]])
    ax.set_yticks(np.arange(len(runs)), [str(r) for r in runs])
    ax.set_xticks(np.arange(-0.5, len(scenes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(runs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    legend_handles = [
        Patch(facecolor="#5AA469", edgecolor="none", label=f"Both ({counts['both']})"),
        Patch(facecolor="#2F6FA3", edgecolor="none", label=f"Base Graph only ({counts['ours_only']})"),
        Patch(facecolor="#D9822B", edgecolor="none", label=f"Random only ({counts['random_only']})"),
        Patch(facecolor="#ECEFF1", edgecolor="#B0B6BC", label=f"Neither ({counts['neither']})"),
        Line2D([0], [0], marker="x", linestyle="", color="#202124", markersize=4, label="Oracle fail"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=3,
        fontsize=5.7,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.8,
    )
    return counts


def make_decomposition_panel(ax):
    metrics = ["ProtoAcc@1", "SelectedTargetAcq", "SemTargetAcq@8m"]
    vals = {
        "Random": [48.3, 82.7, 36.3],
        "Base Graph": [93.3, 75.0, 69.0],
        "Oracle": [99.7, 73.7, 73.7],
    }
    x = np.arange(len(metrics))
    width = 0.23
    for idx, method in enumerate(METHODS):
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            vals[method],
            width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.35,
        )
        for bar, val in zip(bars, vals[method]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.6,
                rotation=90,
            )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, metrics, rotation=18, ha="right")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.4)
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.5)


def save_figure(fig, base: Path, dpi: int = 600):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/scaleup_20260520/main_30x10"))
    parser.add_argument("--out-base", type=Path, default=Path("paper/source/figures/fig4_closed_loop_evidence"))
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/fig4_closed_loop_evidence_source.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("runs/scaleup_20260520/main_30x10/fig4_closed_loop_evidence_summary.json"),
    )
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.frameon": False,
        }
    )

    random_log = load_jsonl(args.run_dir / "random_improved_v4_30x10.jsonl")
    ours_log = load_jsonl(args.run_dir / "gs_improved_v4_30x10.jsonl")
    oracle_log = load_jsonl(args.run_dir / "oracle_improved_v4_30x10.jsonl")

    fig = plt.figure(figsize=(7.16, 5.85), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.98,
        bottom=0.13,
        top=0.95,
        wspace=0.28,
        hspace=0.48,
        height_ratios=[1.0, 1.08],
    )
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, 0]),
        "d": fig.add_subplot(gs[1, 1]),
    }

    make_main_panel(axes["a"])
    make_threshold_panel(axes["b"])
    counts = make_heatmap_panel(axes["c"], random_log, ours_log, oracle_log, args.source_csv)
    make_decomposition_panel(axes["d"])

    titles = {
        "a": "Main closed-loop success",
        "b": "Threshold sensitivity",
        "c": "Paired scene-seed outcomes (30 x 10)",
        "d": "Selection/control decomposition",
    }
    for label, ax in axes.items():
        ax.text(
            -0.13,
            1.08,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_title(titles[label], fontsize=7.5, pad=5, fontweight="bold")

    save_figure(fig, args.out_base)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "paired_outcome_counts": dict(counts),
                "figure_base": str(args.out_base),
                "source_csv": str(args.source_csv),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(args.out_base.with_suffix(".pdf"))
    print(args.summary_json)


if __name__ == "__main__":
    main()
