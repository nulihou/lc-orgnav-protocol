#!/usr/bin/env python3
"""Regenerate Query-Flip validation figures from local paper-facing artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PALETTE = {
    "ours": "#1f77b4",
    "noflip": "#ff7f0e",
    "noquery": "#d62728",
    "geometry": "#555555",
    "swap": "#777777",
    "success": "#1f77b4",
    "chance": "#fff2cc",
    "fail": "#eeeeee",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def switch_rate(result: dict) -> float:
    rows = result.get("results", [])
    if not rows:
        return 0.0
    switched = 0
    for row in rows:
        if "switched" in row:
            switched += int(bool(row["switched"]))
        else:
            switched += int(row.get("task_a_chosen") != row.get("task_b_chosen"))
    return 100.0 * switched / len(rows)


def metrics_from_result(result: dict) -> tuple[float, float, float]:
    return (
        float(result["accuracy"]),
        float(result["both_correct_rate"]),
        switch_rate(result),
    )


def both_correct_count(result: dict) -> tuple[int, int]:
    n = int(result.get("num_scenarios", len(result.get("results", []))))
    if "both_correct_count" in result:
        k = int(result["both_correct_count"])
    else:
        k = int(round(float(result["both_correct_rate"]) * n / 100.0))
    return k, n


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (centre - margin) / denom, (centre + margin) / denom


def yerr_wilson(results: list[dict]) -> tuple[list[float], np.ndarray]:
    y = []
    low_err = []
    high_err = []
    for result in results:
        k, n = both_correct_count(result)
        low, high = wilson_ci(k, n)
        rate = 100.0 * k / n
        y.append(rate)
        low_err.append(rate - 100.0 * low)
        high_err.append(100.0 * high - rate)
    return y, np.asarray([low_err, high_err])


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
        }
    )


def save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_matrix(ax: plt.Axes, rows: list[str], cols: list[str], values: np.ndarray) -> None:
    color_code = np.zeros_like(values, dtype=float)
    color_code[values >= 90.0] = 2.0
    color_code[(values > 10.0) & (values < 90.0)] = 1.0
    cmap = ListedColormap([PALETTE["fail"], PALETTE["chance"], PALETTE["success"]])
    ax.imshow(color_code, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=2)

    ax.set_xticks(np.arange(len(cols)), cols, fontsize=7.2, fontweight="bold")
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=7.2, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            val = values[r, c]
            text_color = "white" if val >= 90.0 else "black"
            ax.text(
                c,
                r,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="bold" if val >= 90.0 else "normal",
                color=text_color,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fig5(results_dir: Path, no_flip_dir: Path, out_base: Path) -> None:
    normal = load_json(results_dir / "normal_v1.json")
    noquery = load_json(results_dir / "noquery_v1.json")
    queryswap = load_json(results_dir / "queryswap_v1.json")
    geometry = load_json(results_dir / "queryflip_v1_geom_only.json")
    progression = {
        "v1": load_json(results_dir / "normal_v1.json"),
        "v2": load_json(results_dir / "normal_v2.json"),
        "Hard": load_json(results_dir / "hard_eval.json"),
        "Ultra": load_json(results_dir / "ultrahard_eval.json"),
    }
    noquery_progression = {
        "v1": load_json(results_dir / "noquery_v1.json"),
        "v2": load_json(results_dir / "noquery_v1.json"),
        "Hard": load_json(results_dir / "noquery_v1.json"),
        "Ultra": load_json(results_dir / "noquery_v1.json"),
    }
    geometry_progression = {
        "v1": load_json(results_dir / "queryflip_v1_geom_only.json"),
        "v2": load_json(results_dir / "queryflip_v1_geom_only.json"),
        "Hard": load_json(results_dir / "queryflip_hard_geom_only.json"),
        "Ultra": load_json(results_dir / "queryflip_ultra_hard_geom_only.json"),
    }
    no_flip_progression = {
        "v1": load_json(no_flip_dir / "no_flip_v1.json"),
        "v2": load_json(no_flip_dir / "no_flip_v2.json"),
        "Hard": load_json(no_flip_dir / "no_flip_hard.json"),
        "Ultra": load_json(no_flip_dir / "no_flip_ultra_hard.json"),
    }

    rows = ["Full model", "No-Query", "Query-Swap", "Geometry-only"]
    cols = ["Accuracy", "Both-Corr.", "Switch-Rate"]
    values = np.asarray(
        [
            metrics_from_result(normal),
            metrics_from_result(noquery),
            metrics_from_result(queryswap),
            metrics_from_result(geometry),
        ]
    )

    fig = plt.figure(figsize=(6.8, 3.15))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, bottom=0.20, top=0.82, wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    draw_matrix(ax0, rows, cols, values)
    ax0.set_title("(a) Counterfactual Audit Matrix", fontsize=9, fontweight="bold", pad=24)

    splits = list(progression)
    x = np.arange(len(splits), dtype=float)
    full_results = [progression[s] for s in splits]
    no_flip_results = [no_flip_progression[s] for s in splits]
    ours_y, ours_err = yerr_wilson(full_results)
    noflip_y, noflip_err = yerr_wilson(no_flip_results)
    geometry_y = [geometry_progression[s]["both_correct_rate"] for s in splits]
    xticklabels = ["Split-A\nN=15", "Split-B\nN=15", "Hard\nN=21", "Ultra\nN=54"]

    ax1.errorbar(
        x,
        ours_y,
        yerr=ours_err,
        marker="o",
        markersize=4.6,
        color=PALETTE["ours"],
        lw=1.8,
        elinewidth=0.8,
        capsize=2.5,
        label="Full model",
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax1.errorbar(
        x,
        noflip_y,
        yerr=noflip_err,
        marker="D",
        markersize=4.0,
        color=PALETTE["noflip"],
        ls="--",
        lw=1.45,
        elinewidth=0.8,
        capsize=2.5,
        label="w/o Flip Loss",
        markeredgecolor="black",
        markeredgewidth=0.45,
    )
    ax1.plot(
        x,
        geometry_y,
        marker="^",
        markersize=4.0,
        color=PALETTE["geometry"],
        ls=":",
        lw=1.2,
        label="Geometry-only",
        markeredgecolor="black",
        markeredgewidth=0.45,
    )
    ax1.axhspan(-2, 8, color="#fff5f5", alpha=0.75, zorder=0)
    ax1.text(1.5, 6.2, "Geometry collapse at 0%", color=PALETTE["geometry"], fontsize=6.2, ha="center")
    ax1.set_title("(b) Robustness and Flip-Loss Ablation", fontsize=9, fontweight="bold", pad=24)
    ax1.set_xticks(x, xticklabels)
    ax1.set_ylim(-10, 112)
    ax1.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax1.set_ylabel("Both-Correct Rate (%)")
    ax1.grid(axis="y", color="#eeeeee", linewidth=0.5)
    ax1.text(0.02, 0.97, "Error bars: 95% Wilson CI", transform=ax1.transAxes, ha="left", va="top", fontsize=5.9, color="#555555")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6.2)

    save_all(fig, out_base)


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def plot_fig6(stats_json: Path, out_base: Path) -> None:
    stats = load_json(stats_json)
    pairs = stats["all_pairs"]
    dist = [float(p["min_dist_diff"]) for p in pairs]
    bearing = [float(p["min_bearing_diff"]) for p in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.25))
    panels = [
        (axes[0], dist, "Min $\\Delta$distance (m)", "(a) Distance Similarity Distribution", [1.0, 1.5, 2.0], "#003366"),
        (axes[1], bearing, "Min $\\Delta$bearing (deg)", "(b) Angle Similarity Distribution", [5.0, 8.0, 10.0], "#D35400"),
    ]
    for ax, values, xlabel, title, thresholds, color in panels:
        x, y = ecdf(values)
        ax.step(x, y, where="post", lw=1.6, color=color)
        for thr in thresholds:
            ax.axvline(thr, color="#C0392B", ls="--", lw=0.75, alpha=0.45)
            suffix = "m" if "distance" in xlabel else "$^\\circ$"
            ax.text(
                thr,
                1.025,
                f"{thr:g}{suffix}",
                ha="center",
                va="bottom",
                color="#C0392B",
                fontsize=6.6,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.1},
            )
        med = float(np.median(values))
        ax.axvline(med, color="black", ls="-", lw=0.7, alpha=0.55)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.12)
        ax.set_xlim(0, max(max(values) * 1.1, max(thresholds) * 1.05))
        ax.grid(True, color="#eeeeee", linewidth=0.45)
    axes[0].set_ylabel("Cumulative Probability")
    fig.tight_layout(w_pad=2.1)
    save_all(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queryflip-results",
        type=Path,
        default=Path("artifacts/submission_v1/paper_artifacts_v1/semantic/queryflip/results"),
    )
    parser.add_argument(
        "--difficulty-stats",
        type=Path,
        default=Path("artifacts/submission_v1/paper_artifacts_v1/semantic/queryflip/results/difficulty_stats.json"),
    )
    parser.add_argument(
        "--no-flip-results",
        type=Path,
        default=Path("runs/full_rerun_20260520/training/core_ablations/no_flip_loss"),
    )
    parser.add_argument(
        "--fig5-out",
        type=Path,
        default=Path("paper/source/figures/fig5_queryflip_panel"),
    )
    parser.add_argument(
        "--fig6-out",
        type=Path,
        default=Path("paper/source/figures/fig6_difficulty"),
    )
    args = parser.parse_args()
    configure_style()
    plot_fig5(args.queryflip_results, args.no_flip_results, args.fig5_out)
    plot_fig6(args.difficulty_stats, args.fig6_out)
    print(args.fig5_out.with_suffix(".pdf"))
    print(args.fig6_out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
