#!/usr/bin/env python3
"""Generate the manuscript-quality qualitative case-study figure.

The figure intentionally shows two reviewer-relevant visual checks: a closed-loop
target/decoy behavior case and a same-graph Query-Flip spatial pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)


def load_episode(path: Path, scen_id: int, run_id: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("scen_id") == scen_id and record.get("run_id") == run_id:
                return record
    raise ValueError(f"Episode scen_id={scen_id}, run_id={run_id} not found in {path}")


def trajectory_xy(record: Dict[str, Any]) -> np.ndarray:
    points = []
    for item in record.get("trajectory", []):
        if "x" in item and "y" in item:
            points.append((float(item["x"]), float(item["y"])))
    if not points:
        raise ValueError("Episode has no trajectory points")
    return np.asarray(points, dtype=float)


def smooth_xy(points: np.ndarray, stride: int = 2) -> np.ndarray:
    if len(points) <= 2:
        return points
    sampled = points[::stride]
    if not np.allclose(sampled[-1], points[-1]):
        sampled = np.vstack([sampled, points[-1]])
    return sampled


def load_candidate_positions(protocol_path: Path, ego_start: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    data = np.load(protocol_path, allow_pickle=False)
    node_feats = data["node_feats"]
    target_idx = int(data["target_idx_gt"])
    candidate_indices = [int(x) for x in data["candidate_indices"]]
    decoy_indices = [idx for idx in candidate_indices if idx != target_idx]
    if not decoy_indices:
        raise ValueError(f"No decoy candidate found in {protocol_path}")
    decoy_idx = decoy_indices[0]

    # The first two node features are ego-centered metric coordinates.
    target_xy = ego_start + node_feats[target_idx, :2].astype(float)
    decoy_xy = ego_start + node_feats[decoy_idx, :2].astype(float)
    return target_xy, decoy_xy, target_idx, decoy_idx


def add_direct_label(ax: plt.Axes, xy: Iterable[float], text: str, dx: float, dy: float) -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=6.2,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.90},
        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.40", "shrinkA": 2, "shrinkB": 4},
    )


def plot_closed_loop_panel(
    ax: plt.Axes,
    gs_record: Dict[str, Any],
    random_record: Dict[str, Any],
    protocol_path: Path,
) -> None:
    ours = smooth_xy(trajectory_xy(gs_record), stride=2)
    random = smooth_xy(trajectory_xy(random_record), stride=2)
    start_abs = ours[0]
    target_abs, decoy_abs, _target_idx, _decoy_idx = load_candidate_positions(protocol_path, start_abs)

    # Use the ego-start frame to match the protocol-graph convention and keep
    # the manuscript panel compact and readable in one IEEE column.
    ours = ours - start_abs
    random = random - start_abs
    start = np.array([0.0, 0.0])
    target_xy = target_abs - start_abs
    decoy_xy = decoy_abs - start_abs

    blue = "#2F6B9A"
    orange = "#D9802E"
    gray = "#4D4D4D"

    ax.plot(
        ours[:, 0],
        ours[:, 1],
        color=blue,
        lw=1.25,
        solid_capstyle="round",
        label="Base Graph",
        zorder=3,
    )
    ax.plot(
        random[:, 0],
        random[:, 1],
        color=orange,
        lw=1.20,
        ls=(0, (3.4, 2.0)),
        solid_capstyle="round",
        label="Random",
        zorder=2,
    )

    ax.scatter(start[0], start[1], marker="^", s=28, color=gray, zorder=6)
    ax.scatter(target_xy[0], target_xy[1], marker="*", s=72, color="black", zorder=7)
    ax.scatter(
        decoy_xy[0],
        decoy_xy[1],
        marker="o",
        s=42,
        facecolor="white",
        edgecolor=orange,
        linewidth=1.0,
        zorder=7,
    )
    ax.scatter(ours[-1, 0], ours[-1, 1], marker="s", s=24, color=blue, zorder=8)
    ax.scatter(random[-1, 0], random[-1, 1], marker="s", s=24, color=orange, zorder=8)

    add_direct_label(ax, start, "start", -16, 10)
    add_direct_label(ax, target_xy, "target", -20, 12)
    add_direct_label(ax, decoy_xy, "decoy", 24, 7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.42, 1.0),
        ncol=2,
        fontsize=6.0,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.2,
    )

    all_xy = np.vstack([ours, random, target_xy[None, :], decoy_xy[None, :], start[None, :]])
    pad_x = max(2.5, 0.12 * (all_xy[:, 0].max() - all_xy[:, 0].min()))
    pad_y = max(2.5, 0.12 * (all_xy[:, 1].max() - all_xy[:, 1].min()))
    ax.set_xlim(all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
    ax.set_ylim(all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("(a) Closed-loop target/decoy behavior", fontsize=7.2, fontweight="bold", pad=3.0)
    ax.set_xlabel("Relative x (m)")
    ax.set_ylabel("Relative y (m)")
    ax.grid(True, color="0.92", lw=0.40)
    ax.tick_params(labelsize=6.0, length=2.2)


def _pair_key(task_id: str) -> str:
    if "_" in task_id:
        return task_id.rsplit("_", 1)[0]
    return str(int(task_id) // 2)


def load_queryflip_pair(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_pair.setdefault(_pair_key(str(rec.get("task_id", rec.get("id", "")))), []).append(rec)
    for pair in by_pair.values():
        if len(pair) < 2:
            continue
        a, b = pair[0], pair[1]
        if int(a["target_idx_gt"]) != int(b["target_idx_gt"]):
            return a, b
    raise ValueError(f"No valid Query-Flip pair with different targets found in {path}")


def query_label(query_type: str) -> str:
    return (
        query_type.replace("Q_", "")
        .replace("_", " ")
        .replace("SAME LANE FRONT", "same-lane front")
        .replace("LEFT ADJ FRONT", "left-adjacent front")
        .lower()
    )


def candidate_xy_from_distance_bearing(task: Dict[str, Any]) -> np.ndarray:
    dists = np.asarray(task["candidate_dists"], dtype=float)
    bearings = np.deg2rad(np.asarray(task["candidate_bearings"], dtype=float))
    return np.column_stack([dists * np.cos(bearings), dists * np.sin(bearings)])


def plot_queryflip_panel(ax: plt.Axes, task_a: Dict[str, Any], task_b: Dict[str, Any]) -> None:
    xy = candidate_xy_from_distance_bearing(task_a)
    target_a = int(task_a["target_idx_gt"])
    target_b = int(task_b["target_idx_gt"])
    blue = "#2F6B9A"
    orange = "#D9802E"
    gray = "#4D4D4D"

    ax.scatter(0.0, 0.0, marker="^", s=32, color=gray, zorder=6)
    add_direct_label(ax, (0.0, 0.0), "ego/anchor", 24, 12)

    for idx, point in enumerate(xy):
        if idx == target_a:
            ax.scatter(point[0], point[1], marker="*", s=82, color=blue, edgecolor="black", linewidth=0.4, zorder=8)
            add_direct_label(ax, point, "target A", -18, 14)
        elif idx == target_b:
            ax.scatter(point[0], point[1], marker="D", s=48, color=orange, edgecolor="black", linewidth=0.4, zorder=8)
            add_direct_label(ax, point, "target B", 24, 10)
        else:
            ax.scatter(point[0], point[1], marker="o", s=34, facecolor="white", edgecolor="0.35", linewidth=0.8, zorder=7)
            add_direct_label(ax, point, f"c{idx}", 16, -9)

    ax.plot([0, xy[target_a, 0]], [0, xy[target_a, 1]], color=blue, lw=0.9, alpha=0.70)
    ax.plot([0, xy[target_b, 0]], [0, xy[target_b, 1]], color=orange, lw=0.9, ls=(0, (3, 2)), alpha=0.85)
    ax.text(
        0.02,
        0.06,
        f"A: {query_label(task_a['query_type'])}\nB: {query_label(task_b['query_type'])}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.85", "lw": 0.4, "alpha": 0.94},
    )

    all_xy = np.vstack([xy, np.array([[0.0, 0.0]])])
    pad_x = max(2.5, 0.16 * (all_xy[:, 0].max() - all_xy[:, 0].min()))
    pad_y = max(2.5, 0.20 * (all_xy[:, 1].max() - all_xy[:, 1].min()))
    ax.set_xlim(all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
    ax.set_ylim(all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(b) Same graph, different query targets", fontsize=7.2, fontweight="bold", pad=3.0)
    ax.set_xlabel("Anchor-frame x (m)")
    ax.set_ylabel("Anchor-frame y (m)")
    ax.grid(True, color="0.92", lw=0.40)
    ax.tick_params(labelsize=6.0, length=2.2)


def plot_case(
    gs_record: Dict[str, Any],
    random_record: Dict[str, Any],
    protocol_path: Path,
    queryflip_path: Path,
    output_base: Path,
) -> None:
    task_a, task_b = load_queryflip_pair(queryflip_path)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), constrained_layout=True)
    plot_closed_loop_panel(axes[0], gs_record, random_record, protocol_path)
    plot_queryflip_panel(axes[1], task_a, task_b)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_legacy_case(
    gs_record: Dict[str, Any],
    random_record: Dict[str, Any],
    protocol_path: Path,
    output_base: Path,
) -> None:
    """Optional single-panel export retained for internal checks."""
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    plot_closed_loop_panel(ax, gs_record, random_record, protocol_path)
    ax.set_title("")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=6.1,
        handlelength=2.2,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    ax.set_xlabel("Relative x (m)")
    ax.set_ylabel("Relative y (m)")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scen-id", type=int, default=13)
    parser.add_argument("--run-id", type=int, default=3)
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("runs/expanded_controller_20260520/v4_full/gs_improved_v4_30x5.jsonl"),
    )
    parser.add_argument(
        "--random",
        type=Path,
        default=Path("runs/expanded_controller_20260520/v4_full/random_improved_v4_30x5.jsonl"),
    )
    parser.add_argument("--protocol-dir", type=Path, default=Path("data/protocol_graphs"))
    parser.add_argument(
        "--queryflip",
        type=Path,
        default=Path("data/semantic_queryflip/scenarios_queryflip_val_group.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper/source/figures/case_study_semantic_target_scen_013"),
    )
    args = parser.parse_args()

    gs_record = load_episode(args.gs, args.scen_id, args.run_id)
    random_record = load_episode(args.random, args.scen_id, args.run_id)
    protocol_path = args.protocol_dir / f"scen_{args.scen_id:03d}.npz"
    plot_case(gs_record, random_record, protocol_path, args.queryflip, args.out)

    print(f"Wrote {args.out.with_suffix('.pdf')}")
    print(f"Wrote {args.out.with_suffix('.png')}")
    print(f"Wrote {args.out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
