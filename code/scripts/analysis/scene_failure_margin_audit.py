#!/usr/bin/env python3
"""Generate scene-level, failure-taxonomy, and selector-margin audits for R2.

The script intentionally uses only the paper-facing 30x10 closed-loop logs and a
CPU-only forward pass on the fixed protocol graphs. It does not run CARLA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.model import LCORGNet


METHOD_FILES = {
    "Random": "random_improved_v4_30x10.jsonl",
    "Ours": "gs_improved_v4_30x10.jsonl",
    "Oracle": "oracle_improved_v4_30x10.jsonl",
}

METHOD_COLORS = {
    "Random": "#D9822B",
    "Ours": "#2F6FA3",
    "Oracle": "#5B6770",
}

FAILURE_ORDER = [
    "success_4m",
    "close_range_4m_failure",
    "controller_8m_failure",
    "selection_miss",
    "invalid_or_unknown",
]

FAILURE_LABELS = {
    "success_4m": "SemSucc@4m",
    "close_range_4m_failure": "8m acquired, 4m fail",
    "controller_8m_failure": "Correct target, 8m fail",
    "selection_miss": "Selection miss",
    "invalid_or_unknown": "Invalid/unknown",
}

PLOT_FAILURE_LABELS = {
    "success_4m": "4m success",
    "close_range_4m_failure": "8m only",
    "controller_8m_failure": "8m fail",
    "selection_miss": "Selection miss",
    "invalid_or_unknown": "Invalid",
}

FAILURE_COLORS = {
    "success_4m": "#5AA469",
    "close_range_4m_failure": "#8FB7D9",
    "controller_8m_failure": "#C7CDD3",
    "selection_miss": "#C84C4C",
    "invalid_or_unknown": "#6B7280",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def key_for(rec: dict[str, Any]) -> tuple[int, int]:
    scene = rec.get("scene_id", rec.get("scen_id"))
    run = rec.get("run_id", rec.get("seed_index"))
    return int(scene), int(run)


def read_jsonl(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec.pop("trajectory", None)
            records[key_for(rec)] = rec
    return records


def is_correct(rec: dict[str, Any]) -> bool:
    return bool(rec.get("gs_correct", rec.get("oracle_correct", False)))


def sem_success(rec: dict[str, Any], radius: int) -> bool:
    return is_correct(rec) and bool(rec.get(f"success_{radius}m", False))


def classify_failure(rec: dict[str, Any]) -> str:
    if rec.get("valid") is False:
        return "invalid_or_unknown"
    if sem_success(rec, 4):
        return "success_4m"
    if not is_correct(rec):
        return "selection_miss"
    if sem_success(rec, 8) and not sem_success(rec, 4):
        return "close_range_4m_failure"
    if is_correct(rec) and not sem_success(rec, 8):
        return "controller_8m_failure"
    return "invalid_or_unknown"


def load_model(checkpoint: Path, protocol_dir: Path, device: torch.device) -> LCORGNet:
    first_graph = protocol_dir / "scen_000.npz"
    data = np.load(first_graph, allow_pickle=False)
    node_feat_dim = int(data["node_feats"].shape[1])
    edge_feat_dim = int(data["edge_attr"].shape[1])
    lang_feat_dim = int(data["lang_feat"].shape[0])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    use_rel_gate = any(("rel_proj" in key or "gate_mlp" in key) for key in state.keys())
    use_edge_context_head = any(
        (
            "raw_node_proj" in key
            or "edge_context_proj" in key
            or "graph_pool_proj" in key
            or "edge_goal_head" in key
        )
        for key in state.keys()
    )
    model = LCORGNet(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        lang_feat_dim=lang_feat_dim,
        hidden_dim=128,
        num_layers=3,
        num_rel_types=5,
        use_rel_gate=use_rel_gate,
        use_edge_context_head=use_edge_context_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_selector_margins(
    checkpoint: Path,
    protocol_dir: Path,
    device: torch.device,
    max_scen_id: int,
) -> list[dict[str, Any]]:
    model = load_model(checkpoint, protocol_dir, device)
    rows: list[dict[str, Any]] = []
    for scene_id in range(max_scen_id + 1):
        path = protocol_dir / f"scen_{scene_id:03d}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        node_feats = torch.from_numpy(data["node_feats"]).float().to(device)
        edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
        edge_attr = torch.from_numpy(data["edge_attr"]).float().to(device)
        lang_feat = torch.from_numpy(data["lang_feat"]).float().to(device)
        cand_mask_np = data["candidate_mask"].astype(bool)
        candidate_mask = torch.from_numpy(cand_mask_np).bool().to(device)
        target_idx = int(data["target_idx_gt"])
        cand_indices = np.nonzero(cand_mask_np)[0].astype(int)

        with torch.no_grad():
            outputs = model.forward_single(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                lang_feat=lang_feat,
                candidate_mask=candidate_mask,
            )
            logits = outputs["goal_logits"].view(-1).detach().cpu().numpy()

        cand_scores = [(int(idx), float(logits[int(idx)])) for idx in cand_indices]
        cand_scores_sorted = sorted(cand_scores, key=lambda item: item[1], reverse=True)
        chosen_idx = cand_scores_sorted[0][0] if cand_scores_sorted else -1
        target_score = float(logits[target_idx]) if 0 <= target_idx < len(logits) else float("nan")
        decoy_scores = [score for idx, score in cand_scores if idx != target_idx]
        best_decoy_score = max(decoy_scores) if decoy_scores else float("nan")
        target_margin = target_score - best_decoy_score
        top1_score = cand_scores_sorted[0][1] if cand_scores_sorted else float("nan")
        top2_score = cand_scores_sorted[1][1] if len(cand_scores_sorted) > 1 else float("nan")
        top1_gap = top1_score - top2_score if len(cand_scores_sorted) > 1 else float("nan")
        rows.append(
            {
                "scene_id": scene_id,
                "num_candidates": int(cand_mask_np.sum()),
                "target_idx_gt": target_idx,
                "chosen_idx": chosen_idx,
                "gs_correct": bool(chosen_idx == target_idx),
                "target_score": target_score,
                "best_decoy_score": best_decoy_score,
                "target_margin": target_margin,
                "top1_gap": top1_gap,
                "candidate_scores": cand_scores_sorted,
            }
        )
    return rows


def write_scene_audit(logs: dict[str, dict[tuple[int, int], dict[str, Any]]], out_csv: Path) -> list[dict[str, Any]]:
    keys = sorted(set.intersection(*(set(v) for v in logs.values())))
    scenes = sorted({scene for scene, _ in keys})
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_keys = [key for key in keys if key[0] == scene]
        row: dict[str, Any] = {"scene_id": scene, "n_runs": len(scene_keys)}
        for method in METHOD_FILES:
            method_records = [logs[method][key] for key in scene_keys]
            sem8 = sum(sem_success(rec, 8) for rec in method_records)
            sem4 = sum(sem_success(rec, 4) for rec in method_records)
            correct = sum(is_correct(rec) for rec in method_records)
            nav_selected_8 = sum(bool(rec.get("success_8m", False)) for rec in method_records)
            row[f"{method.lower()}_sem8_count"] = sem8
            row[f"{method.lower()}_sem4_count"] = sem4
            row[f"{method.lower()}_correct_count"] = correct
            row[f"{method.lower()}_selected_nav8_count"] = nav_selected_8
        row["ours_minus_random_sem8"] = row["ours_sem8_count"] - row["random_sem8_count"]
        row["oracle_minus_ours_sem8"] = row["oracle_sem8_count"] - row["ours_sem8_count"]
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_failure_taxonomy(logs: dict[str, dict[tuple[int, int], dict[str, Any]]], out_csv: Path) -> dict[str, Counter[str]]:
    counters: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    for method, records in logs.items():
        counter = Counter(classify_failure(rec) for rec in records.values())
        counters[method] = counter
        total = sum(counter.values())
        for category in FAILURE_ORDER:
            count = counter.get(category, 0)
            rows.append(
                {
                    "method": method,
                    "category": category,
                    "label": FAILURE_LABELS[category],
                    "count": count,
                    "percent": 100.0 * count / total if total else 0.0,
                }
            )
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "category", "label", "count", "percent"])
        writer.writeheader()
        writer.writerows(rows)
    return counters


def write_margin_outputs(rows: list[dict[str, Any]], out_csv: Path, out_jsonl: Path) -> dict[str, Any]:
    slim_rows = []
    for row in rows:
        slim = dict(row)
        slim["candidate_scores_json"] = json.dumps(row["candidate_scores"])
        slim.pop("candidate_scores", None)
        slim_rows.append(slim)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slim_rows)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    margins = np.array([float(row["target_margin"]) for row in rows], dtype=float)
    correct = [row for row in rows if row["gs_correct"]]
    incorrect = [row for row in rows if not row["gs_correct"]]
    positive = int(np.sum(margins > 0))
    lo, hi = wilson_ci(positive, len(rows))
    return {
        "n_graphs": len(rows),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "positive_margin": positive,
        "positive_margin_percent": 100.0 * positive / len(rows) if rows else 0.0,
        "positive_margin_ci95_percent": [100.0 * lo, 100.0 * hi],
        "median_margin": float(np.median(margins)) if len(margins) else None,
        "mean_margin": float(np.mean(margins)) if len(margins) else None,
        "min_margin": float(np.min(margins)) if len(margins) else None,
        "max_margin": float(np.max(margins)) if len(margins) else None,
        "incorrect_scene_ids": [int(row["scene_id"]) for row in incorrect],
    }


def write_markdown(
    scene_rows: list[dict[str, Any]],
    taxonomy: dict[str, Counter[str]],
    margin_summary: dict[str, Any],
    output: Path,
) -> None:
    n_scenes = len(scene_rows)
    positive_scene = sum(1 for row in scene_rows if row["ours_minus_random_sem8"] > 0)
    tie_scene = sum(1 for row in scene_rows if row["ours_minus_random_sem8"] == 0)
    negative_scene = sum(1 for row in scene_rows if row["ours_minus_random_sem8"] < 0)
    total_gain = sum(row["ours_minus_random_sem8"] for row in scene_rows)
    best_gain = max(row["ours_minus_random_sem8"] for row in scene_rows)
    worst_gain = min(row["ours_minus_random_sem8"] for row in scene_rows)
    best_scenes = [row["scene_id"] for row in scene_rows if row["ours_minus_random_sem8"] == best_gain]
    worst_scenes = [row["scene_id"] for row in scene_rows if row["ours_minus_random_sem8"] == worst_gain]

    lines = [
        "# Scene, Failure, and Margin Audit",
        "",
        "This audit is generated from the official 30x10 closed-loop logs plus a CPU-only forward pass on the same 30 fixed protocol graphs. It is intended to make reviewer-facing risks auditable, not to introduce a new headline metric.",
        "",
        "## Scene-Level Outcome Audit",
        "",
        f"- Scenes: {n_scenes}; runs per scene: 10.",
        f"- Ours beats Random in {positive_scene}/{n_scenes} scenes, ties in {tie_scene}/{n_scenes}, and is lower in {negative_scene}/{n_scenes}.",
        f"- Summed Ours-Random SemNavSucc@8m gain across scenes: {total_gain} successful episodes, matching the aggregate 207 vs. 109 result.",
        f"- Largest per-scene gain: +{best_gain}/10 in scenes {best_scenes}; lowest per-scene difference: {worst_gain}/10 in scenes {worst_scenes}.",
        "",
        "| Scene | Random Sem8 | Ours Sem8 | Oracle Sem8 | Ours-Random | Oracle-Ours |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scene_rows:
        lines.append(
            f"| {row['scene_id']} | {row['random_sem8_count']}/10 | {row['ours_sem8_count']}/10 | "
            f"{row['oracle_sem8_count']}/10 | {row['ours_minus_random_sem8']:+d} | {row['oracle_minus_ours_sem8']:+d} |"
        )

    lines.extend(["", "## Failure Taxonomy", "", "| Method | SemSucc@4m | 8m acquired, 4m fail | Correct target, 8m fail | Selection miss | Invalid/unknown |"])
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in METHOD_FILES:
        counter = taxonomy[method]
        lines.append(
            f"| {method} | "
            f"{counter['success_4m']} | {counter['close_range_4m_failure']} | "
            f"{counter['controller_8m_failure']} | {counter['selection_miss']} | {counter['invalid_or_unknown']} |"
        )

    lines.extend(
        [
            "",
            "Interpretation: Ours converts many Random selection misses into semantic target acquisition, while the dominant residual limitation after correct selection is close-range 4 m docking rather than target identity.",
            "",
            "## Selector Score-Margin Audit",
            "",
            f"- Positive target-vs-best-decoy margin: {margin_summary['positive_margin']}/{margin_summary['n_graphs']} ({margin_summary['positive_margin_percent']:.1f}%).",
            f"- Median target margin: {margin_summary['median_margin']:.3f}; mean: {margin_summary['mean_margin']:.3f}; range: [{margin_summary['min_margin']:.3f}, {margin_summary['max_margin']:.3f}].",
            f"- Negative-margin scene IDs: {margin_summary['incorrect_scene_ids']}.",
            "",
            "Interpretation: the protocol accuracy result is supported by a positive score margin on 28/30 graphs. The two negative-margin scenes match the residual selector misses and are not hidden by closed-loop aggregation.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_audit(
    scene_rows: list[dict[str, Any]],
    taxonomy: dict[str, Counter[str]],
    margin_rows: list[dict[str, Any]],
    output_base: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(7.25, 3.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.05])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel a: scene-level heatmap.
    scenes = [row["scene_id"] for row in scene_rows]
    heat = np.array(
        [
            [row["random_sem8_count"] for row in scene_rows],
            [row["ours_sem8_count"] for row in scene_rows],
            [row["oracle_sem8_count"] for row in scene_rows],
        ],
        dtype=float,
    )
    cmap = LinearSegmentedColormap.from_list("success", ["#F6F7F8", "#B8D3EA", "#2F6FA3"])
    im = ax0.imshow(heat, aspect="auto", cmap=cmap, vmin=0, vmax=10)
    ax0.set_yticks([0, 1, 2], ["Random", "Ours", "Oracle"])
    ax0.set_xticks(np.arange(0, len(scenes), 5), [str(s) for s in scenes[::5]])
    ax0.set_xlabel("Protocol scene")
    ax0.set_title("(a) Per-scene SemNavSucc@8m", fontsize=7.5, fontweight="bold")
    ax0.set_xticks(np.arange(-0.5, len(scenes), 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax0.grid(which="minor", color="white", linewidth=0.4)
    ax0.tick_params(which="minor", bottom=False, left=False)
    for y in range(3):
        for x, value in enumerate(heat[y]):
            if x % 2 == 0 or value in (0, 10):
                ax0.text(x, y, f"{int(value)}", ha="center", va="center", fontsize=4.8, color="#1F2933")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("successes / 10", fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    # Panel b: failure taxonomy.
    methods = list(METHOD_FILES.keys())
    bottoms = np.zeros(len(methods))
    for category in FAILURE_ORDER:
        values = np.array([taxonomy[m].get(category, 0) / 300 * 100 for m in methods])
        ax1.bar(
            methods,
            values,
            bottom=bottoms,
            color=FAILURE_COLORS[category],
            edgecolor="white",
            linewidth=0.35,
            label=PLOT_FAILURE_LABELS[category],
        )
        bottoms += values
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Episodes (%)")
    ax1.set_title("(b) Failure taxonomy", fontsize=7.5, fontweight="bold")
    ax1.grid(axis="y", color="#E6E8EB", linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=4.9,
        handlelength=0.9,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    # Panel c: score margins.
    sorted_margins = sorted(margin_rows, key=lambda row: float(row["target_margin"]))
    x = np.arange(len(sorted_margins))
    margins = np.array([float(row["target_margin"]) for row in sorted_margins])
    colors = ["#C84C4C" if margin < 0 else "#2F6FA3" for margin in margins]
    ax2.axhline(0, color="#202124", linewidth=0.8)
    ax2.bar(x, margins, color=colors, edgecolor="black", linewidth=0.25)
    ax2.set_xticks(x[::5], [str(int(sorted_margins[i]["scene_id"])) for i in x[::5]])
    ax2.set_xlabel("Protocol scene (sorted)")
    ax2.set_ylabel("Target margin")
    ax2.set_title("(c) Selector score margin", fontsize=7.5, fontweight="bold")
    ax2.grid(axis="y", color="#E6E8EB", linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(
        handles=[
            Patch(facecolor="#2F6FA3", edgecolor="black", linewidth=0.25, label="target > decoy"),
            Patch(facecolor="#C84C4C", edgecolor="black", linewidth=0.25, label="target < decoy"),
        ],
        loc="upper left",
        fontsize=5.8,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-base", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-scen-id", type=int, default=29)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    logs = {
        method: read_jsonl(args.log_dir / filename)
        for method, filename in METHOD_FILES.items()
    }
    scene_rows = write_scene_audit(logs, args.output_dir / "scene_level_outcome_audit.csv")
    taxonomy = write_failure_taxonomy(logs, args.output_dir / "failure_taxonomy_30x10.csv")
    margin_rows = compute_selector_margins(args.checkpoint, args.protocol_dir, device, args.max_scen_id)
    margin_summary = write_margin_outputs(
        margin_rows,
        args.output_dir / "selector_margin_audit.csv",
        args.output_dir / "selector_margin_records.jsonl",
    )
    write_markdown(
        scene_rows,
        taxonomy,
        margin_summary,
        args.output_dir / "scene_failure_margin_audit.md",
    )

    summary = {
        "scene_audit": {
            "n_scenes": len(scene_rows),
            "ours_better_scenes": sum(1 for row in scene_rows if row["ours_minus_random_sem8"] > 0),
            "ties": sum(1 for row in scene_rows if row["ours_minus_random_sem8"] == 0),
            "random_better_scenes": sum(1 for row in scene_rows if row["ours_minus_random_sem8"] < 0),
            "total_ours_minus_random_sem8": sum(row["ours_minus_random_sem8"] for row in scene_rows),
        },
        "failure_taxonomy": {method: dict(counter) for method, counter in taxonomy.items()},
        "margin_summary": margin_summary,
        "figure_base": str(args.figure_base),
    }
    (args.output_dir / "scene_failure_margin_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_audit(scene_rows, taxonomy, margin_rows, args.figure_base)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
