#!/usr/bin/env python3
"""Protocol goal-selection robustness under controlled feature perturbations.

This is a static selector diagnostic, not a closed-loop CARLA rerun. It keeps
the protocol graph, query, target label, and candidate set fixed, then injects
meter-scale Gaussian perturbations into the relative position/distance feature
blocks used by the selector. The goal is to check whether protocol accuracy is
fragile to small numeric feature perturbations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.model import LCORGNet


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def normalize_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    return state


def load_model(checkpoint: Path, protocol_dir: Path, device: torch.device) -> LCORGNet:
    sample = np.load(protocol_dir / "scen_000.npz", allow_pickle=False)
    state = normalize_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
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
        node_feat_dim=int(sample["node_feats"].shape[1]),
        edge_feat_dim=int(sample["edge_attr"].shape[1]),
        lang_feat_dim=int(sample["lang_feat"].shape[0]),
        hidden_dim=128,
        num_layers=3,
        num_rel_types=5,
        use_rel_gate=use_rel_gate,
        use_edge_context_head=use_edge_context_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def perturb_graph(data: np.lib.npyio.NpzFile, sigma_m: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    node_feats = data["node_feats"].astype(np.float32).copy()
    edge_attr = data["edge_attr"].astype(np.float32).copy()
    if sigma_m <= 0:
        return node_feats, edge_attr

    # Protocol graphs store relative x/y/distance twice in the 32-D node vector:
    # [0,1,2] and [5,6,7]. Perturb x/y and recompute the associated distance.
    for x_col, y_col, d_col in ((0, 1, 2), (5, 6, 7)):
        noise = rng.normal(0.0, sigma_m, size=(node_feats.shape[0], 2)).astype(np.float32)
        node_feats[:, x_col] += noise[:, 0]
        node_feats[:, y_col] += noise[:, 1]
        node_feats[:, d_col] = np.sqrt(node_feats[:, x_col] ** 2 + node_feats[:, y_col] ** 2)

    # Edge attributes use relative dx/dy/distance in columns [0,1,2].
    edge_noise = rng.normal(0.0, sigma_m, size=(edge_attr.shape[0], 2)).astype(np.float32)
    edge_attr[:, 0] += edge_noise[:, 0]
    edge_attr[:, 1] += edge_noise[:, 1]
    edge_attr[:, 2] = np.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2)
    return node_feats, edge_attr


def predict(model: LCORGNet, data: np.lib.npyio.NpzFile, node_feats_np: np.ndarray, edge_attr_np: np.ndarray, device: torch.device) -> int:
    candidate_mask_np = data["candidate_mask"].astype(bool)
    cand_indices = np.nonzero(candidate_mask_np)[0]
    if len(cand_indices) == 0:
        return -1
    with torch.no_grad():
        outputs = model.forward_single(
            node_feats=torch.from_numpy(node_feats_np).float().to(device),
            edge_index=torch.from_numpy(data["edge_index"]).long().to(device),
            edge_attr=torch.from_numpy(edge_attr_np).float().to(device),
            lang_feat=torch.from_numpy(data["lang_feat"].astype(np.float32)).float().to(device),
            candidate_mask=torch.from_numpy(candidate_mask_np).bool().to(device),
        )
        logits = outputs["goal_logits"].view(-1)
        cand_tensor = torch.from_numpy(cand_indices).long().to(device)
        local = int(torch.argmax(logits[cand_tensor]).item())
    return int(cand_indices[local])


def evaluate(
    model: LCORGNet,
    protocol_dir: Path,
    sigma_values: Iterable[float],
    repeats: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    graph_paths = sorted(protocol_dir.glob("scen_*.npz"))
    results: List[dict] = []
    all_records: List[dict] = []
    for sigma_m in sigma_values:
        correct = 0
        total = 0
        unchanged_from_clean = 0
        clean_by_scene: Dict[int, int] = {}
        rng = np.random.default_rng(seed + int(round(sigma_m * 1000)))
        for path in graph_paths:
            data = np.load(path, allow_pickle=False)
            scene_id = int(path.stem.split("_")[-1])
            clean_node, clean_edge = perturb_graph(data, 0.0, rng)
            clean_choice = predict(model, data, clean_node, clean_edge, device)
            clean_by_scene[scene_id] = clean_choice
            for repeat in range(repeats if sigma_m > 0 else 1):
                node_feats, edge_attr = perturb_graph(data, sigma_m, rng)
                choice = predict(model, data, node_feats, edge_attr, device)
                target = int(data["target_idx_gt"])
                ok = bool(choice == target)
                same = bool(choice == clean_choice)
                correct += int(ok)
                total += 1
                unchanged_from_clean += int(same)
                all_records.append(
                    {
                        "sigma_m": sigma_m,
                        "scene_id": scene_id,
                        "repeat": repeat,
                        "target_idx": target,
                        "chosen_idx": choice,
                        "clean_chosen_idx": clean_choice,
                        "correct": ok,
                        "unchanged_from_clean": same,
                    }
                )
        lo, hi = wilson_ci(correct, total)
        slo, shi = wilson_ci(unchanged_from_clean, total)
        results.append(
            {
                "sigma_m": sigma_m,
                "total": total,
                "correct": correct,
                "proto_acc": 100.0 * correct / total if total else 0.0,
                "proto_acc_ci95": [100.0 * lo, 100.0 * hi],
                "unchanged_from_clean": unchanged_from_clean,
                "stability": 100.0 * unchanged_from_clean / total if total else 0.0,
                "stability_ci95": [100.0 * slo, 100.0 * shi],
            }
        )
    return {
        "protocol_dir": str(protocol_dir),
        "num_graphs": len(graph_paths),
        "repeats_per_sigma": repeats,
        "seed": seed,
        "results": results,
        "records": all_records,
    }


def write_markdown(path: Path, checkpoint: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Input Feature Perturbation Robustness",
        "",
        "This diagnostic keeps the 30 protocol graphs fixed and injects Gaussian perturbations into relative x/y/distance node and edge features. It tests selector stability under small numeric perturbations; it is not a replacement for real perception-noise closed-loop evaluation.",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Protocol graphs: `{summary['protocol_dir']}`",
        f"- Repeats per nonzero noise level: {summary['repeats_per_sigma']}",
        "",
        "| Position/distance noise sigma | Trials | ProtoAcc@1 | Prediction stability vs clean |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lo, hi = row["proto_acc_ci95"]
        slo, shi = row["stability_ci95"]
        lines.append(
            "| {sigma_m:.2f} m | {total} | {proto_acc:.1f}% [{lo:.1f}, {hi:.1f}] | {stability:.1f}% [{slo:.1f}, {shi:.1f}] |".format(
                lo=lo,
                hi=hi,
                slo=slo,
                shi=shi,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation for R2: the clean protocol result is not a brittle single-point numerical artifact; meter-scale perturbations preserve the learned selector's advantage over chance on this controlled protocol. The result should still be described as a static selector stress test, not broad sensor robustness.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_lc_org_mtl_mixed_seed42.pt"))
    parser.add_argument("--protocol-dir", type=Path, default=Path("scripts/online_multi/protocol_graphs"))
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--out-md", type=Path, default=Path("runs/scaleup_20260520/main_30x10/input_perturbation_robustness.md"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/scaleup_20260520/main_30x10/input_perturbation_robustness.json"))
    parser.add_argument("--out-records", type=Path, default=Path("runs/scaleup_20260520/main_30x10/input_perturbation_robustness_records.jsonl"))
    args = parser.parse_args()

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = load_model(args.checkpoint, args.protocol_dir, device)
    summary = evaluate(model, args.protocol_dir, args.sigma, args.repeats, args.seed, device)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), encoding="utf-8")
    write_jsonl(args.out_records, summary["records"])
    write_markdown(args.out_md, args.checkpoint, summary)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_records}")


if __name__ == "__main__":
    main()
