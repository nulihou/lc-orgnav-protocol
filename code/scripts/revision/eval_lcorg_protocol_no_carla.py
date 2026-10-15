#!/usr/bin/env python3
"""Evaluate LCORG checkpoints on protocol graphs without importing CARLA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def evaluate(
    checkpoint: Path,
    protocol_dir: Path,
    output: Path,
    device: torch.device,
    max_scen_id: int,
) -> Dict[str, Any]:
    model = load_model(checkpoint, protocol_dir, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for scen_id in range(max_scen_id + 1):
            path = protocol_dir / f"scen_{scen_id:03d}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False)
            node_feats = torch.from_numpy(data["node_feats"]).float().to(device)
            edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
            edge_attr = torch.from_numpy(data["edge_attr"]).float().to(device)
            lang_feat = torch.from_numpy(data["lang_feat"]).float().to(device)
            cand_mask_np = data["candidate_mask"].astype(bool)
            candidate_mask = torch.from_numpy(cand_mask_np).bool().to(device)
            target_idx_gt = int(data["target_idx_gt"])
            with torch.no_grad():
                outputs = model.forward_single(
                    node_feats=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    lang_feat=lang_feat,
                    candidate_mask=candidate_mask,
                )
                logits = outputs["goal_logits"].view(-1)
                cand_indices = np.nonzero(cand_mask_np)[0]
                if len(cand_indices) == 0:
                    chosen_idx = -1
                else:
                    cand_tensor = torch.from_numpy(cand_indices).long().to(device)
                    chosen_idx = int(cand_indices[int(torch.argmax(logits[cand_tensor]).item())])
            rec = {
                "scen_id": scen_id,
                "num_nodes": int(node_feats.shape[0]),
                "num_candidates": int(cand_mask_np.sum()),
                "target_idx_gt": target_idx_gt,
                "chosen_idx": chosen_idx,
                "gs_correct": bool(chosen_idx == target_idx_gt),
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    correct = sum(1 for item in records if item["gs_correct"])
    total = len(records)
    lo, hi = wilson_ci(correct, total)
    return {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "ci95": [100.0 * lo, 100.0 * hi],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("scripts/online_multi/protocol_graphs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-scen-id", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    result = evaluate(args.checkpoint, args.protocol_dir, args.output, torch.device(device_name), args.max_scen_id)
    print(json.dumps(result, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
