#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯协议图上的GS评估（不连接CARLA）

从 protocol_graphs/scen_xxx.npz 读图，在 candidate_mask 为 True 的节点上跑 LCORGNet，
选 logit 最大的 index，然后写入 JSONL。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/online_multi
_online_multi_dir = os.path.dirname(_script_dir)  # scripts
_project_root = os.path.dirname(_online_multi_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lc_org_nav.online_nav.gs_loader import load_gs_model

PROTO_DIR = "scripts/online_multi/protocol_graphs"
OUT_PATH = "logs_v2/proto_main_eval/gs_protocol.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_relation_mask_from_node_feats(node_feats_np: np.ndarray, rel_type: int) -> np.ndarray:
    """Return a relation-consistent mask reconstructed from anchor-relative features."""
    d_a = node_feats_np[:, 7]
    cos_a = node_feats_np[:, 8]
    sin_a = node_feats_np[:, 9]
    is_anchor = node_feats_np[:, 11] if node_feats_np.shape[1] > 11 else np.zeros_like(d_a)
    theta = np.arctan2(sin_a, cos_a)
    near_d = 20.0
    far_min_d = 5.0
    far_max_d = 40.0
    front_deg = np.deg2rad(30.0)
    side_min = np.deg2rad(60.0)
    side_max = np.deg2rad(120.0)
    if rel_type == 0:
        ok = d_a < near_d
    elif rel_type == 1:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (np.abs(theta) < front_deg)
    elif rel_type == 2:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (np.abs(np.abs(theta) - np.pi) < front_deg)
    elif rel_type == 3:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (theta > side_min) & (theta < side_max)
    elif rel_type == 4:
        ok = (d_a > far_min_d) & (d_a < far_max_d) & (theta > -side_max) & (theta < -side_min)
    else:
        ok = np.zeros_like(d_a, dtype=bool)
    return ok.astype(bool) & (is_anchor < 0.5)


def choose_with_selector_profile(
    logits: torch.Tensor,
    candidate_mask_np: np.ndarray,
    node_feats_np: np.ndarray,
    lang_feat_np: np.ndarray,
    selector_profile: str,
) -> Dict[str, Any]:
    cand_indices = np.nonzero(candidate_mask_np)[0]
    logits_cpu = logits.detach().cpu()
    rel_type = int(np.argmax(lang_feat_np[:5]))
    if selector_profile == "relation_gated":
        relation_mask = build_relation_mask_from_node_feats(node_feats_np, rel_type)
        gated_mask = candidate_mask_np & relation_mask
        gated_indices = np.nonzero(gated_mask)[0]
        if len(gated_indices) > 0:
            logits_gated = logits_cpu[torch.from_numpy(gated_mask)]
            k_best = int(torch.argmax(logits_gated).item())
            return {
                "chosen_idx": int(gated_indices[k_best]),
                "rel_type": rel_type,
                "relation_gate_candidates": [int(x) for x in gated_indices.tolist()],
                "relation_gate_fallback": False,
            }
        logits_cands = logits_cpu[torch.from_numpy(candidate_mask_np)]
        k_best = int(torch.argmax(logits_cands).item())
        return {
            "chosen_idx": int(cand_indices[k_best]),
            "rel_type": rel_type,
            "relation_gate_candidates": [],
            "relation_gate_fallback": True,
        }

    logits_cands = logits_cpu[torch.from_numpy(candidate_mask_np)]
    k_best = int(torch.argmax(logits_cands).item())
    return {
        "chosen_idx": int(cand_indices[k_best]),
        "rel_type": rel_type,
        "relation_gate_candidates": [],
        "relation_gate_fallback": False,
    }


def main():
    parser = argparse.ArgumentParser(description="GS Protocol Graph Evaluation (No CARLA)")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default=PROTO_DIR,
        help="Directory containing protocol graphs"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GS model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUT_PATH,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--max-scen-id",
        type=int,
        default=29,
        help="Maximum scene ID to evaluate (default: 29 for all 30 scenes)"
    )
    parser.add_argument(
        "--selector-profile",
        choices=["neural", "relation_gated"],
        default="neural",
        help="Target selector profile; default preserves the original neural-only selector"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GS Protocol Graph Evaluation (Pure Protocol, No CARLA)")
    print("=" * 80)
    print(f"[Config] Protocol dir: {args.protocol_dir}")
    print(f"[Config] Checkpoint: {args.checkpoint}")
    print(f"[Config] Output: {args.output}")
    print(f"[Config] Device: {args.device}")
    print(f"[Config] Max scen_id: {args.max_scen_id}")
    print(f"[Config] Selector profile: {args.selector_profile}")
    
    # Load model
    device = torch.device(args.device)
    print(f"\n[Model] Loading GS model...")
    model = load_gs_model(args.checkpoint, device)
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Open output file
    with open(args.output, "w", encoding="utf-8") as f_out:
        print(f"\n[Evaluation] Processing scenes 0-{args.max_scen_id}...")
        
        for scen_id in range(args.max_scen_id + 1):
            npz_path = os.path.join(args.protocol_dir, f"scen_{scen_id:03d}.npz")
            
            if not os.path.exists(npz_path):
                print(f"[WARN] scen_{scen_id:03d}: Protocol graph not found, skipping")
                continue
            
            try:
                data = np.load(npz_path, allow_pickle=False)
                
                # Extract data
                node_feats = torch.from_numpy(data["node_feats"]).float().to(device)
                edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
                edge_attr = torch.from_numpy(data["edge_attr"]).float().to(device)
                lang_feat = torch.from_numpy(data["lang_feat"]).float().to(device)
                cand_mask_np = data["candidate_mask"].astype(bool)
                candidate_mask = torch.from_numpy(cand_mask_np).bool().to(device)
                target_idx_gt = int(data["target_idx_gt"])
                
                # Model inference
                with torch.no_grad():
                    # Add batch dimension if needed
                    node_feats_batch = node_feats.unsqueeze(0) if node_feats.dim() == 2 else node_feats
                    edge_attr_batch = edge_attr.unsqueeze(0) if edge_attr.dim() == 2 else edge_attr
                    lang_feat_batch = lang_feat.unsqueeze(0) if lang_feat.dim() == 1 else lang_feat
                    candidate_mask_batch = candidate_mask.unsqueeze(0) if candidate_mask.dim() == 1 else candidate_mask
                    
                    outputs = model.forward_single(
                        node_feats=node_feats_batch[0],
                        edge_index=edge_index,
                        edge_attr=edge_attr_batch[0],
                        lang_feat=lang_feat_batch[0],
                        candidate_mask=candidate_mask_batch[0],
                    )
                    goal_logits = outputs["goal_logits"]  # (N,)
                
                # 只在候选集合上取 argmax
                logits = goal_logits.view(-1)
                choice = choose_with_selector_profile(
                    logits=logits,
                    candidate_mask_np=cand_mask_np,
                    node_feats_np=data["node_feats"].astype(np.float32),
                    lang_feat_np=data["lang_feat"].astype(np.float32),
                    selector_profile=args.selector_profile,
                )
                chosen_idx = int(choice["chosen_idx"])
                
                gs_correct = (chosen_idx == target_idx_gt)
                
                # Record result
                rec = {
                    "scen_id": scen_id,
                    "num_nodes": int(node_feats.shape[0]),
                    "num_candidates": int(cand_mask_np.sum()),
                    "target_idx_gt": target_idx_gt,
                    "chosen_idx": chosen_idx,
                    "gs_correct": gs_correct,
                    "selector_profile": args.selector_profile,
                    "rel_type": int(choice["rel_type"]),
                    "relation_gate_candidates": choice["relation_gate_candidates"],
                    "relation_gate_fallback": bool(choice["relation_gate_fallback"]),
                }
                
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                
                status = "✓" if gs_correct else "✗"
                print(f"  scen_{scen_id:03d}: chosen={chosen_idx}, target={target_idx_gt}, gs_correct={status}")
                
            except Exception as e:
                print(f"[ERROR] scen_{scen_id:03d}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n[Done] Evaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
