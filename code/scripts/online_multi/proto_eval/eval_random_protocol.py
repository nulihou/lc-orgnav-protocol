#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯协议图上的Random评估（不连接CARLA）

从 protocol_graphs/scen_xxx.npz 读图，在 candidate_mask 为 True 的集合里随机选一个 index。
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import io
import random

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

PROTO_DIR = "scripts/online_multi/protocol_graphs"
OUT_PATH = "logs_v2/proto_main_eval/random_protocol.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Random Protocol Graph Evaluation (No CARLA)")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default=PROTO_DIR,
        help="Directory containing protocol graphs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUT_PATH,
        help="Output JSONL path"
    )
    parser.add_argument(
        "--max-scen-id",
        type=int,
        default=29,
        help="Maximum scene ID to evaluate (default: 29 for all 30 scenes)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("Random Protocol Graph Evaluation (Pure Protocol, No CARLA)")
    print("=" * 80)
    print(f"[Config] Protocol dir: {args.protocol_dir}")
    print(f"[Config] Output: {args.output}")
    print(f"[Config] Max scen_id: {args.max_scen_id}")
    print(f"[Config] Seed: {args.seed}")
    
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
                node_feats = data["node_feats"]
                cand_mask_np = data["candidate_mask"].astype(bool)
                target_idx_gt = int(data["target_idx_gt"])
                
                # Random selection
                cand_indices = np.nonzero(cand_mask_np)[0]
                chosen_idx = int(np.random.choice(cand_indices))
                
                gs_correct = (chosen_idx == target_idx_gt)
                
                # Record result
                rec = {
                    "scen_id": scen_id,
                    "num_nodes": int(node_feats.shape[0]),
                    "num_candidates": int(cand_mask_np.sum()),
                    "target_idx_gt": target_idx_gt,
                    "chosen_idx": chosen_idx,
                    "gs_correct": gs_correct,
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
