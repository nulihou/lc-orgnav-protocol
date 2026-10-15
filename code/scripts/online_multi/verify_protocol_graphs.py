#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证协议图正确性

检查预计算的协议图是否符合要求：
- candidate_mask.sum() 至少是 2
- candidate_mask[target_idx_gt] == True
- len(node_spawn_indices) == node_feats.shape[0]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def verify_single_protocol_graph(graph_path: str, scen_id: int) -> Dict[str, Any]:
    """
    验证单个协议图
    
    Returns:
        验证结果字典
    """
    if not os.path.exists(graph_path):
        return {
            "valid": False,
            "error": f"File not found: {graph_path}"
        }
    
    try:
        d = np.load(graph_path, allow_pickle=False)
        
        # 检查必需字段
        required_fields = ["node_feats", "edge_index", "edge_attr", "candidate_mask", 
                          "target_idx_gt", "node_spawn_indices", "candidate_indices"]
        missing_fields = [f for f in required_fields if f not in d.files]
        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing fields: {missing_fields}"
            }
        
        # 提取数据
        node_feats = d["node_feats"]
        edge_index = d["edge_index"]
        candidate_mask = d["candidate_mask"]
        target_idx_gt = int(d["target_idx_gt"])
        node_spawn_indices = d["node_spawn_indices"]
        candidate_indices = d["candidate_indices"]
        
        # 验证1: candidate_mask.sum() 至少是 2
        num_candidates = candidate_mask.sum()
        check1 = num_candidates >= 2
        
        # 验证2: candidate_mask[target_idx_gt] == True
        if target_idx_gt < 0 or target_idx_gt >= len(candidate_mask):
            check2 = False
            check2_error = f"target_idx_gt={target_idx_gt} out of range (len={len(candidate_mask)})"
        else:
            check2 = bool(candidate_mask[target_idx_gt])
            check2_error = None
        
        # 验证3: len(node_spawn_indices) == node_feats.shape[0]
        check3 = len(node_spawn_indices) == node_feats.shape[0]
        
        # 验证4: candidate_indices 与 candidate_mask 一致
        candidate_indices_from_mask = np.nonzero(candidate_mask)[0].tolist()
        candidate_indices_list = candidate_indices.tolist() if hasattr(candidate_indices, 'tolist') else list(candidate_indices)
        check4 = set(candidate_indices_from_mask) == set(candidate_indices_list)
        
        # 验证5: target_idx_gt 在 candidate_indices 中
        check5 = target_idx_gt in candidate_indices_list
        
        all_valid = check1 and check2 and check3 and check4 and check5
        
        return {
            "valid": all_valid,
            "scen_id": scen_id,
            "checks": {
                "num_candidates >= 2": check1,
                "target_idx_gt in candidate_mask": check2,
                "node_spawn_indices length matches": check3,
                "candidate_indices consistent": check4,
                "target_idx_gt in candidate_indices": check5,
            },
            "stats": {
                "num_nodes": node_feats.shape[0],
                "num_edges": edge_index.shape[1],
                "num_candidates": int(num_candidates),
                "target_idx_gt": target_idx_gt,
                "node_spawn_indices_len": len(node_spawn_indices),
                "candidate_indices": candidate_indices_list,
            },
            "error": check2_error if not check2 else None,
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"Exception: {str(e)}"
        }


def verify_all_protocol_graphs(protocol_dir: str, max_scen_id: int = None) -> Dict[str, Any]:
    """
    验证所有协议图
    
    Args:
        protocol_dir: 协议图目录
        max_scen_id: 最大场景ID（用于限制验证范围）
    
    Returns:
        验证结果汇总
    """
    protocol_dir = Path(protocol_dir)
    
    # 确定要验证的场景ID列表
    if max_scen_id is None:
        scen_ids = []
        for graph_file in protocol_dir.glob("scen_*.npz"):
            scen_id_str = graph_file.stem.replace("scen_", "")
            try:
                scen_id = int(scen_id_str)
                scen_ids.append(scen_id)
            except ValueError:
                continue
        scen_ids = sorted(scen_ids)
    else:
        scen_ids = list(range(max_scen_id + 1))
    
    print(f"[Verify] Verifying {len(scen_ids)} protocol graphs...")
    print("=" * 80)
    
    results = []
    valid_count = 0
    invalid_count = 0
    
    for scen_id in scen_ids:
        graph_path = protocol_dir / f"scen_{scen_id:03d}.npz"
        result = verify_single_protocol_graph(str(graph_path), scen_id)
        results.append(result)
        
        if result["valid"]:
            valid_count += 1
            print(f"✅ scen_{scen_id:03d}: PASS")
        else:
            invalid_count += 1
            print(f"❌ scen_{scen_id:03d}: FAIL")
            if result.get("error"):
                print(f"   Error: {result['error']}")
            if result.get("checks"):
                failed_checks = [k for k, v in result["checks"].items() if not v]
                if failed_checks:
                    print(f"   Failed checks: {', '.join(failed_checks)}")
    
    print("=" * 80)
    print(f"[Summary]")
    print(f"  Total: {len(scen_ids)}")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    
    # 打印第一个场景的详细信息（用于sanity check）
    if results:
        first_result = results[0]
        if first_result["valid"]:
            print(f"\n[Example] scen_000 details:")
            print(f"  Files: {list(np.load(str(protocol_dir / 'scen_000.npz'), allow_pickle=False).files)}")
            print(f"  node_feats shape: {first_result['stats']['num_nodes']}")
            print(f"  edge_index shape: (2, {first_result['stats']['num_edges']})")
            print(f"  candidate_mask sum: {first_result['stats']['num_candidates']}")
            print(f"  target_idx_gt: {first_result['stats']['target_idx_gt']}")
            print(f"  node_spawn_indices: {first_result['stats'].get('node_spawn_indices', 'N/A')}")
            print(f"  candidate_indices: {first_result['stats']['candidate_indices']}")
    
    return {
        "total": len(scen_ids),
        "valid": valid_count,
        "invalid": invalid_count,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify protocol graphs")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="scripts/online_multi/protocol_graphs",
        help="Directory containing protocol graphs"
    )
    parser.add_argument(
        "--scen-id",
        type=int,
        default=None,
        help="Verify only a specific scene ID (for sanity check)"
    )
    parser.add_argument(
        "--max-scen-id",
        type=int,
        default=None,
        help="Maximum scene ID to verify"
    )
    
    args = parser.parse_args()
    
    if args.scen_id is not None:
        # 只验证单个场景
        graph_path = Path(args.protocol_dir) / f"scen_{args.scen_id:03d}.npz"
        print(f"[Verify] Verifying single protocol graph: {graph_path}")
        print("=" * 80)
        
        result = verify_single_protocol_graph(str(graph_path), args.scen_id)
        
        if result["valid"]:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            if result.get("error"):
                print(f"Error: {result['error']}")
        
        print("\n[Details]")
        print(f"  Checks: {result.get('checks', {})}")
        print(f"  Stats: {result.get('stats', {})}")
        
        # 打印原始数据（用于sanity check）
        if os.path.exists(graph_path):
            d = np.load(str(graph_path), allow_pickle=False)
            print(f"\n[Raw Data]")
            print(f"  Files: {list(d.files)}")
            print(f"  node_feats shape: {d['node_feats'].shape}")
            print(f"  edge_index shape: {d['edge_index'].shape}")
            print(f"  candidate_mask: {d['candidate_mask']} (sum={d['candidate_mask'].sum()})")
            print(f"  target_idx_gt: {d['target_idx_gt']}")
            print(f"  node_spawn_indices: {d['node_spawn_indices']}")
            print(f"  candidate_indices: {d['candidate_indices']}")
    else:
        # 验证所有场景
        summary = verify_all_protocol_graphs(args.protocol_dir, args.max_scen_id)
        
        if summary["invalid"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()

