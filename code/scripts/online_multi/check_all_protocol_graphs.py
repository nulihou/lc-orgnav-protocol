#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查所有协议图的健康状态

遍历 scen_000.npz 到 scen_029.npz，对每个 npz 检查：
- candidate_mask.sum() >= 2
- candidate_mask[target_idx_gt] == True
- len(node_spawn_indices) == node_feats.shape[0]
- len(candidate_indices) == candidate_mask.sum()
"""

import os
import sys
import numpy as np
from pathlib import Path
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def check_single_protocol_graph(npz_path: str, scen_id: int) -> tuple:
    """
    检查单个协议图
    
    Returns:
        (is_ok, summary_str, details_dict)
    """
    if not os.path.exists(npz_path):
        return False, f"scen_{scen_id:03d}: FILE NOT FOUND", {}
    
    try:
        data = np.load(npz_path, allow_pickle=False)
        
        # 提取数据
        node_feats = data["node_feats"]
        candidate_mask = data["candidate_mask"]
        target_idx_gt = int(data["target_idx_gt"])
        node_spawn_indices = data["node_spawn_indices"]
        candidate_indices = data["candidate_indices"]
        
        N = node_feats.shape[0]
        K = int(candidate_mask.sum())
        
        # 检查1: candidate_mask.sum() >= 2
        check1 = K >= 2
        
        # 检查2: candidate_mask[target_idx_gt] == True
        if target_idx_gt < 0 or target_idx_gt >= len(candidate_mask):
            check2 = False
        else:
            check2 = bool(candidate_mask[target_idx_gt])
        
        # 检查3: len(node_spawn_indices) == node_feats.shape[0]
        check3 = len(node_spawn_indices) == N
        
        # 检查4: len(candidate_indices) == candidate_mask.sum()
        check4 = len(candidate_indices) == K
        
        all_ok = check1 and check2 and check3 and check4
        
        # 生成summary字符串
        status = "OK" if all_ok else "FAIL"
        summary = f"scen_{scen_id:03d}: N={N}, K={K}, target={target_idx_gt}, {status}"
        
        details = {
            "N": N,
            "K": K,
            "target_idx_gt": target_idx_gt,
            "check1": check1,
            "check2": check2,
            "check3": check3,
            "check4": check4,
        }
        
        return all_ok, summary, details
        
    except Exception as e:
        return False, f"scen_{scen_id:03d}: ERROR - {str(e)}", {}


def main():
    protocol_dir = Path("scripts/online_multi/protocol_graphs")
    
    if not protocol_dir.exists():
        print(f"[ERROR] Protocol directory not found: {protocol_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("Protocol Graph Health Check")
    print("=" * 80)
    print()
    
    ok_count = 0
    fail_count = 0
    all_summaries = []
    failed_details = []
    
    # 检查所有场景 (0-29)
    for scen_id in range(30):
        npz_path = protocol_dir / f"scen_{scen_id:03d}.npz"
        is_ok, summary, details = check_single_protocol_graph(str(npz_path), scen_id)
        
        all_summaries.append(summary)
        print(summary)
        
        if is_ok:
            ok_count += 1
        else:
            fail_count += 1
            failed_details.append((scen_id, summary, details))
    
    print()
    print("=" * 80)
    print(f"总结: {ok_count}/30 场景协议图通过")
    
    if fail_count > 0:
        print(f"\n[失败场景详情]")
        for scen_id, summary, details in failed_details:
            print(f"  {summary}")
            if details:
                failed_checks = []
                if not details.get("check1", True):
                    failed_checks.append("K < 2")
                if not details.get("check2", True):
                    failed_checks.append("target not in candidates")
                if not details.get("check3", True):
                    failed_checks.append("node_spawn_indices length mismatch")
                if not details.get("check4", True):
                    failed_checks.append("candidate_indices length mismatch")
                if failed_checks:
                    print(f"    失败检查: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print("✅ 所有场景协议图健康检查通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()

