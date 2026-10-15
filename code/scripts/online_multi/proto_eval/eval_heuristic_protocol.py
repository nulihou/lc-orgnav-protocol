#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启发式baseline评估（协议图级别）

实现简单的启发式规则：
1. Nearest: 选与anchor最近的车
2. Largest: 选尺寸最大的车（如果有尺寸信息）
"""

import os
import sys
import json
import argparse
import numpy as np
import io
from pathlib import Path
from typing import Dict, Any, List, Optional

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_online_multi_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(os.path.dirname(_online_multi_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_protocol_graph(proto_path: Path) -> Dict[str, Any]:
    """加载协议图"""
    data = np.load(proto_path, allow_pickle=True)
    return {
        "node_feats": data["node_feats"],
        "candidate_mask": data["candidate_mask"],
        "target_idx_gt": int(data["target_idx_gt"]),
        "node_spawn_indices": data.get("node_spawn_indices", []),
    }


def heuristic_nearest(proto: Dict[str, Any]) -> int:
    """
    启发式规则1: 选与anchor最近的车
    
    从node_feats中提取距离信息（d_a在索引7）
    """
    node_feats = proto["node_feats"]
    candidate_mask = proto["candidate_mask"]
    
    # 找到所有候选节点
    candidate_indices = np.where(candidate_mask)[0]
    
    if len(candidate_indices) == 0:
        return -1
    
    # 提取每个候选节点到anchor的距离（d_a在索引7）
    distances = []
    for idx in candidate_indices:
        d_a = float(node_feats[idx, 7])  # d_a是anchor到该节点的距离
        distances.append((idx, d_a))
    
    # 选距离最小的
    distances.sort(key=lambda x: x[1])
    chosen_idx = distances[0][0]
    
    return int(chosen_idx)


def heuristic_nearest_with_front_priority(proto: Dict[str, Any]) -> int:
    """
    启发式规则2: 如果有near+front，优先选front，否则选最近
    
    从node_feats中提取：
    - d_a (索引7): anchor到节点的距离
    - rel_type (需要从边信息中获取，这里简化处理)
    """
    node_feats = proto["node_feats"]
    candidate_mask = proto["candidate_mask"]
    
    candidate_indices = np.where(candidate_mask)[0]
    
    if len(candidate_indices) == 0:
        return -1
    
    # 简化版：只选最近的（因为协议图中没有直接存储rel_type）
    # 如果需要更复杂的规则，可以从edge_attr中提取
    return heuristic_nearest(proto)


def evaluate_heuristic(
    protocol_dir: Path,
    max_scen_id: int = 29,
    heuristic_type: str = "nearest",
) -> List[Dict[str, Any]]:
    """评估启发式baseline"""
    
    results = []
    
    print(f"[Config] Protocol dir: {protocol_dir}")
    print(f"[Config] Heuristic: {heuristic_type}")
    print(f"[Config] Max scen_id: {max_scen_id}")
    print()
    
    print("[Evaluation] Processing scenes 0-{}...".format(max_scen_id))
    
    for scen_id in range(max_scen_id + 1):
        proto_path = protocol_dir / f"scen_{scen_id:03d}.npz"
        
        if not proto_path.exists():
            print(f"  [WARNING] scen_{scen_id:03d}.npz not found, skipping")
            continue
        
        proto = load_protocol_graph(proto_path)
        target_idx_gt = proto["target_idx_gt"]
        candidate_mask = proto["candidate_mask"]
        
        # 应用启发式规则
        if heuristic_type == "nearest":
            chosen_idx = heuristic_nearest(proto)
        elif heuristic_type == "nearest_with_front":
            chosen_idx = heuristic_nearest_with_front_priority(proto)
        else:
            print(f"  [ERROR] Unknown heuristic type: {heuristic_type}")
            chosen_idx = -1
        
        # 检查是否正确
        gs_correct = (chosen_idx == target_idx_gt) if chosen_idx >= 0 else False
        
        # 检查是否在候选集合内
        is_valid = (chosen_idx >= 0 and chosen_idx < len(candidate_mask) and 
                   candidate_mask[chosen_idx]) if chosen_idx >= 0 else False
        
        result = {
            "scen_id": scen_id,
            "mode": f"heuristic_{heuristic_type}",
            "chosen_idx_graph": int(chosen_idx) if chosen_idx >= 0 else None,
            "target_idx_gt": int(target_idx_gt),
            "gs_correct": bool(gs_correct),
            "is_valid": bool(is_valid),
            "num_candidates": int(candidate_mask.sum()),
        }
        
        results.append(result)
        
        status = "OK" if gs_correct else "FAIL"
        print(f"  scen_{scen_id:03d}: chosen={chosen_idx}, target={target_idx_gt}, gs_correct={status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Heuristic Baseline on Protocol Graphs")
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="scripts/online_multi/protocol_graphs",
        help="Directory containing protocol graph .npz files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs_v2/proto_main_eval/heuristic_nearest_protocol.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max-scen-id",
        type=int,
        default=29,
        help="Maximum scenario ID to evaluate"
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        default="nearest",
        choices=["nearest", "nearest_with_front"],
        help="Heuristic type"
    )
    
    args = parser.parse_args()
    
    protocol_dir = Path(args.protocol_dir)
    if not protocol_dir.exists():
        print(f"[ERROR] Protocol directory not found: {protocol_dir}")
        return
    
    # 评估
    results = evaluate_heuristic(
        protocol_dir=protocol_dir,
        max_scen_id=args.max_scen_id,
        heuristic_type=args.heuristic,
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 统计
    total = len(results)
    correct = sum(1 for r in results if r["gs_correct"])
    valid = sum(1 for r in results if r["is_valid"])
    
    print()
    print("[Summary]")
    print(f"  Total scenarios: {total}")
    print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Valid: {valid}/{total} ({valid/total*100:.1f}%)")
    print()
    print(f"[Done] Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()

