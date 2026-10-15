#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协议层一致性检查（层2验证）

读取三份 JSONL：
- gs_protocol.jsonl
- random_protocol.jsonl
- oracle_protocol.jsonl

对于每个 scen_id，检查：
- num_candidates 是否完全一致
- target_idx_gt 是否完全一致

最后打印：Consistent: 30/30 scen
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_jsonl(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """
    加载JSONL文件，返回 {scen_id: result_dict}
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"[WARNING] JSONL file not found: {jsonl_path}")
        return {}
    
    results = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                scen_id = result.get("scen_id")
                if scen_id is not None:
                    results[scen_id] = result
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse line: {e}")
                continue
    
    return results


def check_consistency(
    gs_jsonl: str,
    random_jsonl: str,
    oracle_jsonl: str,
) -> Dict[str, Any]:
    """
    检查三个JSONL文件的一致性
    
    Returns:
        检查结果字典
    """
    print("=" * 80)
    print("Protocol Graph Evaluation Consistency Check (Layer 2)")
    print("=" * 80)
    
    # 加载三个JSONL文件
    print(f"\n[Load] Loading JSONL files...")
    gs_results = load_jsonl(gs_jsonl)
    random_results = load_jsonl(random_jsonl)
    oracle_results = load_jsonl(oracle_jsonl)
    
    print(f"  GS:     {len(gs_results)} scenarios")
    print(f"  Random: {len(random_results)} scenarios")
    print(f"  Oracle: {len(oracle_results)} scenarios")
    
    # 获取所有场景ID的并集
    all_scen_ids = set(gs_results.keys()) | set(random_results.keys()) | set(oracle_results.keys())
    all_scen_ids = sorted(all_scen_ids)
    
    print(f"\n[Check] Checking consistency for {len(all_scen_ids)} scenarios...")
    
    inconsistencies = []
    consistent_count = 0
    
    for scen_id in all_scen_ids:
        gs = gs_results.get(scen_id)
        rnd = random_results.get(scen_id)
        ora = oracle_results.get(scen_id)
        
        # 检查是否存在
        missing = []
        if gs is None:
            missing.append("GS")
        if rnd is None:
            missing.append("Random")
        if ora is None:
            missing.append("Oracle")
        
        if missing:
            inconsistencies.append({
                "scen_id": scen_id,
                "type": "missing",
                "missing_modes": missing,
            })
            continue
        
        # 检查 num_candidates
        gs_num = gs.get("num_candidates")
        rnd_num = rnd.get("num_candidates")
        ora_num = ora.get("num_candidates")
        
        if not (gs_num == rnd_num == ora_num):
            inconsistencies.append({
                "scen_id": scen_id,
                "type": "num_candidates_mismatch",
                "gs": gs_num,
                "random": rnd_num,
                "oracle": ora_num,
            })
            continue
        
        # 检查 target_idx_gt
        gs_target = gs.get("target_idx_gt")
        rnd_target = rnd.get("target_idx_gt")
        ora_target = ora.get("target_idx_gt")
        
        if not (gs_target == rnd_target == ora_target):
            inconsistencies.append({
                "scen_id": scen_id,
                "type": "target_idx_gt_mismatch",
                "gs": gs_target,
                "random": rnd_target,
                "oracle": ora_target,
            })
            continue
        
        # 所有检查通过
        consistent_count += 1
    
    # 打印结果
    print(f"\n[Summary]")
    print(f"  Total scenarios: {len(all_scen_ids)}")
    print(f"  Consistent: {consistent_count}")
    print(f"  Inconsistent: {len(inconsistencies)}")
    
    if inconsistencies:
        print(f"\n[Inconsistencies]")
        for inc in inconsistencies:
            scen_id = inc["scen_id"]
            inc_type = inc["type"]
            
            if inc_type == "missing":
                print(f"  scen_{scen_id:03d}: Missing in {', '.join(inc['missing_modes'])}")
            elif inc_type == "num_candidates_mismatch":
                print(f"  scen_{scen_id:03d}: num_candidates mismatch")
                print(f"    GS: {inc['gs']}, Random: {inc['random']}, Oracle: {inc['oracle']}")
            elif inc_type == "target_idx_gt_mismatch":
                print(f"  scen_{scen_id:03d}: target_idx_gt mismatch")
                print(f"    GS: {inc['gs']}, Random: {inc['random']}, Oracle: {inc['oracle']}")
    else:
        print(f"\n✅ Consistent: {consistent_count}/{len(all_scen_ids)} scen")
    
    return {
        "total": len(all_scen_ids),
        "consistent": consistent_count,
        "inconsistent": len(inconsistencies),
        "inconsistencies": inconsistencies,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Protocol Graph Evaluation Consistency")
    parser.add_argument(
        "--gs-jsonl",
        type=str,
        default="logs_v2/proto_main_eval/gs_protocol.jsonl",
        help="GS evaluation JSONL path"
    )
    parser.add_argument(
        "--random-jsonl",
        type=str,
        default="logs_v2/proto_main_eval/random_protocol.jsonl",
        help="Random evaluation JSONL path"
    )
    parser.add_argument(
        "--oracle-jsonl",
        type=str,
        default="logs_v2/proto_main_eval/oracle_protocol.jsonl",
        help="Oracle evaluation JSONL path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report path (optional)"
    )
    
    args = parser.parse_args()
    
    # 运行检查
    result = check_consistency(
        gs_jsonl=args.gs_jsonl,
        random_jsonl=args.random_jsonl,
        oracle_jsonl=args.oracle_jsonl,
    )
    
    # 保存报告（如果需要）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[Report] Consistency report saved to {args.output}")
    
    # 返回退出码
    if result["inconsistent"] > 0:
        print(f"\n[Result] ❌ Found {result['inconsistent']} inconsistencies")
        sys.exit(1)
    else:
        print(f"\n[Result] ✅ Consistent: {result['consistent']}/{result['total']} scen")
        sys.exit(0)


if __name__ == "__main__":
    main()
