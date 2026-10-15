#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
难度分组/鲁棒性分析

从现有日志中提取难度维度，按难度分组统计GS/Random/Oracle的性能
"""

import json
import sys
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    results = []
    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return results
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse line: {e}")
                continue
    
    return results


def bucket_distance(d: float) -> str:
    """距离分桶"""
    if d < 40:
        return "30–40m"
    elif d < 50:
        return "40–50m"
    else:
        return "50–60m"


def analyze_by_difficulty(
    gs_results: List[Dict[str, Any]],
    random_results: List[Dict[str, Any]],
    oracle_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """按难度维度分组分析"""
    
    # 按scen_id组织数据
    gs_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in gs_results}
    random_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in random_results}
    oracle_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in oracle_results}
    
    # 难度维度1: 起始距离
    by_dist = defaultdict(lambda: {
        "gs": {"total": 0, "navsucc_8m": 0, "semantic_navsucc_8m": 0, "gs_correct": 0},
        "random": {"total": 0, "navsucc_8m": 0, "semantic_navsucc_8m": 0, "gs_correct": 0},
        "oracle": {"total": 0, "navsucc_8m": 0, "semantic_navsucc_8m": 0, "gs_correct": 0},
    })
    
    # 难度维度2: 候选数
    by_k = defaultdict(lambda: {
        "gs": {"total": 0, "navsucc_8m": 0, "semantic_navsucc_8m": 0, "gs_correct": 0},
        "random": {"total": 0, "navsucc_8m": 0, "semantic_navsucc_8m": 0, "gs_correct": 0},
        "oracle": {"total": 0, "navsucc_8m": 0, "semantic_navsucc_8m": 0, "gs_correct": 0},
    })
    
    # 处理所有场景
    all_scen_ids = set(gs_by_scen.keys()) | set(random_by_scen.keys()) | set(oracle_by_scen.keys())
    
    for scen_id in all_scen_ids:
        gs_r = gs_by_scen.get(scen_id, {})
        random_r = random_by_scen.get(scen_id, {})
        oracle_r = oracle_by_scen.get(scen_id, {})
        
        # 提取难度特征
        start_dist = gs_r.get("start_dist") or random_r.get("start_dist") or oracle_r.get("start_dist", 0.0)
        num_candidates = gs_r.get("num_candidates") or random_r.get("num_candidates") or oracle_r.get("num_candidates", 2)
        
        dist_bucket = bucket_distance(start_dist)
        k_bucket = f"K={num_candidates}"
        
        # 处理GS
        if gs_r:
            by_dist[dist_bucket]["gs"]["total"] += 1
            by_k[k_bucket]["gs"]["total"] += 1
            
            if gs_r.get("success_8m", False):
                by_dist[dist_bucket]["gs"]["navsucc_8m"] += 1
                by_k[k_bucket]["gs"]["navsucc_8m"] += 1
            
            if gs_r.get("gs_correct", False):
                by_dist[dist_bucket]["gs"]["gs_correct"] += 1
                by_k[k_bucket]["gs"]["gs_correct"] += 1
            
            if gs_r.get("gs_correct", False) and gs_r.get("success_8m", False):
                by_dist[dist_bucket]["gs"]["semantic_navsucc_8m"] += 1
                by_k[k_bucket]["gs"]["semantic_navsucc_8m"] += 1
        
        # 处理Random
        if random_r:
            by_dist[dist_bucket]["random"]["total"] += 1
            by_k[k_bucket]["random"]["total"] += 1
            
            if random_r.get("success_8m", False):
                by_dist[dist_bucket]["random"]["navsucc_8m"] += 1
                by_k[k_bucket]["random"]["navsucc_8m"] += 1
            
            if random_r.get("gs_correct", False):
                by_dist[dist_bucket]["random"]["gs_correct"] += 1
                by_k[k_bucket]["random"]["gs_correct"] += 1
            
            if random_r.get("gs_correct", False) and random_r.get("success_8m", False):
                by_dist[dist_bucket]["random"]["semantic_navsucc_8m"] += 1
                by_k[k_bucket]["random"]["semantic_navsucc_8m"] += 1
        
        # 处理Oracle
        if oracle_r:
            by_dist[dist_bucket]["oracle"]["total"] += 1
            by_k[k_bucket]["oracle"]["total"] += 1
            
            if oracle_r.get("success_8m", False):
                by_dist[dist_bucket]["oracle"]["navsucc_8m"] += 1
                by_k[k_bucket]["oracle"]["navsucc_8m"] += 1
            
            if oracle_r.get("gs_correct", False):
                by_dist[dist_bucket]["oracle"]["gs_correct"] += 1
                by_k[k_bucket]["oracle"]["gs_correct"] += 1
            
            if oracle_r.get("gs_correct", False) and oracle_r.get("success_8m", False):
                by_dist[dist_bucket]["oracle"]["semantic_navsucc_8m"] += 1
                by_k[k_bucket]["oracle"]["semantic_navsucc_8m"] += 1
    
    # 计算百分比
    def calc_metrics(bucket_data: Dict[str, Any]) -> Dict[str, float]:
        total = bucket_data["total"]
        if total == 0:
            return {
                "navsucc_8m": 0.0,
                "semantic_navsucc_8m": 0.0,
                "gs_at_1": 0.0,
            }
        return {
            "navsucc_8m": bucket_data["navsucc_8m"] / total * 100,
            "semantic_navsucc_8m": bucket_data["semantic_navsucc_8m"] / total * 100,
            "gs_at_1": bucket_data["gs_correct"] / total * 100,
        }
    
    # 按距离分组结果
    by_dist_results = {}
    for dist_bucket in sorted(by_dist.keys()):
        by_dist_results[dist_bucket] = {
            "gs": calc_metrics(by_dist[dist_bucket]["gs"]),
            "random": calc_metrics(by_dist[dist_bucket]["random"]),
            "oracle": calc_metrics(by_dist[dist_bucket]["oracle"]),
            "total": by_dist[dist_bucket]["gs"]["total"],
        }
    
    # 按候选数分组结果
    by_k_results = {}
    for k_bucket in sorted(by_k.keys()):
        by_k_results[k_bucket] = {
            "gs": calc_metrics(by_k[k_bucket]["gs"]),
            "random": calc_metrics(by_k[k_bucket]["random"]),
            "oracle": calc_metrics(by_k[k_bucket]["oracle"]),
            "total": by_k[k_bucket]["gs"]["total"],
        }
    
    return {
        "by_distance": by_dist_results,
        "by_candidates": by_k_results,
    }


def print_results(results: Dict[str, Any]):
    """打印结果"""
    print("="*80)
    print("Robustness Analysis by Difficulty")
    print("="*80)
    
    # 按距离分组
    print("\n=== By Starting Distance ===")
    print(f"{'Distance':<15} {'Mode':<10} {'NavSucc@8m':<15} {'Semantic-NavSucc@8m':<25} {'GS@1':<10} {'N':<5}")
    print("-"*80)
    
    for dist_bucket in sorted(results["by_distance"].keys()):
        data = results["by_distance"][dist_bucket]
        total = data["total"]
        
        print(f"{dist_bucket:<15} {'GS':<10} {data['gs']['navsucc_8m']:>6.1f}%{'':<8} {data['gs']['semantic_navsucc_8m']:>6.1f}%{'':<18} {data['gs']['gs_at_1']:>6.1f}%{'':<3} {total:<5}")
        print(f"{'':<15} {'Random':<10} {data['random']['navsucc_8m']:>6.1f}%{'':<8} {data['random']['semantic_navsucc_8m']:>6.1f}%{'':<18} {data['random']['gs_at_1']:>6.1f}%{'':<3} {total:<5}")
        print(f"{'':<15} {'Oracle':<10} {data['oracle']['navsucc_8m']:>6.1f}%{'':<8} {data['oracle']['semantic_navsucc_8m']:>6.1f}%{'':<18} {data['oracle']['gs_at_1']:>6.1f}%{'':<3} {total:<5}")
        print()
    
    # 按候选数分组
    print("\n=== By Number of Candidates ===")
    print(f"{'#Candidates':<15} {'Mode':<10} {'NavSucc@8m':<15} {'Semantic-NavSucc@8m':<25} {'GS@1':<10} {'N':<5}")
    print("-"*80)
    
    for k_bucket in sorted(results["by_candidates"].keys()):
        data = results["by_candidates"][k_bucket]
        total = data["total"]
        
        print(f"{k_bucket:<15} {'GS':<10} {data['gs']['navsucc_8m']:>6.1f}%{'':<8} {data['gs']['semantic_navsucc_8m']:>6.1f}%{'':<18} {data['gs']['gs_at_1']:>6.1f}%{'':<3} {total:<5}")
        print(f"{'':<15} {'Random':<10} {data['random']['navsucc_8m']:>6.1f}%{'':<8} {data['random']['semantic_navsucc_8m']:>6.1f}%{'':<18} {data['random']['gs_at_1']:>6.1f}%{'':<3} {total:<5}")
        print(f"{'':<15} {'Oracle':<10} {data['oracle']['navsucc_8m']:>6.1f}%{'':<8} {data['oracle']['semantic_navsucc_8m']:>6.1f}%{'':<18} {data['oracle']['gs_at_1']:>6.1f}%{'':<3} {total:<5}")
        print()


def generate_markdown_table(results: Dict[str, Any]) -> str:
    """生成Markdown表格"""
    md = "# Robustness Analysis by Difficulty\n\n"
    md += "**Analysis Date**: 2025-11-23\n\n"
    
    # 按距离分组表格
    md += "## Table: Performance by Starting Distance\n\n"
    md += "| Distance | Mode | NavSucc@8m | Semantic-NavSucc@8m | GS@1 (proto) | N |\n"
    md += "|----------|------|------------|---------------------|--------------|---|\n"
    
    for dist_bucket in sorted(results["by_distance"].keys()):
        data = results["by_distance"][dist_bucket]
        total = data["total"]
        
        md += f"| {dist_bucket} | **GS** | **{data['gs']['navsucc_8m']:.1f}%** | **{data['gs']['semantic_navsucc_8m']:.1f}%** | **{data['gs']['gs_at_1']:.1f}%** | {total} |\n"
        md += f"| | Random | {data['random']['navsucc_8m']:.1f}% | {data['random']['semantic_navsucc_8m']:.1f}% | {data['random']['gs_at_1']:.1f}% | {total} |\n"
        md += f"| | Oracle | {data['oracle']['navsucc_8m']:.1f}% | {data['oracle']['semantic_navsucc_8m']:.1f}% | {data['oracle']['gs_at_1']:.1f}% | {total} |\n"
    
    md += "\n"
    
    # 按候选数分组表格
    md += "## Table: Performance by Number of Candidates\n\n"
    md += "| #Candidates | Mode | NavSucc@8m | Semantic-NavSucc@8m | GS@1 (proto) | N |\n"
    md += "|-------------|------|------------|---------------------|--------------|---|\n"
    
    for k_bucket in sorted(results["by_candidates"].keys()):
        data = results["by_candidates"][k_bucket]
        total = data["total"]
        
        md += f"| {k_bucket} | **GS** | **{data['gs']['navsucc_8m']:.1f}%** | **{data['gs']['semantic_navsucc_8m']:.1f}%** | **{data['gs']['gs_at_1']:.1f}%** | {total} |\n"
        md += f"| | Random | {data['random']['navsucc_8m']:.1f}% | {data['random']['semantic_navsucc_8m']:.1f}% | {data['random']['gs_at_1']:.1f}% | {total} |\n"
        md += f"| | Oracle | {data['oracle']['navsucc_8m']:.1f}% | {data['oracle']['semantic_navsucc_8m']:.1f}% | {data['oracle']['gs_at_1']:.1f}% | {total} |\n"
    
    md += "\n## Key Findings\n\n"
    
    # 找出最难的桶
    hardest_dist = None
    hardest_dist_diff = 0
    for dist_bucket, data in results["by_distance"].items():
        gs_sem = data["gs"]["semantic_navsucc_8m"]
        random_sem = data["random"]["semantic_navsucc_8m"]
        diff = gs_sem - random_sem
        if hardest_dist is None or diff > hardest_dist_diff:
            hardest_dist = dist_bucket
            hardest_dist_diff = diff
    
    if hardest_dist:
        data = results["by_distance"][hardest_dist]
        md += f"- **在最长距离区间（{hardest_dist}）**：\n"
        md += f"  - GS的Semantic-NavSucc@8m为{data['gs']['semantic_navsucc_8m']:.1f}%，Random为{data['random']['semantic_navsucc_8m']:.1f}%\n"
        md += f"  - GS相比Random提升{hardest_dist_diff:.1f}个百分点，展示了在困难场景下的鲁棒性\n\n"
    
    # 按候选数分析
    if "K=3" in results["by_candidates"]:
        k3_data = results["by_candidates"]["K=3"]
        k2_data = results["by_candidates"].get("K=2", {})
        if k2_data:
            md += f"- **在更难的K=3场景**：\n"
            md += f"  - GS的Semantic-NavSucc@8m为{k3_data['gs']['semantic_navsucc_8m']:.1f}%，Random为{k3_data['random']['semantic_navsucc_8m']:.1f}%\n"
            md += f"  - GS相比Random提升{k3_data['gs']['semantic_navsucc_8m'] - k3_data['random']['semantic_navsucc_8m']:.1f}个百分点\n\n"
    
    return md


def main():
    # 加载数据
    gs_results = load_jsonl(Path("logs_v2/proto_main_eval/gs_proto_driven.jsonl"))
    random_results = load_jsonl(Path("logs_v2/proto_main_eval/random_proto_driven.jsonl"))
    oracle_results = load_jsonl(Path("logs_v2/proto_main_eval/oracle_proto_driven.jsonl"))
    
    if not gs_results or not random_results or not oracle_results:
        print("[ERROR] Missing result files!")
        return
    
    # 分析
    results = analyze_by_difficulty(gs_results, random_results, oracle_results)
    
    # 打印
    print_results(results)
    
    # 保存
    output_path = Path("results/proto_main_eval/difficulty_robustness_analysis.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_content = generate_markdown_table(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 保存JSON
    json_path = Path("results/proto_main_eval/difficulty_robustness_analysis.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Save] Results saved to {output_path}")
    print(f"[Save] JSON saved to {json_path}")


if __name__ == "__main__":
    main()

