#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总协议图评估结果

读取三份 JSONL 文件，计算统计指标并生成报告。
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


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return []
    
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError:
                continue
    
    return results


def summarize_results(
    gs_jsonl: str,
    random_jsonl: str,
    oracle_jsonl: str,
) -> Dict[str, Any]:
    """汇总评估结果"""
    print("=" * 80)
    print("Protocol Graph Evaluation Summary")
    print("=" * 80)
    
    # 加载结果
    gs_results = load_jsonl(gs_jsonl)
    random_results = load_jsonl(random_jsonl)
    oracle_results = load_jsonl(oracle_jsonl)
    
    print(f"\n[Load]")
    print(f"  GS:     {len(gs_results)} scenarios")
    print(f"  Random: {len(random_results)} scenarios")
    print(f"  Oracle: {len(oracle_results)} scenarios")
    
    # 计算GS指标
    if gs_results:
        gs_correct = sum(1 for r in gs_results if r.get("gs_correct", False))
        gs_accuracy = gs_correct / len(gs_results) * 100
        print(f"\n[GS Metrics]")
        print(f"  GS@1 Accuracy: {gs_correct}/{len(gs_results)} = {gs_accuracy:.1f}%")
    
    # 计算Random指标
    if random_results:
        random_correct = sum(1 for r in random_results if r.get("gs_correct", False))
        random_accuracy = random_correct / len(random_results) * 100
        print(f"\n[Random Metrics]")
        print(f"  Random Accuracy: {random_correct}/{len(random_results)} = {random_accuracy:.1f}%")
    
    # 计算Oracle指标（应该总是100%）
    if oracle_results:
        oracle_correct = sum(1 for r in oracle_results if r.get("gs_correct", False))
        oracle_accuracy = oracle_correct / len(oracle_results) * 100
        print(f"\n[Oracle Metrics]")
        print(f"  Oracle Accuracy: {oracle_correct}/{len(oracle_results)} = {oracle_accuracy:.1f}%")
    
    # 对比
    if gs_results and random_results:
        improvement = gs_accuracy - random_accuracy
        print(f"\n[Comparison]")
        print(f"  GS vs Random: {improvement:+.1f}% improvement")
    
    return {
        "gs": {
            "total": len(gs_results),
            "correct": gs_correct if gs_results else 0,
            "accuracy": gs_accuracy if gs_results else 0.0,
        },
        "random": {
            "total": len(random_results),
            "correct": random_correct if random_results else 0,
            "accuracy": random_accuracy if random_results else 0.0,
        },
        "oracle": {
            "total": len(oracle_results),
            "correct": oracle_correct if oracle_results else 0,
            "accuracy": oracle_accuracy if oracle_results else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize Protocol Graph Evaluation Results")
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
        help="Output JSON summary path (optional)"
    )
    
    args = parser.parse_args()
    
    # 汇总结果
    summary = summarize_results(
        gs_jsonl=args.gs_jsonl,
        random_jsonl=args.random_jsonl,
        oracle_jsonl=args.oracle_jsonl,
    )
    
    # 保存汇总（如果需要）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[Report] Summary saved to {args.output}")


if __name__ == "__main__":
    main()

