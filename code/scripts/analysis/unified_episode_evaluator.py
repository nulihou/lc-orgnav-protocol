#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的Episode评测函数（Step 1.1）

强制统一口径，确保所有评测指标使用相同的计算逻辑。

输入：episode日志（每步ego/goal位置速度、选中目标id、GT目标id）
输出：标准化CSV字段（每个episode一行）
"""

import json
import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def evaluate_episode(
    episode_log: Dict[str, Any],
    trajectory: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    统一的episode评测函数
    
    输入：
        episode_log: 包含以下字段的字典：
            - episode: int, episode ID
            - mode: str, 评估模式（"gs", "random", "oracle"）
            - scen_id: int, 场景ID
            - target_idx_gt: int, GT目标在协议图中的索引
            - chosen_idx_graph: int, 被选择目标在协议图中的索引
            - trajectory: List[Dict], 每步的轨迹数据（可选，如果episode_log中没有）
                每步包含：
                    - ego_position: [x, y, z] 或 Dict with x, y, z
                    - target_gt_position: [x, y, z] 或 Dict with x, y, z（GT目标位置）
                    - target_selected_position: [x, y, z] 或 Dict with x, y, z（被选择目标位置）
                    - step: int, 步数
                    - timeout: bool（可选）
                    - collision: bool（可选）
                    - stuck: bool（可选）
            - min_dist: float（可选，如果trajectory中没有，则从trajectory计算）
            - success_4m: bool（可选，如果trajectory中没有，则从trajectory计算）
            - success_8m: bool（可选，如果trajectory中没有，则从trajectory计算）
            - stuck: bool（可选）
            - timeout: bool（可选）
            - collision: bool（可选）
    
    输出：
        标准化字段字典：
            - is_correct: bool, (y_hat == y_gt)
            - d_min_to_gt: float, 全程ego到GT目标车的最小距离
            - d_min_to_selected: float, 全程ego到被选择目标车的最小距离
            - navsucc8: bool, (d_min_to_selected < 8m)
            - navsucc4: bool, (d_min_to_selected < 4m)
            - semnavsucc8: bool, is_correct AND (d_min_to_gt < 8m)
            - semnavsucc4: bool, is_correct AND (d_min_to_gt < 4m)
            - timeout: bool
            - collision: bool
            - stuck: bool
            - episode: int
            - mode: str
            - scen_id: int
    """
    # 处理valid=false的失败记录（确保分母完整，避免选择性报告）
    valid = episode_log.get("valid", True)  # 默认True以兼容旧日志
    if valid is False:
        # 直接返回失败结果，但保留基本信息（确保分母=150）
        episode = episode_log.get("episode", -1)
        mode = episode_log.get("mode", "unknown")
        scen_id = episode_log.get("scen_id", -1)
        target_idx_gt = episode_log.get("target_idx_gt", None)
        chosen_idx_graph = episode_log.get("chosen_idx_graph", None)
        
        # 计算is_correct（即使valid=false，Oracle模式也应该is_correct=True）
        if target_idx_gt is not None and chosen_idx_graph is not None:
            is_correct = (chosen_idx_graph == target_idx_gt)
        else:
            # 尝试从其他字段推断
            gs_correct = episode_log.get("gs_correct", None)
            oracle_correct = episode_log.get("oracle_correct", None)
            if gs_correct is not None:
                is_correct = bool(gs_correct)
            elif oracle_correct is not None:
                is_correct = bool(oracle_correct)
            else:
                is_correct = False
        
        return {
            "episode": int(episode),
            "mode": str(mode),
            "scen_id": int(scen_id),
            "is_correct": bool(is_correct),  # 根据target_idx_gt和chosen_idx_graph计算
            "d_min_to_gt": None,
            "d_min_to_selected": None,
            "navsucc8": False,
            "navsucc4": False,
            "semnavsucc8": False,
            "semnavsucc4": False,
            "timeout": bool(episode_log.get("timeout", False)),
            "collision": bool(episode_log.get("collision", False)),
            "stuck": bool(episode_log.get("stuck", False)),
            "target_idx_gt": target_idx_gt,
            "chosen_idx_graph": chosen_idx_graph,
            "valid": False,  # 标记为无效记录
            "error_type": episode_log.get("error_type"),
            "error_msg": episode_log.get("error_msg"),
        }
    
    # 提取基本信息
    episode = episode_log.get("episode", -1)
    mode = episode_log.get("mode", "unknown")
    scen_id = episode_log.get("scen_id", -1)
    target_idx_gt = episode_log.get("target_idx_gt", None)
    chosen_idx_graph = episode_log.get("chosen_idx_graph", None)
    
    # 1. 计算 is_correct
    if target_idx_gt is not None and chosen_idx_graph is not None:
        is_correct = (chosen_idx_graph == target_idx_gt)
    else:
        # 如果没有这些字段，尝试从其他字段推断
        gs_correct = episode_log.get("gs_correct", None)
        if gs_correct is not None:
            is_correct = bool(gs_correct)
        else:
            is_correct = False
    
    # 2. 从trajectory计算距离
    traj = trajectory if trajectory is not None else episode_log.get("trajectory", [])
    
    d_min_to_gt = float('inf')
    d_min_to_selected = float('inf')
    
    if traj and len(traj) > 0:
        # 从trajectory计算最小距离
        for step_data in traj:
            # 提取ego位置
            ego_pos = step_data.get("ego_position") or step_data.get("ego_pos")
            if isinstance(ego_pos, dict):
                ego_x, ego_y, ego_z = ego_pos.get("x", 0), ego_pos.get("y", 0), ego_pos.get("z", 0)
            elif isinstance(ego_pos, (list, tuple)) and len(ego_pos) >= 3:
                ego_x, ego_y, ego_z = ego_pos[0], ego_pos[1], ego_pos[2]
            else:
                continue
            
            # 提取GT目标位置
            gt_pos = step_data.get("target_gt_position") or step_data.get("target_gt_pos")
            if gt_pos:
                if isinstance(gt_pos, dict):
                    gt_x, gt_y, gt_z = gt_pos.get("x", 0), gt_pos.get("y", 0), gt_pos.get("z", 0)
                elif isinstance(gt_pos, (list, tuple)) and len(gt_pos) >= 3:
                    gt_x, gt_y, gt_z = gt_pos[0], gt_pos[1], gt_pos[2]
                else:
                    gt_x, gt_y, gt_z = None, None, None
                
                if gt_x is not None:
                    dist_gt = math.sqrt((ego_x - gt_x)**2 + (ego_y - gt_y)**2 + (ego_z - gt_z)**2)
                    d_min_to_gt = min(d_min_to_gt, dist_gt)
            
            # 提取被选择目标位置
            sel_pos = step_data.get("target_selected_position") or step_data.get("target_selected_pos")
            if sel_pos:
                if isinstance(sel_pos, dict):
                    sel_x, sel_y, sel_z = sel_pos.get("x", 0), sel_pos.get("y", 0), sel_pos.get("z", 0)
                elif isinstance(sel_pos, (list, tuple)) and len(sel_pos) >= 3:
                    sel_x, sel_y, sel_z = sel_pos[0], sel_pos[1], sel_pos[2]
                else:
                    sel_x, sel_y, sel_z = None, None, None
                
                if sel_x is not None:
                    dist_sel = math.sqrt((ego_x - sel_x)**2 + (ego_y - sel_y)**2 + (ego_z - sel_z)**2)
                    d_min_to_selected = min(d_min_to_selected, dist_sel)
    
    # 如果trajectory中没有，尝试从episode_log中获取
    if d_min_to_gt == float('inf'):
        # 尝试从min_dist推断（如果只有一个目标）
        min_dist = episode_log.get("min_dist", None)
        if min_dist is not None:
            # 如果选对了，min_dist就是到GT的距离
            if is_correct:
                d_min_to_gt = float(min_dist)
                d_min_to_selected = float(min_dist)
            else:
                # 如果选错了，我们无法区分，设为inf
                d_min_to_gt = float('inf')
                d_min_to_selected = float(min_dist)
    
    # 如果还是inf，设为NaN（表示无法计算）
    if d_min_to_gt == float('inf'):
        d_min_to_gt = float('nan')
    if d_min_to_selected == float('inf'):
        d_min_to_selected = float('nan')
    
    # 3. 计算导航成功指标
    navsucc8 = (d_min_to_selected < 8.0) if not math.isnan(d_min_to_selected) else False
    navsucc4 = (d_min_to_selected < 4.0) if not math.isnan(d_min_to_selected) else False
    
    # 4. 计算语义导航成功指标（关键：使用GT距离）
    semnavsucc8 = is_correct and (d_min_to_gt < 8.0) if not math.isnan(d_min_to_gt) else False
    semnavsucc4 = is_correct and (d_min_to_gt < 4.0) if not math.isnan(d_min_to_gt) else False
    
    # 5. 提取其他标志
    timeout = bool(episode_log.get("timeout", False))
    collision = bool(episode_log.get("collision", False))
    stuck = bool(episode_log.get("stuck", False))
    
    # 如果trajectory中有这些信息，也检查
    if traj:
        for step_data in traj:
            if step_data.get("timeout", False):
                timeout = True
            if step_data.get("collision", False):
                collision = True
            if step_data.get("stuck", False):
                stuck = True
    
    # 构建输出字典
    result = {
        "episode": int(episode),
        "mode": str(mode),
        "scen_id": int(scen_id),
        "is_correct": bool(is_correct),
        "d_min_to_gt": float(d_min_to_gt) if not math.isnan(d_min_to_gt) else None,
        "d_min_to_selected": float(d_min_to_selected) if not math.isnan(d_min_to_selected) else None,
        "navsucc8": bool(navsucc8),
        "navsucc4": bool(navsucc4),
        "semnavsucc8": bool(semnavsucc8),
        "semnavsucc4": bool(semnavsucc4),
        "timeout": bool(timeout),
        "collision": bool(collision),
        "stuck": bool(stuck),
        "target_idx_gt": int(target_idx_gt) if target_idx_gt is not None else None,
        "chosen_idx_graph": int(chosen_idx_graph) if chosen_idx_graph is not None else None,
    }
    
    return result


def evaluate_episodes_from_jsonl(
    jsonl_path: str,
    output_csv_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从JSONL文件读取episodes并评估
    
    Args:
        jsonl_path: 输入JSONL文件路径
        output_csv_path: 输出CSV文件路径（可选）
    
    Returns:
        (evaluated_results, errors)
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    evaluated_results = []
    errors = []
    
    # 读取JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                episode_log = json.loads(line)
                
                # 评估episode
                result = evaluate_episode(episode_log)
                evaluated_results.append(result)
                
            except Exception as e:
                errors.append({
                    "line": line_idx,
                    "error": str(e),
                    "line_content": line[:100] if len(line) > 100 else line
                })
    
    # 写入CSV（如果指定了输出路径）
    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if evaluated_results:
            fieldnames = list(evaluated_results[0].keys())
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in evaluated_results:
                    # 将None转换为空字符串以便CSV写入
                    row = {k: (v if v is not None else '') for k, v in result.items()}
                    writer.writerow(row)
    
    return evaluated_results, errors


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一Episode评测函数")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSONL文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出CSV文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] 读取JSONL文件: {args.input}")
    results, errors = evaluate_episodes_from_jsonl(args.input, args.output)
    
    print(f"[INFO] 成功评估 {len(results)} 个episodes")
    if errors:
        print(f"[WARNING] 遇到 {len(errors)} 个错误")
        for err in errors[:5]:  # 只显示前5个错误
            print(f"  Line {err['line']}: {err['error']}")
    
    if args.output:
        print(f"[INFO] 结果已写入: {args.output}")
    
    # 打印统计信息
    if results:
        modes = set(r["mode"] for r in results)
        print(f"\n[统计]")
        print(f"  总episodes: {len(results)}")
        print(f"  模式: {', '.join(modes)}")
        
        for mode in modes:
            mode_results = [r for r in results if r["mode"] == mode]
            n = len(mode_results)
            if n > 0:
                correct = sum(1 for r in mode_results if r["is_correct"])
                navsucc8 = sum(1 for r in mode_results if r["navsucc8"])
                semnavsucc8 = sum(1 for r in mode_results if r["semnavsucc8"])
                print(f"  {mode}: {n} episodes, "
                      f"Correct={correct}/{n} ({100*correct/n:.1f}%), "
                      f"NavSucc@8m={navsucc8}/{n} ({100*navsucc8/n:.1f}%), "
                      f"SemNavSucc@8m={semnavsucc8}/{n} ({100*semnavsucc8/n:.1f}%)")


if __name__ == "__main__":
    main()

