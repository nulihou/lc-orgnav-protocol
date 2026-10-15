#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定性可视化：轨迹和案例分析

从现有日志中提取代表性episodes，生成轨迹对比图
"""

import json
import sys
import io
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    results = []
    if not path.exists():
        return results
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return results


def extract_trajectory(episode_data: Dict[str, Any]) -> List[Tuple[float, float]]:
    """提取轨迹点"""
    traj = episode_data.get("trajectory", [])
    if not traj:
        return []
    
    points = []
    for point in traj:
        if isinstance(point, dict):
            x = point.get("x", 0.0)
            y = point.get("y", 0.0)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        points.append((float(x), float(y)))
    
    return points


def find_representative_episodes(
    gs_results: List[Dict[str, Any]],
    random_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """找到代表性episodes"""
    
    # 按scen_id组织
    gs_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in gs_results}
    random_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in random_results}
    
    representatives = {}
    
    # 1. GS成功，Random失败
    for scen_id, gs_r in gs_by_scen.items():
        random_r = random_by_scen.get(scen_id, {})
        if (gs_r.get("gs_correct", False) and gs_r.get("success_8m", False) and
            not random_r.get("success_8m", False)):
            representatives["gs_success_random_fail"] = {
                "scen_id": scen_id,
                "gs": gs_r,
                "random": random_r,
            }
            break
    
    # 2. GS选对但失败，Random选错但成功
    for scen_id, gs_r in gs_by_scen.items():
        random_r = random_by_scen.get(scen_id, {})
        if (gs_r.get("gs_correct", False) and not gs_r.get("success_8m", False) and
            not random_r.get("gs_correct", False) and random_r.get("success_8m", False)):
            representatives["gs_correct_fail_random_wrong_success"] = {
                "scen_id": scen_id,
                "gs": gs_r,
                "random": random_r,
            }
            break
    
    # 3. GS和Random都成功，但GS选对，Random选错
    for scen_id, gs_r in gs_by_scen.items():
        random_r = random_by_scen.get(scen_id, {})
        if (gs_r.get("gs_correct", False) and gs_r.get("success_8m", False) and
            not random_r.get("gs_correct", False) and random_r.get("success_8m", False)):
            representatives["both_success_gs_correct"] = {
                "scen_id": scen_id,
                "gs": gs_r,
                "random": random_r,
            }
            break
    
    return representatives


def plot_trajectory_comparison(
    gs_traj: List[Tuple[float, float]],
    random_traj: List[Tuple[float, float]],
    gs_data: Dict[str, Any],
    random_data: Dict[str, Any],
    output_path: Path,
    title: str = "Trajectory Comparison",
):
    """绘制轨迹对比图"""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 获取目标车辆位置（目标地点）
    target_pos_gs = gs_data.get("target_position", {})
    target_pos_random = random_data.get("target_position", {})
    # 优先使用GS的目标位置，如果GS没有则使用Random的
    target_pos = target_pos_gs if target_pos_gs else target_pos_random
    
    # 绘制GS轨迹
    if gs_traj and len(gs_traj) > 0:
        gs_xs = [p[0] for p in gs_traj]
        gs_ys = [p[1] for p in gs_traj]
        ax.plot(gs_xs, gs_ys, 'b-', linewidth=2.5, label='GS Trajectory', alpha=0.7, zorder=1)
        
        # 明确标记起点
        ax.plot(gs_xs[0], gs_ys[0], 'go', markersize=15, markeredgewidth=2, 
                markeredgecolor='darkgreen', label='GS Start', zorder=6)
        ax.annotate('GS Start', xy=(gs_xs[0], gs_ys[0]), xytext=(10, 10),
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='darkgreen'),
                   zorder=7)
        
        # 明确标记ego终点（ego的最终位置）
        ax.plot(gs_xs[-1], gs_ys[-1], 'bs', markersize=12, markeredgewidth=2,
                markeredgecolor='darkblue', label='GS Ego End', zorder=6, alpha=0.7)
    
    # 绘制Random轨迹
    if random_traj and len(random_traj) > 0:
        random_xs = [p[0] for p in random_traj]
        random_ys = [p[1] for p in random_traj]
        ax.plot(random_xs, random_ys, 'r--', linewidth=2.5, label='Random Trajectory', alpha=0.7, zorder=1)
        
        # 明确标记起点
        ax.plot(random_xs[0], random_ys[0], 'go', markersize=15, markeredgewidth=2,
                markeredgecolor='darkgreen', label='Random Start', zorder=6)
        ax.annotate('Random Start', xy=(random_xs[0], random_ys[0]), xytext=(-10, 10),
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', color='darkgreen'),
                   zorder=7)
        
        # 明确标记ego终点（ego的最终位置）
        ax.plot(random_xs[-1], random_ys[-1], 'rs', markersize=12, markeredgewidth=2,
                markeredgecolor='darkred', label='Random Ego End', zorder=6, alpha=0.7)
    
    # 标记目标车辆位置（真正的目标地点）
    if target_pos and "x" in target_pos and "y" in target_pos:
        target_x = target_pos["x"]
        target_y = target_pos["y"]
        # 使用大号星形标记目标地点
        ax.plot(target_x, target_y, '*', markersize=20, markeredgewidth=3,
                markeredgecolor='red', markerfacecolor='yellow', 
                label='Target Location', zorder=8)
        ax.annotate('Target Location', xy=(target_x, target_y), xytext=(15, 15),
                   textcoords='offset points', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='red', lw=2),
                   zorder=9)
    
    # 添加文本信息框（更详细的统计信息）
    info_lines = [
        "=== GS Mode ===",
        f"Min Distance: {gs_data.get('min_dist', 0):.2f}m",
        f"Start Distance: {gs_data.get('start_dist', 0):.2f}m",
        f"Final Distance: {gs_data.get('final_dist', 0):.2f}m",
        f"Success@8m: {'✓' if gs_data.get('success_8m', False) else '✗'}",
        f"Success@4m: {'✓' if gs_data.get('success_4m', False) else '✗'}",
        f"Correct Selection: {'✓' if gs_data.get('gs_correct', False) else '✗'}",
        f"Steps: {gs_data.get('steps', 0)}",
        "",
        "=== Random Mode ===",
        f"Min Distance: {random_data.get('min_dist', 0):.2f}m",
        f"Start Distance: {random_data.get('start_dist', 0):.2f}m",
        f"Final Distance: {random_data.get('final_dist', 0):.2f}m",
        f"Success@8m: {'✓' if random_data.get('success_8m', False) else '✗'}",
        f"Success@4m: {'✓' if random_data.get('success_4m', False) else '✗'}",
        f"Correct Selection: {'✓' if random_data.get('gs_correct', False) else '✗'}",
        f"Steps: {random_data.get('steps', 0)}",
    ]
    
    # 如果有目标位置信息，添加到说明中
    if target_pos and "x" in target_pos and "y" in target_pos:
        info_lines.append("")
        info_lines.append(f"Target Location: ({target_pos['x']:.2f}, {target_pos['y']:.2f})")
    
    info_text = "\n".join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5),
            fontsize=9, family='monospace', zorder=8)
    
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 改进图例，避免重复
    handles, labels = ax.get_legend_handles_labels()
    # 去重，保留顺序
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    
    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=10, 
              framealpha=0.9, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # 添加图例说明
    note_text = 'Legend: Start (green circle), Ego End (colored square), Target Location (yellow star)'
    ax.text(0.98, 0.02, note_text,
            transform=ax.transAxes, fontsize=9, style='italic',
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            zorder=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [Save] Saved to {output_path}")


def main():
    # 优先使用轨迹评估数据（包含完整轨迹和目标位置），否则使用测试文件，最后使用主实验数据
    gs_path = Path("logs_v2/proto_main_eval/trajectory_eval/gs_proto_driven.jsonl")
    random_path = Path("logs_v2/proto_main_eval/trajectory_eval/random_proto_driven.jsonl")
    
    if not gs_path.exists():
        gs_path = Path("logs_v2/proto_main_eval/gs_proto_driven_test.jsonl")
    if not random_path.exists():
        random_path = Path("logs_v2/proto_main_eval/random_proto_driven_test.jsonl")
    
    if not gs_path.exists():
        gs_path = Path("logs_v2/proto_main_eval/gs_proto_driven.jsonl")
    if not random_path.exists():
        random_path = Path("logs_v2/proto_main_eval/random_proto_driven.jsonl")
    
    # 加载数据
    gs_results = load_jsonl(gs_path)
    random_results = load_jsonl(random_path)
    
    print(f"[INFO] 使用GS文件: {gs_path}")
    print(f"[INFO] 使用Random文件: {random_path}")
    
    if not gs_results or not random_results:
        print("[ERROR] Missing result files!")
        return
    
    # 找到代表性episodes
    representatives = find_representative_episodes(gs_results, random_results)
    
    # 如果没有找到代表性案例，至少显示所有有轨迹数据的案例
    if not representatives:
        print("[WARNING] No representative episodes found!")
        print("[INFO] Trying to visualize all available episodes with trajectory data...")
        
        # 按scen_id组织
        gs_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in gs_results}
        random_by_scen = {r.get("scen_id", r.get("episode", 0)): r for r in random_results}
        
        # 找到所有有轨迹数据的场景
        for scen_id in sorted(set(list(gs_by_scen.keys()) + list(random_by_scen.keys()))):
            gs_r = gs_by_scen.get(scen_id, {})
            random_r = random_by_scen.get(scen_id, {})
            
            gs_traj = extract_trajectory(gs_r)
            random_traj = extract_trajectory(random_r)
            
            if (gs_traj and len(gs_traj) > 0) or (random_traj and len(random_traj) > 0):
                representatives[f"all_episode_scen_{scen_id:03d}"] = {
                    "scen_id": scen_id,
                    "gs": gs_r,
                    "random": random_r,
                }
        
        if not representatives:
            print("[ERROR] No episodes with trajectory data found!")
            return
    
    # 创建输出目录
    output_dir = Path("results/proto_main_eval/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Case Study Visualization")
    print("="*80)
    print()
    
    # 绘制每个代表性案例
    for case_name, case_data in representatives.items():
        scen_id = case_data["scen_id"]
        gs_data = case_data["gs"]
        random_data = case_data["random"]
        
        # 如果数据中没有target_position，尝试从场景JSON获取
        if "target_position" not in gs_data and "target_position" not in random_data:
            print(f"  [INFO] scen_{scen_id:03d}: No target_position in data, trying to get from scenario JSON...")
            target_pos_info = get_target_position_from_scenario(scen_id)
            if target_pos_info:
                # 注意：这里只能获取spawn_idx，实际位置需要从CARLA获取
                # 为了简化，我们暂时跳过目标位置标记
                print(f"  [INFO] Found target_spawn_idx={target_pos_info.get('spawn_idx')}, but cannot get actual position without CARLA")
        
        gs_traj = extract_trajectory(gs_data)
        random_traj = extract_trajectory(random_data)
        
        if (not gs_traj or len(gs_traj) == 0) and (not random_traj or len(random_traj) == 0):
            print(f"  [SKIP] scen_{scen_id:03d}: No trajectory data (GS: {len(gs_traj) if gs_traj else 0}, Random: {len(random_traj) if random_traj else 0})")
            continue
        
        # 如果只有一个有轨迹，也绘制
        if not gs_traj or len(gs_traj) == 0:
            print(f"  [WARNING] scen_{scen_id:03d}: GS has no trajectory, only Random")
        if not random_traj or len(random_traj) == 0:
            print(f"  [WARNING] scen_{scen_id:03d}: Random has no trajectory, only GS")
        
        title = f"Case Study: scen_{scen_id:03d} ({case_name.replace('_', ' ').title()})"
        output_path = output_dir / f"case_study_{case_name}_scen_{scen_id:03d}.png"
        
        plot_trajectory_comparison(
            gs_traj=gs_traj,
            random_traj=random_traj,
            gs_data=gs_data,
            random_data=random_data,
            output_path=output_path,
            title=title,
        )
    
    print()
    print(f"[Done] Generated {len(representatives)} case study visualizations")
    print(f"[Output] Directory: {output_dir}")


if __name__ == "__main__":
    main()

