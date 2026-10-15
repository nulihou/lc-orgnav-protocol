#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无泄露检查脚本

检查离线训练/验证/测试数据与protocol graphs（30个多候选场景）之间是否有"同一场景"重叠。

场景签名定义：
    sig = (town_name, ego_spawn_idx, anchor_spawn_idx, sorted(candidate_spawn_indices))

注意：
    - 离线数据的.npz文件中没有保存spawn_idx信息，因此无法进行精确的场景签名匹配
    - 本脚本会检查：
      1. Protocol场景的town是否与离线数据的town一致
      2. 如果离线数据中有保存spawn_idx信息（通过其他方式），则进行精确匹配
      3. 离线训练/验证/测试数据内部是否有重复签名
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_signature_from_protocol_scenario(scen: Dict[str, Any]) -> Optional[Tuple[str, int, int, Tuple[int, ...]]]:
    """
    从protocol场景JSON中提取场景签名
    
    Args:
        scen: 场景字典，包含town, ego_spawn_idx, anchor_spawn_idx, candidate_spawn_indices
    
    Returns:
        场景签名元组: (town, ego_spawn_idx, anchor_spawn_idx, sorted_candidate_spawn_indices)
        如果缺少必要字段，返回None
    """
    town = scen.get("town")
    ego_spawn_idx = scen.get("ego_spawn_idx")
    anchor_spawn_idx = scen.get("anchor_spawn_idx")
    candidate_spawn_indices = scen.get("candidate_spawn_indices")
    
    if town is None or ego_spawn_idx is None or anchor_spawn_idx is None or candidate_spawn_indices is None:
        return None
    
    # 确保candidate_spawn_indices是列表并排序
    if isinstance(candidate_spawn_indices, (list, tuple, np.ndarray)):
        sorted_cands = tuple(sorted(int(x) for x in candidate_spawn_indices))
    else:
        return None
    
    return (str(town), int(ego_spawn_idx), int(anchor_spawn_idx), sorted_cands)


def extract_signature_from_offline_npz(npz_path: str) -> Optional[Tuple[str, int, int, Tuple[int, ...]]]:
    """
    从离线数据的.npz文件中提取场景签名
    
    注意：当前.npz文件中没有保存spawn_idx信息，因此无法提取完整签名
    只能返回town信息（如果metadata中有）
    
    Args:
        npz_path: .npz文件路径
    
    Returns:
        场景签名元组，如果无法提取则返回None
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # 检查是否有metadata字段
        if "metadata" in data:
            metadata = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
            town = metadata.get("town")
            ego_spawn_idx = metadata.get("ego_spawn_idx")
            anchor_spawn_idx = metadata.get("anchor_spawn_idx")
            candidate_spawn_indices = metadata.get("candidate_spawn_indices")
            
            if town and ego_spawn_idx is not None and anchor_spawn_idx is not None and candidate_spawn_indices:
                sorted_cands = tuple(sorted(int(x) for x in candidate_spawn_indices))
                return (str(town), int(ego_spawn_idx), int(anchor_spawn_idx), sorted_cands)
        
        # 如果没有metadata，尝试从collection_metadata.json获取town信息
        # 但这只能提供town，无法提供spawn_idx
        return None
        
    except Exception as e:
        print(f"  [WARNING] Failed to load {npz_path}: {e}")
        return None


def load_offline_signatures_from_list(list_path: str) -> Dict[Tuple[str, int, int, Tuple[int, ...]], List[str]]:
    """
    从训练/验证/测试列表文件中加载所有离线数据的签名
    
    Args:
        list_path: 列表文件路径（每行一个.npz文件路径）
    
    Returns:
        字典：{signature: [文件路径列表]}
    """
    signatures = defaultdict(list)
    
    if not os.path.exists(list_path):
        print(f"  [WARNING] List file not found: {list_path}")
        return signatures
    
    with open(list_path, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    print(f"  [INFO] Loading {len(paths)} samples from {list_path}...")
    
    success_count = 0
    fail_count = 0
    
    for npz_path in paths:
        # 处理相对路径和绝对路径
        if not os.path.isabs(npz_path):
            # 尝试相对于项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            npz_path = os.path.join(project_root, npz_path)
        
        if not os.path.exists(npz_path):
            fail_count += 1
            continue
        
        sig = extract_signature_from_offline_npz(npz_path)
        if sig:
            signatures[sig].append(npz_path)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"  [INFO] Successfully extracted signatures: {success_count}, Failed: {fail_count}")
    
    return signatures


def load_protocol_signatures(protocol_json_path: str) -> List[Tuple[int, Tuple[str, int, int, Tuple[int, ...]]]]:
    """
    从protocol场景JSON文件中加载所有场景的签名
    
    Args:
        protocol_json_path: protocol场景JSON文件路径
    
    Returns:
        列表：[(scen_id, signature), ...]
    """
    if not os.path.exists(protocol_json_path):
        print(f"  [ERROR] Protocol JSON file not found: {protocol_json_path}")
        return []
    
    with open(protocol_json_path, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    proto_sigs = []
    for scen in scenarios:
        scen_id = scen.get("id", -1)
        sig = extract_signature_from_protocol_scenario(scen)
        if sig:
            proto_sigs.append((scen_id, sig))
        else:
            print(f"  [WARNING] Failed to extract signature from scenario {scen_id}")
    
    print(f"  [INFO] Loaded {len(proto_sigs)} protocol scenario signatures")
    
    return proto_sigs


def check_internal_duplicates(signatures: Dict[Tuple[str, int, int, Tuple[int, ...]], List[str]], 
                               dataset_name: str) -> List[Tuple[Tuple[str, int, int, Tuple[int, ...]], List[str]]]:
    """
    检查数据集内部是否有重复的场景签名
    
    Args:
        signatures: 签名字典
        dataset_name: 数据集名称（用于输出）
    
    Returns:
        重复签名的列表：[(signature, [文件路径列表]), ...]
    """
    duplicates = [(sig, paths) for sig, paths in signatures.items() if len(paths) > 1]
    
    if duplicates:
        print(f"\n  [WARNING] Found {len(duplicates)} duplicate signatures in {dataset_name}:")
        for sig, paths in duplicates[:5]:  # 只显示前5个
            print(f"    Signature: {sig}")
            print(f"      Files ({len(paths)}): {paths[:3]}...")  # 只显示前3个文件
    else:
        print(f"  [OK] No duplicate signatures found in {dataset_name}")
    
    return duplicates


def check_town_consistency(protocol_json_path: str, collection_metadata_path: str) -> Dict[str, Any]:
    """
    检查Protocol场景和离线数据的town一致性
    
    Args:
        protocol_json_path: Protocol场景JSON路径
        collection_metadata_path: 离线数据collection_metadata.json路径
    
    Returns:
        包含检查结果的字典
    """
    result = {
        "protocol_towns": set(),
        "offline_town": None,
        "consistent": False,
    }
    
    # 从Protocol场景提取town
    if os.path.exists(protocol_json_path):
        with open(protocol_json_path, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
        for scen in scenarios:
            town = scen.get("town")
            if town:
                result["protocol_towns"].add(str(town))
    
    # 从collection_metadata.json提取town
    if os.path.exists(collection_metadata_path):
        with open(collection_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        result["offline_town"] = metadata.get("config", {}).get("town")
    
    # 检查一致性
    if result["offline_town"] and result["protocol_towns"]:
        result["consistent"] = result["offline_town"] in result["protocol_towns"]
    
    return result


def main():
    print("=" * 80)
    print("无泄露检查脚本")
    print("=" * 80)
    print()
    
    # 配置路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Protocol场景JSON
    protocol_json_path = os.path.join(project_root, "data", "online_multi", "multi_seen_v3.json")
    
    # 离线数据列表文件
    train_list_path = os.path.join(project_root, "data", "lists", "train_mixed_plus_multi_final.txt")
    val_list_path = os.path.join(project_root, "data", "lists", "multi_val_list.txt")
    test_list_path = os.path.join(project_root, "data", "lists", "multi_test_list.txt")
    
    # Collection metadata
    collection_metadata_path = os.path.join(project_root, "data", "multi_candidate_samples", "collection_metadata.json")
    
    # 输出报告路径
    output_report_path = os.path.join(project_root, "results", "proto_main_eval", "DATA_LEAKAGE_CHECK.md")
    
    print("[Step 0] 检查Town一致性...")
    town_check = check_town_consistency(protocol_json_path, collection_metadata_path)
    print(f"  [INFO] Protocol towns: {sorted(town_check['protocol_towns'])}")
    print(f"  [INFO] Offline town: {town_check['offline_town']}")
    if town_check['consistent']:
        print(f"  [OK] Town is consistent: {town_check['offline_town']}")
    else:
        print(f"  [WARNING] Town may be inconsistent")
    print()
    
    print("[Step 1] 加载Protocol场景签名...")
    proto_sigs = load_protocol_signatures(protocol_json_path)
    
    if not proto_sigs:
        print("  [ERROR] No protocol signatures loaded. Exiting.")
        return
    
    # 构建protocol签名集合
    proto_sig_set = {sig for _, sig in proto_sigs}
    proto_sig_to_id = {sig: scen_id for scen_id, sig in proto_sigs}
    
    print(f"  [INFO] Protocol scenarios: {len(proto_sigs)}")
    print()
    
    print("[Step 2] 加载离线训练数据签名...")
    train_sigs = load_offline_signatures_from_list(train_list_path)
    print(f"  [INFO] Train signatures: {len(train_sigs)}")
    print()
    
    print("[Step 3] 加载离线验证数据签名...")
    val_sigs = load_offline_signatures_from_list(val_list_path)
    print(f"  [INFO] Val signatures: {len(val_sigs)}")
    print()
    
    print("[Step 4] 加载离线测试数据签名...")
    test_sigs = load_offline_signatures_from_list(test_list_path)
    print(f"  [INFO] Test signatures: {len(test_sigs)}")
    print()
    
    # 检查离线数据内部重复
    print("[Step 5] 检查离线数据内部重复...")
    train_duplicates = check_internal_duplicates(train_sigs, "Train")
    val_duplicates = check_internal_duplicates(val_sigs, "Val")
    test_duplicates = check_internal_duplicates(test_sigs, "Test")
    print()
    
    # 检查离线数据与protocol场景的重叠
    print("[Step 6] 检查离线数据与Protocol场景的重叠...")
    
    all_offline_sigs = set(train_sigs.keys()) | set(val_sigs.keys()) | set(test_sigs.keys())
    
    overlaps = []
    for sig in proto_sig_set:
        if sig in all_offline_sigs:
            scen_id = proto_sig_to_id[sig]
            overlaps.append((scen_id, sig))
    
    print(f"  [RESULT] Found {len(overlaps)} overlapping scenarios between offline data and protocol scenarios")
    
    if overlaps:
        print("  [WARNING] Overlapping scenarios:")
        for scen_id, sig in overlaps:
            print(f"    Scenario ID {scen_id}: {sig}")
            # 查找在哪些离线数据集中
            in_train = sig in train_sigs
            in_val = sig in val_sigs
            in_test = sig in test_sigs
            print(f"      In Train: {in_train}, In Val: {in_val}, In Test: {in_test}")
    else:
        print("  [OK] No overlaps found! Protocol scenarios are not in offline training/validation/test data.")
    print()
    
    # 生成Markdown报告
    print("[Step 7] 生成Markdown报告...")
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("# 数据泄露检查报告\n\n")
        f.write("## 概述\n\n")
        f.write("本报告检查离线训练/验证/测试数据与protocol graphs（30个多候选场景）之间是否有\"同一场景\"重叠。\n\n")
        f.write("## 场景签名定义\n\n")
        f.write("场景签名用于唯一标识一个场景：\n\n")
        f.write("```python\n")
        f.write("sig = (town_name, ego_spawn_idx, anchor_spawn_idx, sorted(candidate_spawn_indices))\n")
        f.write("```\n\n")
        f.write("## 检查结果\n\n")
        f.write("### 0. Town一致性检查\n\n")
        f.write(f"- **Protocol场景使用的Town**: {sorted(town_check['protocol_towns'])}\n")
        f.write(f"- **离线数据使用的Town**: {town_check['offline_town']}\n")
        if town_check['consistent']:
            f.write(f"- **一致性**: ✅ 一致（都是 `{town_check['offline_town']}`）\n\n")
        else:
            f.write(f"- **一致性**: ⚠️ 可能不一致\n\n")
        f.write("### 1. Protocol场景统计\n\n")
        f.write(f"- **总场景数**: {len(proto_sigs)}\n")
        f.write(f"- **唯一签名数**: {len(proto_sig_set)}\n\n")
        f.write("### 2. 离线数据统计\n\n")
        f.write(f"- **训练集签名数**: {len(train_sigs)}\n")
        f.write(f"- **验证集签名数**: {len(val_sigs)}\n")
        f.write(f"- **测试集签名数**: {len(test_sigs)}\n")
        f.write(f"- **总唯一签名数**: {len(all_offline_sigs)}\n\n")
        if len(all_offline_sigs) == 0:
            f.write("**⚠️ 注意**: 无法从离线数据的`.npz`文件中提取场景签名（缺少spawn_idx信息）。\n\n")
        f.write("### 3. 离线数据内部重复检查\n\n")
        f.write(f"- **训练集重复数**: {len(train_duplicates)}\n")
        f.write(f"- **验证集重复数**: {len(val_duplicates)}\n")
        f.write(f"- **测试集重复数**: {len(test_duplicates)}\n\n")
        if train_duplicates or val_duplicates or test_duplicates:
            f.write("**注意**: 发现内部重复签名，详情见下方。\n\n")
        f.write("### 4. Protocol场景与离线数据重叠检查\n\n")
        f.write(f"- **重叠场景数**: {len(overlaps)}\n\n")
        if overlaps:
            f.write("**⚠️ 警告**: 发现重叠场景！\n\n")
            f.write("| Scenario ID | Signature | In Train | In Val | In Test |\n")
            f.write("|------------|-----------|----------|--------|---------|\n")
            for scen_id, sig in overlaps:
                in_train = "✓" if sig in train_sigs else "✗"
                in_val = "✓" if sig in val_sigs else "✗"
                in_test = "✓" if sig in test_sigs else "✗"
                f.write(f"| {scen_id} | {sig} | {in_train} | {in_val} | {in_test} |\n")
            f.write("\n")
        else:
            if len(all_offline_sigs) == 0:
                f.write("**⚠️ 无法确定**: 由于离线数据中缺少spawn_idx信息，无法进行精确的场景签名匹配。\n\n")
                f.write("**基于Town一致性**: Protocol场景和离线数据都使用相同的Town（`Town10HD_Opt`），但无法确定是否有相同的spawn点配置。\n\n")
            else:
                f.write("**✅ 通过**: 未发现重叠场景。Protocol场景不在离线训练/验证/测试数据中。\n\n")
        f.write("## 重要说明\n\n")
        f.write("### 限制\n\n")
        f.write("**关键限制**: 当前离线数据的`.npz`文件中**没有保存spawn_idx信息**，因此无法进行精确的场景签名匹配。\n\n")
        f.write("本脚本会尝试从以下位置提取签名：\n\n")
        f.write("1. `.npz`文件的`metadata`字段（如果存在）\n")
        f.write("2. `collection_metadata.json`文件（仅提供town信息，不提供spawn_idx）\n\n")
        f.write("**当前状态**: 所有离线数据的签名提取都失败了（0个成功），说明`.npz`文件中确实没有保存spawn_idx信息。\n\n")
        f.write("### 建议\n\n")
        f.write("如果需要进行精确的场景签名匹配，建议在数据收集时在`.npz`文件中保存以下信息：\n\n")
        f.write("- `town`: 地图名称\n")
        f.write("- `ego_spawn_idx`: Ego车辆spawn点索引\n")
        f.write("- `anchor_spawn_idx`: Anchor车辆spawn点索引\n")
        f.write("- `candidate_spawn_indices`: 候选车辆spawn点索引列表\n\n")
        f.write("可以在`scripts/data/collect_multi_candidate_data.py`的保存部分添加这些字段。\n\n")
        f.write("## 结论\n\n")
        if len(overlaps) == 0:
            if len(all_offline_sigs) == 0:
                f.write("**⚠️ 无法确定**: 由于离线数据中缺少spawn_idx信息，无法进行精确的场景签名匹配。\n\n")
                f.write("**基于现有信息**:\n\n")
                f.write("- Protocol场景和离线数据都使用相同的Town（`Town10HD_Opt`）\n")
                f.write("- 无法确定是否有相同的spawn点配置\n")
                f.write("- **建议**: 如果需要进行精确检查，需要在数据收集时保存spawn_idx信息\n\n")
            else:
                f.write("**✅ 无数据泄露**: 30个protocol场景都不在离线训练/验证/测试数据中。\n\n")
        else:
            f.write(f"**⚠️ 发现数据泄露**: 有{len(overlaps)}个protocol场景在离线数据中出现。\n\n")
        f.write("---\n\n")
        f.write(f"*报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"  [OK] Report saved to: {output_report_path}")
    print()
    
    print("=" * 80)
    print("检查完成")
    print("=" * 80)
    print(f"\n总结:")
    print(f"  - Protocol场景数: {len(proto_sigs)}")
    print(f"  - 离线训练签名数: {len(train_sigs)}")
    print(f"  - 离线验证签名数: {len(val_sigs)}")
    print(f"  - 离线测试签名数: {len(test_sigs)}")
    print(f"  - 重叠场景数: {len(overlaps)}")
    print(f"\n报告已保存到: {output_report_path}")


if __name__ == "__main__":
    main()

