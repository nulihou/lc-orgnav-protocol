#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断关系语义消融实验：检查为什么Shuffled-Relation不掉

按照三个步骤诊断：
1. Step 1: 确认shuffle真的改变了模型输入
2. Step 2: 统计30个协议场景里关系是否有区分度
3. Step 3: 检查模型是否实际用到了关系特征
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import io
from collections import Counter

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/analysis
_analysis_dir = os.path.dirname(_script_dir)  # scripts
_project_root = os.path.dirname(_analysis_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lc_org_nav.online_nav.gs_loader import load_gs_model

NUM_REL_TYPES = 5
REL_ONEHOT_DIM = 5


def get_relation_onehot(lang_feat: torch.Tensor) -> np.ndarray:
    """提取关系onehot（lang_feat的前5维）"""
    return lang_feat[:REL_ONEHOT_DIM].cpu().numpy()


def get_relation_type(rel_onehot: np.ndarray) -> int:
    """从onehot获取关系类型"""
    return int(np.argmax(rel_onehot))


def shuffle_relation(lang_feat: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """打乱关系onehot"""
    np.random.seed(seed)
    lang_feat_perturbed = lang_feat.clone()
    rel_onehot = lang_feat_perturbed[:REL_ONEHOT_DIM].clone()
    shuffled_indices = np.random.permutation(REL_ONEHOT_DIM)
    rel_onehot_shuffled = rel_onehot[shuffled_indices]
    lang_feat_perturbed[:REL_ONEHOT_DIM] = rel_onehot_shuffled
    return lang_feat_perturbed


def step1_check_shuffle_effect(protocol_dir: str, scene_id: int = 0, seed: int = 42) -> Dict[str, Any]:
    """
    Step 1: 确认shuffle真的改变了模型输入
    抽1个scene，打印full和shuffle后的关系onehot
    """
    print("=" * 80)
    print("Step 1: Checking if shuffle actually changes model input")
    print("=" * 80)
    
    npz_path = os.path.join(protocol_dir, f"scen_{scene_id:03d}.npz")
    if not os.path.exists(npz_path):
        print(f"[ERROR] Protocol graph not found: {npz_path}")
        return {}
    
    data = np.load(npz_path, allow_pickle=False)
    lang_feat_full = torch.from_numpy(data["lang_feat"]).float()
    lang_feat_shuffled = shuffle_relation(lang_feat_full, seed=seed)
    
    rel_onehot_full = get_relation_onehot(lang_feat_full)
    rel_onehot_shuffled = get_relation_onehot(lang_feat_shuffled)
    rel_type_full = get_relation_type(rel_onehot_full)
    rel_type_shuffled = get_relation_type(rel_onehot_shuffled)
    
    print(f"\nScene {scene_id:03d}:")
    print(f"  Full relation onehot:     {rel_onehot_full}")
    print(f"  Shuffled relation onehot: {rel_onehot_shuffled}")
    print(f"  Full relation type:       {rel_type_full}")
    print(f"  Shuffled relation type:   {rel_type_shuffled}")
    print(f"  Changed:                   {not np.array_equal(rel_onehot_full, rel_onehot_shuffled)}")
    
    if np.array_equal(rel_onehot_full, rel_onehot_shuffled):
        print(f"\n  ⚠️  WARNING: Shuffle did not change the relation onehot!")
        print(f"     This means the shuffle operation is not working correctly.")
    else:
        print(f"\n  ✅ Shuffle successfully changed the relation onehot.")
    
    return {
        "scene_id": scene_id,
        "rel_onehot_full": rel_onehot_full.tolist(),
        "rel_onehot_shuffled": rel_onehot_shuffled.tolist(),
        "rel_type_full": rel_type_full,
        "rel_type_shuffled": rel_type_shuffled,
        "changed": not np.array_equal(rel_onehot_full, rel_onehot_shuffled),
    }


def step2_check_relation_diversity(protocol_dir: str, max_scen_id: int = 29) -> Dict[str, Any]:
    """
    Step 2: 统计30个协议场景里关系是否有区分度
    每个scene看候选车的关系标签集合大小
    """
    print("\n" + "=" * 80)
    print("Step 2: Checking relation diversity across scenes")
    print("=" * 80)
    
    scene_stats = []
    unique_rel_counts = []
    
    for scen_id in range(max_scen_id + 1):
        npz_path = os.path.join(protocol_dir, f"scen_{scen_id:03d}.npz")
        if not os.path.exists(npz_path):
            continue
        
        data = np.load(npz_path, allow_pickle=False)
        lang_feat = data["lang_feat"]
        rel_onehot = lang_feat[:REL_ONEHOT_DIM]
        rel_type = int(np.argmax(rel_onehot))
        
        # 注意：协议图中lang_feat是anchor→target的关系
        # 我们需要检查的是：如果场景中有多个候选，它们的关系是否不同
        # 但协议图只存储了anchor→target的关系，不是所有候选的关系
        
        # 这里我们只能检查当前场景的关系类型
        scene_stats.append({
            "scene_id": scen_id,
            "rel_type": rel_type,
            "rel_onehot": rel_onehot.tolist(),
        })
        unique_rel_counts.append(1)  # 每个场景只有一个关系（anchor→target）
    
    # 统计所有场景的关系类型分布
    rel_type_distribution = Counter([s["rel_type"] for s in scene_stats])
    
    print(f"\nRelation type distribution across {len(scene_stats)} scenes:")
    for rel_type in range(NUM_REL_TYPES):
        count = rel_type_distribution.get(rel_type, 0)
        print(f"  Rel type {rel_type}: {count} scenes ({count/len(scene_stats)*100:.1f}%)")
    
    print(f"\n⚠️  NOTE: Protocol graphs only store anchor→target relation.")
    print(f"   To check candidate diversity, we need to check if different candidates")
    print(f"   have different relations. This requires examining the original scenario data.")
    
    return {
        "total_scenes": len(scene_stats),
        "rel_type_distribution": dict(rel_type_distribution),
        "scene_stats": scene_stats,
    }


def step3_check_model_sensitivity(
    model,
    protocol_dir: str,
    device: torch.device,
    scene_id: int = 0,
) -> Dict[str, Any]:
    """
    Step 3: 检查模型是否实际用到了关系特征
    对同一scene，改变关系onehot，看输出score的变化幅度
    """
    print("\n" + "=" * 80)
    print("Step 3: Checking model sensitivity to relation features")
    print("=" * 80)
    
    npz_path = os.path.join(protocol_dir, f"scen_{scene_id:03d}.npz")
    if not os.path.exists(npz_path):
        print(f"[ERROR] Protocol graph not found: {npz_path}")
        return {}
    
    data = np.load(npz_path, allow_pickle=False)
    node_feats = torch.from_numpy(data["node_feats"]).float().to(device)
    edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
    edge_attr = torch.from_numpy(data["edge_attr"]).float().to(device)
    lang_feat_full = torch.from_numpy(data["lang_feat"]).float().to(device)
    cand_mask_np = data["candidate_mask"].astype(bool)
    candidate_mask = torch.from_numpy(cand_mask_np).bool().to(device)
    target_idx_gt = int(data["target_idx_gt"])
    
    model.eval()
    
    # 测试不同的关系扰动
    perturbations = {
        "full": lang_feat_full,
        "no_rel": lang_feat_full.clone(),
        "shuffle_rel": shuffle_relation(lang_feat_full, seed=42),
        "rel_type_0": lang_feat_full.clone(),
        "rel_type_1": lang_feat_full.clone(),
        "rel_type_2": lang_feat_full.clone(),
        "rel_type_3": lang_feat_full.clone(),
        "rel_type_4": lang_feat_full.clone(),
    }
    
    # 设置不同的关系类型
    perturbations["no_rel"][:REL_ONEHOT_DIM] = 0.0
    for rel_type in range(NUM_REL_TYPES):
        perturbations[f"rel_type_{rel_type}"][:REL_ONEHOT_DIM] = 0.0
        perturbations[f"rel_type_{rel_type}"][rel_type] = 1.0
    
    results = {}
    
    with torch.no_grad():
        for name, lang_feat_perturbed in perturbations.items():
            outputs = model.forward_single(
                node_feats=node_feats,
                edge_index=edge_index,
                edge_attr=edge_attr,
                lang_feat=lang_feat_perturbed,
                candidate_mask=candidate_mask,
            )
            goal_logits = outputs["goal_logits"]  # (N,)
            
            # 获取候选的logits
            logits_cands = goal_logits[candidate_mask]
            cand_indices = np.nonzero(cand_mask_np)[0]
            
            # 获取预测
            k_best = torch.argmax(logits_cands).item()
            chosen_idx = int(cand_indices[k_best])
            
            # 获取所有候选的logits和probs
            cand_logits = logits_cands.cpu().numpy()
            cand_probs = torch.softmax(logits_cands, dim=0).cpu().numpy()
            
            # 找到target在候选中的位置
            target_in_cands = np.where(cand_indices == target_idx_gt)[0]
            if len(target_in_cands) > 0:
                target_logit = cand_logits[target_in_cands[0]]
                target_prob = cand_probs[target_in_cands[0]]
            else:
                target_logit = None
                target_prob = None
            
            results[name] = {
                "chosen_idx": chosen_idx,
                "target_idx_gt": target_idx_gt,
                "correct": (chosen_idx == target_idx_gt),
                "max_logit": float(np.max(cand_logits)),
                "target_logit": float(target_logit) if target_logit is not None else None,
                "target_prob": float(target_prob) if target_prob is not None else None,
                "cand_logits": cand_logits.tolist(),
                "cand_probs": cand_probs.tolist(),
            }
    
    # 打印结果
    print(f"\nScene {scene_id:03d} - Model output sensitivity:")
    print(f"\n{'Perturbation':<20} {'Chosen':<10} {'Target':<10} {'Correct':<10} {'Max Logit':<12} {'Target Logit':<15} {'Target Prob':<15}")
    print("-" * 100)
    
    baseline_logit = results["full"]["max_logit"]
    baseline_target_logit = results["full"]["target_logit"]
    baseline_target_prob = results["full"]["target_prob"]
    
    for name, res in results.items():
        delta_logit = res["max_logit"] - baseline_logit if baseline_logit is not None else 0.0
        delta_target_logit = (res["target_logit"] - baseline_target_logit) if (res["target_logit"] is not None and baseline_target_logit is not None) else 0.0
        delta_target_prob = (res["target_prob"] - baseline_target_prob) if (res["target_prob"] is not None and baseline_target_prob is not None) else 0.0
        
        print(f"{name:<20} {res['chosen_idx']:<10} {res['target_idx_gt']:<10} {'✓' if res['correct'] else '✗':<10} "
              f"{res['max_logit']:<12.4f} {res['target_logit']:<15.4f} {res['target_prob']:<15.4f}")
        if name != "full":
            print(f"{'':20} {'':10} {'':10} {'':10} {'':12} "
                  f"Δ={delta_target_logit:+.4f}      Δ={delta_target_prob:+.4f}")
    
    # 计算敏感度
    shuffle_delta_logit = results["shuffle_rel"]["max_logit"] - baseline_logit
    shuffle_delta_target_logit = (results["shuffle_rel"]["target_logit"] - baseline_target_logit) if (results["shuffle_rel"]["target_logit"] is not None and baseline_target_logit is not None) else 0.0
    shuffle_delta_target_prob = (results["shuffle_rel"]["target_prob"] - baseline_target_prob) if (results["shuffle_rel"]["target_prob"] is not None and baseline_target_prob is not None) else 0.0
    
    no_rel_delta_logit = results["no_rel"]["max_logit"] - baseline_logit
    no_rel_delta_target_logit = (results["no_rel"]["target_logit"] - baseline_target_logit) if (results["no_rel"]["target_logit"] is not None and baseline_target_logit is not None) else 0.0
    no_rel_delta_target_prob = (results["no_rel"]["target_prob"] - baseline_target_prob) if (results["no_rel"]["target_prob"] is not None and baseline_target_prob is not None) else 0.0
    
    print(f"\nSensitivity Analysis:")
    print(f"  Shuffled-Relation vs Full:")
    print(f"    Δ max_logit:        {shuffle_delta_logit:+.4f}")
    print(f"    Δ target_logit:     {shuffle_delta_target_logit:+.4f}")
    print(f"    Δ target_prob:      {shuffle_delta_target_prob:+.4f}")
    print(f"  No-Relation vs Full:")
    print(f"    Δ max_logit:        {no_rel_delta_logit:+.4f}")
    print(f"    Δ target_logit:     {no_rel_delta_target_logit:+.4f}")
    print(f"    Δ target_prob:      {no_rel_delta_target_prob:+.4f}")
    
    # 判断敏感度
    if abs(shuffle_delta_target_logit) < 0.01 and abs(shuffle_delta_target_prob) < 0.01:
        print(f"\n  ⚠️  WARNING: Model is NOT sensitive to relation shuffling!")
        print(f"     The output barely changes when relations are shuffled.")
        print(f"     This suggests the model may be ignoring relation features.")
    else:
        print(f"\n  ✅ Model IS sensitive to relation changes.")
        print(f"     The output changes when relations are modified.")
    
    return {
        "scene_id": scene_id,
        "results": results,
        "sensitivity": {
            "shuffle_delta_logit": float(shuffle_delta_logit),
            "shuffle_delta_target_logit": float(shuffle_delta_target_logit),
            "shuffle_delta_target_prob": float(shuffle_delta_target_prob),
            "no_rel_delta_logit": float(no_rel_delta_logit),
            "no_rel_delta_target_logit": float(no_rel_delta_target_logit),
            "no_rel_delta_target_prob": float(no_rel_delta_target_prob),
        },
    }


def check_candidate_relation_diversity(scenarios_path: str, protocol_dir: str) -> Dict[str, Any]:
    """
    检查候选之间的关系多样性
    需要从原始场景数据中检查每个候选的关系
    """
    print("\n" + "=" * 80)
    print("Step 2 (Extended): Checking candidate relation diversity")
    print("=" * 80)
    
    # 加载场景数据
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    
    scene_diversity_stats = []
    
    for scen in scenarios:
        scen_id = scen.get("id", -1)
        if scen_id < 0:
            continue
        
        # 加载协议图获取候选信息
        npz_path = os.path.join(protocol_dir, f"scen_{scen_id:03d}.npz")
        if not os.path.exists(npz_path):
            continue
        
        data = np.load(npz_path, allow_pickle=False)
        candidate_indices = data["candidate_indices"].tolist() if "candidate_indices" in data else []
        ego_index = int(data["ego_index"]) if "ego_index" in data else 0
        
        # 注意：协议图中只存储了anchor→target的关系
        # 要检查所有候选的关系，需要从原始场景数据中获取
        # 或者需要重新计算每个候选的关系
        
        # 这里我们只能检查当前场景的关系类型
        lang_feat = data["lang_feat"]
        rel_onehot = lang_feat[:REL_ONEHOT_DIM]
        rel_type = int(np.argmax(rel_onehot))
        
        scene_diversity_stats.append({
            "scene_id": scen_id,
            "num_candidates": len(candidate_indices),
            "rel_type": rel_type,
            "unique_rel_count": 1,  # 协议图只存储一个关系
        })
    
    # 统计
    unique_rel_counts = [s["unique_rel_count"] for s in scene_diversity_stats]
    avg_unique_rel = np.mean(unique_rel_counts) if unique_rel_counts else 0.0
    
    print(f"\nCandidate relation diversity statistics:")
    print(f"  Total scenes: {len(scene_diversity_stats)}")
    print(f"  Average unique relations per scene: {avg_unique_rel:.2f}")
    print(f"  Scenes with unique_rel_count=1: {sum(1 for c in unique_rel_counts if c == 1)}")
    print(f"  Scenes with unique_rel_count>1: {sum(1 for c in unique_rel_counts if c > 1)}")
    
    print(f"\n⚠️  NOTE: Protocol graphs only store anchor→target relation.")
    print(f"   To fully check candidate diversity, we need to compute relations")
    print(f"   for all candidates in each scene, not just the target.")
    
    return {
        "scene_diversity_stats": scene_diversity_stats,
        "avg_unique_rel": float(avg_unique_rel),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose relation ablation: Why Shuffled-Relation doesn't drop"
    )
    parser.add_argument(
        "--protocol-dir",
        type=str,
        default="scripts/online_multi/protocol_graphs",
        help="Directory containing protocol graphs"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="data/online_multi/multi_seen_v3.json",
        help="Path to scenarios JSON file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs_v2/multitask_mixed_v2_online_multi_rank/seed_42/best_lc_org_mtl_mixed.pt",
        help="Path to GS model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/proto_main_eval/ablation_relation_diagnosis.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--test-scene-id",
        type=int,
        default=0,
        help="Scene ID to test in Step 1 and Step 3"
    )
    parser.add_argument(
        "--max-scen-id",
        type=int,
        default=29,
        help="Maximum scene ID to check"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Relation Ablation Diagnosis")
    print("=" * 80)
    print(f"[Config] Protocol dir: {args.protocol_dir}")
    print(f"[Config] Checkpoint: {args.checkpoint}")
    print(f"[Config] Device: {args.device}")
    print(f"[Config] Test scene ID: {args.test_scene_id}")
    
    # Step 1: Check if shuffle actually changes input
    step1_result = step1_check_shuffle_effect(
        args.protocol_dir, 
        scene_id=args.test_scene_id,
        seed=42
    )
    
    # Step 2: Check relation diversity
    step2_result = step2_check_relation_diversity(
        args.protocol_dir,
        max_scen_id=args.max_scen_id
    )
    
    # Step 2 Extended: Check candidate relation diversity
    step2_ext_result = check_candidate_relation_diversity(
        args.scenarios,
        args.protocol_dir
    )
    
    # Step 3: Check model sensitivity
    device = torch.device(args.device)
    print(f"\n[Model] Loading GS model...")
    model = load_gs_model(args.checkpoint, device)
    
    step3_result = step3_check_model_sensitivity(
        model,
        args.protocol_dir,
        device,
        scene_id=args.test_scene_id
    )
    
    # Save results
    results = {
        "step1_shuffle_check": step1_result,
        "step2_relation_diversity": step2_result,
        "step2_ext_candidate_diversity": step2_ext_result,
        "step3_model_sensitivity": step3_result,
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Done] Diagnosis results saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Diagnosis Summary")
    print("=" * 80)
    
    if step1_result.get("changed", False):
        print("✅ Step 1: Shuffle operation is working correctly")
    else:
        print("❌ Step 1: Shuffle operation is NOT working - this is the problem!")
    
    print(f"✅ Step 2: Relation diversity checked ({step2_result.get('total_scenes', 0)} scenes)")
    
    if step3_result:
        sensitivity = step3_result.get("sensitivity", {})
        shuffle_delta = abs(sensitivity.get("shuffle_delta_target_logit", 0.0))
        if shuffle_delta < 0.01:
            print("❌ Step 3: Model is NOT sensitive to relation changes - model may be ignoring relations")
        else:
            print("✅ Step 3: Model IS sensitive to relation changes")


if __name__ == "__main__":
    main()

