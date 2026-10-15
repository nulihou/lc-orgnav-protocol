#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预计算协议图脚本

读取 multi_seen_v3.json，对每个场景构建一张协议图，存成 .npz。
图完全由 JSON 协议决定，与 CARLA 运行时的 NPC spawn 状态无关。
"""

import os
import sys
import json
import argparse
import numpy as np
import carla
from pathlib import Path
from typing import List, Dict, Any, Optional
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/online_multi
_online_multi_dir = os.path.dirname(_script_dir)  # scripts
_project_root = os.path.dirname(_online_multi_dir)  # 项目根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.data.carla_collect_data import (
    build_node_features,
    build_edges,
    build_lang_feat_from_rel_type,
    classify_relation,
    NUM_REL_TYPES,
)


# ---- 小工具：FakeVehicle，用 spawn_point 来伪造一个"车辆" ----
class FakeVehicle:
    """用于离线构建协议图的假车辆对象"""
    def __init__(self, vid, transform):
        self.id = vid
        self._transform = transform

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        # 离线协议图不关心真实速度，这里统一设 0
        return carla.Vector3D(0.0, 0.0, 0.0)


def precompute_protocol_graphs(args):
    """
    预计算所有场景的协议图
    
    Args:
        args: 命令行参数
    """
    # 1) 读场景 JSON
    with open(args.scenario_json, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # 2) 连接 CARLA，用于拿 spawn_points
    print(f"\n[CARLA] Connecting to {args.host}:{args.port}...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
    except Exception as e:
        print(f"[ERROR] Failed to connect to CARLA: {e}")
        print(f"[ERROR] Please make sure CARLA is running on {args.host}:{args.port}")
        sys.exit(1)

    current_town = None
    world = None
    spawn_points_cache = {}

    results = []
    failed_scenarios = []

    print(f"\n[PRECOMPUTE] Processing {len(scenarios)} scenarios...")

    for scen in scenarios:
        scen_id = scen["id"]
        town = scen.get("town", "Town10HD_Opt")
        
        # 如果指定了max_episodes，只处理前N个场景
        if args.max_episodes is not None and scen_id >= args.max_episodes:
            continue

        print(f"\n[PRECOMPUTE] scen={scen_id} town={town}")

        try:
            # 2.1 切换地图
            if town != current_town:
                print(f"  - Loading world {town} ...")
                world = client.load_world(town)
                current_town = town
                spawn_points_cache[town] = world.get_map().get_spawn_points()
                print(f"  - Loaded {len(spawn_points_cache[town])} spawn points")

            spawn_points = spawn_points_cache[town]

            # 3) 协议信息
            node_spawn_indices = scen.get("node_spawn_indices")
            candidate_indices = scen.get("candidate_indices")
            target_index_gt = scen.get("target_index_gt")
            ego_spawn_idx = scen["ego_spawn_idx"]
            anchor_spawn_idx = scen.get("anchor_spawn_idx", ego_spawn_idx)

            # 如果没有协议字段，报错
            if node_spawn_indices is None:
                print(f"  [ERROR] scen {scen_id}: missing node_spawn_indices, please run add_protocol_fields_v3.py first")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": "missing node_spawn_indices"
                })
                continue

            if candidate_indices is None:
                print(f"  [ERROR] scen {scen_id}: missing candidate_indices, please run add_protocol_fields_v3.py first")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": "missing candidate_indices"
                })
                continue

            if target_index_gt is None:
                print(f"  [ERROR] scen {scen_id}: missing target_index_gt, please run add_protocol_fields_v3.py first")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": "missing target_index_gt"
                })
                continue

            # 找到 ego 在 node_spawn_indices 中的索引
            try:
                ego_index = node_spawn_indices.index(ego_spawn_idx)
            except ValueError:
                print(f"  [ERROR] scen {scen_id}: ego_spawn_idx={ego_spawn_idx} 不在 node_spawn_indices 里")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": f"ego_spawn_idx={ego_spawn_idx} not in node_spawn_indices"
                })
                continue

            # 4) 构造 FakeVehicle 列表（一个节点一个 FakeVehicle）
            vehicles = []
            vehicles_valid = True
            for node_idx, sp_idx in enumerate(node_spawn_indices):
                if sp_idx >= len(spawn_points):
                    print(f"  [ERROR] scen {scen_id}: spawn_idx={sp_idx} 越界 (len(spawn_points)={len(spawn_points)})")
                    failed_scenarios.append({
                        "id": scen_id,
                        "reason": f"spawn_idx={sp_idx} out of range"
                    })
                    vehicles_valid = False
                    break
                tf = spawn_points[sp_idx]
                veh = FakeVehicle(vid=1000 + node_idx, transform=tf)
                vehicles.append(veh)
            
            if not vehicles_valid:
                continue  # 如果break了，跳过这个场景

            try:
                anchor_index = node_spawn_indices.index(anchor_spawn_idx)
            except ValueError:
                print(f"  [ERROR] scen {scen_id}: anchor_spawn_idx={anchor_spawn_idx} not in node_spawn_indices")
                failed_scenarios.append({
                    "id": scen_id,
                    "reason": f"anchor_spawn_idx={anchor_spawn_idx} not in node_spawn_indices"
                })
                continue

            ego_tf = vehicles[ego_index].get_transform()
            anchor_id = vehicles[anchor_index].id
            target_id = vehicles[target_index_gt].id

            # 5) 用离线的 build_node_features + build_edges 构图
            node_feats_full, cand_mask_offline = build_node_features(
                ego_tf,
                vehicles,
                anchor_id,
                target_id,
            )
            edge_index, edge_attr = build_edges(vehicles)

            N = node_feats_full.shape[0]

            # 6) 按协议构造 candidate_mask
            candidate_mask = np.zeros(N, dtype=bool)
            for ci in candidate_indices:
                if ci < 0 or ci >= N:
                    print(f"  [WARN] scen {scen_id}: candidate index {ci} 越界 (N={N})，跳过")
                    continue
                candidate_mask[ci] = True

            # 确保 target 在候选集合中
            if not candidate_mask[target_index_gt]:
                print(f"  [WARN] scen {scen_id}: GT 不在 candidate_indices 中，强制加入")
                candidate_mask[target_index_gt] = True

            if candidate_mask.sum() < 2:
                print(f"  [WARN] scen {scen_id}: 候选数 < 2 (={candidate_mask.sum()})，该场景对 GS 区分度很弱")

            # 7) 计算 rel_type 和 lang_feat（anchor vs target）
            anchor_tf = ego_tf
            anchor_tf = vehicles[anchor_index].get_transform()
            target_tf = vehicles[target_index_gt].get_transform()
            if args.honor_scenario_rel_type and "rel_type" in scen:
                rel_type = int(scen["rel_type"])
                if rel_type < 0 or rel_type >= NUM_REL_TYPES:
                    print(f"  [WARN] scen {scen_id}: invalid scenario rel_type={rel_type}; falling back to inferred rel_type")
                    rel_type = 0
                    for rt in range(NUM_REL_TYPES):
                        if classify_relation(anchor_tf, target_tf, rt):
                            rel_type = rt
                            break
            else:
                rel_type = 0
                for rt in range(NUM_REL_TYPES):
                    if classify_relation(anchor_tf, target_tf, rt):
                        rel_type = rt
                        break
            if args.honor_scenario_lang_feat and "lang_feat" in scen:
                lang_feat = np.asarray(scen["lang_feat"], dtype=np.float32)
                if lang_feat.shape != (32,):
                    print(f"  [WARN] scen {scen_id}: invalid scenario lang_feat shape={lang_feat.shape}; falling back to rel_type one-hot")
                    lang_feat = build_lang_feat_from_rel_type(rel_type)
            else:
                lang_feat = build_lang_feat_from_rel_type(rel_type)

            extra_arrays = {}
            if "query_policy_id" in scen:
                extra_arrays["query_policy_id"] = np.int64(int(scen["query_policy_id"]))
            if "anchor_node_idx" in scen:
                extra_arrays["anchor_node_idx"] = np.int64(int(scen["anchor_node_idx"]))

            # 8) 保存 npz
            out_path = os.path.join(args.out_dir, f"scen_{scen_id:03d}.npz")
            np.savez(
                out_path,
                node_feats=node_feats_full.astype(np.float32),
                edge_index=edge_index.astype(np.int64),
                edge_attr=edge_attr.astype(np.float32),
                candidate_mask=candidate_mask,
                target_idx_gt=np.int64(target_index_gt),
                rel_type=np.int64(rel_type),
                lang_feat=lang_feat.astype(np.float32),
                node_spawn_indices=np.array(node_spawn_indices, dtype=np.int64),
                candidate_indices=np.array(candidate_indices, dtype=np.int64),
                ego_index=np.int64(ego_index),
                anchor_index=np.int64(anchor_index),
                **extra_arrays,
            )
            print(f"  [OK] saved protocol graph to {out_path}")
            print(f"    node_feats: {node_feats_full.shape}")
            print(f"    edge_index: {edge_index.shape}")
            print(f"    candidate_mask: {candidate_mask.sum()} candidates")
            print(f"    target_idx_gt: {target_index_gt}")

            results.append({
                "scen_id": scen_id,
                "status": "success",
                "num_nodes": N,
                "num_candidates": int(candidate_mask.sum()),
            })

        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed_scenarios.append({
                "id": scen_id,
                "reason": str(e)
            })
            continue

    # 总结
    print(f"\n[Summary]")
    print(f"  Success: {len(results)}/{len(scenarios)}")
    print(f"  Failed: {len(failed_scenarios)}/{len(scenarios)}")

    if failed_scenarios:
        print(f"\n[Failed Scenarios]")
        for fs in failed_scenarios:
            print(f"  scen_id={fs['id']}: {fs['reason']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute protocol graphs")
    parser.add_argument(
        "--scenario_json",
        type=str,
        default="data/online_multi/multi_seen_v3.json",
        help="Path to scenarios JSON file"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="scripts/online_multi/protocol_graphs",
        help="Output directory for protocol graphs"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="CARLA server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (for testing, e.g., 1 for scen_id=0 only)"
    )
    parser.add_argument(
        "--honor-scenario-rel-type",
        action="store_true",
        help="Use rel_type from the scenario JSON instead of inferring it from geometry"
    )
    parser.add_argument(
        "--honor-scenario-lang-feat",
        action="store_true",
        help="Use a 32-dim lang_feat vector from the scenario JSON when present"
    )

    args = parser.parse_args()

    precompute_protocol_graphs(args)
