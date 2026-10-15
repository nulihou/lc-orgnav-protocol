"""
Scene graph builder for online navigation evaluation.

Reuses data collection logic to build scene graphs from CARLA world state.
"""

import math
import random
import numpy as np
import torch
import carla
from typing import Dict, Any, List, Optional

from lc_org_nav.geometry import ego_to_target_geometry

# Constants (must match data collection)
NODE_FEAT_DIM = 32
EDGE_FEAT_DIM = 8
LANG_FEAT_DIM = 32
NUM_REL_TYPES = 5
MAX_CAND_DIST = 50.0


def classify_relation(anchor_tf, target_tf, rel_type: int) -> bool:
    """Check if target satisfies the given relation type relative to anchor."""
    ax, ay = anchor_tf.location.x, anchor_tf.location.y
    ayaw = math.radians(anchor_tf.rotation.yaw)
    tx, ty = target_tf.location.x, target_tf.location.y
    dx = tx - ax
    dy = ty - ay
    d = math.sqrt(dx * dx + dy * dy)
    global_angle = math.atan2(dy, dx)
    theta = global_angle - ayaw
    while theta <= -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi
    
    # Thresholds (must match data collection)
    near_d = 20.0
    far_min_d = 5.0
    far_max_d = 40.0
    front_deg = math.radians(30.0)
    side_min = math.radians(60.0)
    side_max = math.radians(120.0)
    
    if rel_type == 0:  # near
        return d < near_d
    elif rel_type == 1:  # front
        return (far_min_d < d < far_max_d) and abs(theta) < front_deg
    elif rel_type == 2:  # behind
        return (far_min_d < d < far_max_d) and abs(abs(theta) - math.pi) < front_deg
    elif rel_type == 3:  # left
        return (far_min_d < d < far_max_d) and (side_min < theta < side_max)
    elif rel_type == 4:  # right
        return (far_min_d < d < far_max_d) and (-side_max < theta < -side_min)
    else:
        return False


def sample_anchor_target_with_relation(vehicles, max_trials: int = 50):
    """Sample anchor/target/rel_type with unique relation constraint."""
    if len(vehicles) < 3:
        raise RuntimeError("Not enough NPC vehicles to form a relational scene.")
    
    for _ in range(max_trials):
        anchor = random.choice(vehicles)
        anchor_tf = anchor.get_transform()
        rel_type = random.randrange(NUM_REL_TYPES)
        candidates = []
        for v in vehicles:
            if v.id == anchor.id:
                continue
            if classify_relation(anchor_tf, v.get_transform(), rel_type):
                candidates.append(v)
        if len(candidates) == 1:
            target = candidates[0]
            assert classify_relation(anchor_tf, target.get_transform(), rel_type), \
                "Target does not satisfy relation at collection time!"
            return anchor, target, rel_type
    
    raise RuntimeError("Failed to sample anchor/target with desired relation.")


def build_node_features(ego_transform, vehicles, anchor_id, target_id):
    """Build node features (same as data collection)."""
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    ego_yaw = math.radians(ego_transform.rotation.yaw)
    
    # Find anchor transform
    anchor_tf = None
    for v in vehicles:
        if v.id == anchor_id:
            anchor_tf = v.get_transform()
            break
    if anchor_tf is None:
        raise RuntimeError("Anchor transform not found")
    ax, ay = anchor_tf.location.x, anchor_tf.location.y
    ayaw = math.radians(anchor_tf.rotation.yaw)
    
    node_feats_list = []
    candidate_mask = []
    
    for veh in vehicles:
        loc = veh.get_transform().location
        vel = veh.get_velocity()
        
        # Ego-relative
        dx_ego = loc.x - ego_x
        dy_ego = loc.y - ego_y
        d_ego = math.sqrt(dx_ego * dx_ego + dy_ego * dy_ego)
        global_angle_ego = math.atan2(dy_ego, dx_ego)
        theta_ego = global_angle_ego - ego_yaw
        while theta_ego <= -math.pi:
            theta_ego += 2 * math.pi
        while theta_ego > math.pi:
            theta_ego -= 2 * math.pi
        
        # Anchor-relative
        dx_a = loc.x - ax
        dy_a = loc.y - ay
        d_a = math.sqrt(dx_a * dx_a + dy_a * dy_a)
        global_angle_a = math.atan2(dy_a, dx_a)
        theta_a = global_angle_a - ayaw
        while theta_a <= -math.pi:
            theta_a += 2 * math.pi
        while theta_a > math.pi:
            theta_a -= 2 * math.pi
        
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
        is_anchor = 1.0 if veh.id == anchor_id else 0.0
        is_target = 0.0
        
        base_feat = np.array([
            dx_ego, dy_ego, d_ego,
            math.cos(theta_ego), math.sin(theta_ego),
            dx_a, dy_a, d_a,
            math.cos(theta_a), math.sin(theta_a),
            speed,
            is_anchor,
            is_target,
        ], dtype=np.float32)
        
        if base_feat.shape[0] < NODE_FEAT_DIM:
            pad = np.zeros((NODE_FEAT_DIM - base_feat.shape[0],), dtype=np.float32)
            feat_vec = np.concatenate([base_feat, pad], axis=0)
        else:
            feat_vec = base_feat[:NODE_FEAT_DIM]
        
        node_feats_list.append(feat_vec)
        is_candidate = (d_ego < MAX_CAND_DIST) and (veh.id != anchor_id)
        if veh.id == target_id:
            is_candidate = True
        candidate_mask.append(is_candidate)
    
    node_feats = np.stack(node_feats_list, axis=0)
    candidate_mask = np.array(candidate_mask, dtype=bool)
    return node_feats, candidate_mask


def build_edges(vehicles):
    """Build edges (same as data collection)."""
    N = len(vehicles)
    edge_src = []
    edge_dst = []
    edge_attr_list = []
    
    for i in range(N):
        loc_i = vehicles[i].get_transform().location
        for j in range(N):
            if i == j:
                continue
            loc_j = vehicles[j].get_transform().location
            
            dx = loc_j.x - loc_i.x
            dy = loc_j.y - loc_i.y
            d = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)
            
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            base_edge = np.array([
                dx, dy, d,
                cos_a, sin_a,
                1.0 if d < 30.0 else 0.0,
                0.0, 0.0,
            ], dtype=np.float32)
            
            if base_edge.shape[0] < EDGE_FEAT_DIM:
                pad = np.zeros((EDGE_FEAT_DIM - base_edge.shape[0],), dtype=np.float32)
                feat_edge = np.concatenate([base_edge, pad], axis=0)
            else:
                feat_edge = base_edge[:EDGE_FEAT_DIM]
            
            edge_src.append(i)
            edge_dst.append(j)
            edge_attr_list.append(feat_edge)
    
    edge_index = np.stack([np.array(edge_src, dtype=np.int64), np.array(edge_dst, dtype=np.int64)], axis=0)
    edge_attr = np.stack(edge_attr_list, axis=0).astype(np.float32)
    return edge_index, edge_attr


def build_lang_feat_from_rel_type(rel_type: int):
    """Encode relation type as one-hot."""
    lang_feat = np.zeros((LANG_FEAT_DIM,), dtype=np.float32)
    if 0 <= rel_type < NUM_REL_TYPES:
        lang_feat[rel_type] = 1.0
    return lang_feat.astype(np.float32)


def build_geom_label(ego_transform, target_transform):
    """Compute [distance, theta] from ego to target."""
    d, theta = ego_to_target_geometry(
        ego_transform.location.x,
        ego_transform.location.y,
        math.radians(ego_transform.rotation.yaw),
        target_transform.location.x,
        target_transform.location.y,
    )
    return np.array([d, theta], dtype=np.float32)


def build_graph_from_world(world, ego_vehicle) -> Optional[Dict[str, Any]]:
    """
    Build a scene graph from current CARLA world state.
    
    This function replicates the data collection logic but returns
    torch.Tensor format suitable for model inference.
    
    Args:
        world: CARLA world
        ego_vehicle: ego vehicle actor
        
    Returns:
        Sample dict with torch.Tensor fields, or None if failed.
        Key fields:
            - vehicles: List of CARLA vehicle actors (for index mapping)
            - node_feats: (1, N, F) torch.Tensor
            - edge_index: (2, E) torch.Tensor
            - edge_attr: (1, E, F_e) torch.Tensor
            - lang_feat: (1, D_l) torch.Tensor
            - candidate_mask: (1, N) torch.Tensor
            - target_index_gt: int (ground truth target index)
            - rel_type: int
            - geom: (1, 2) torch.Tensor
    """
    actors = world.get_actors().filter("vehicle.*")
    vehicles = [v for v in actors if v.id != ego_vehicle.id]
    
    if len(vehicles) < 3:
        return None
    
    # Sample anchor/target/rel_type (same as data collection)
    try:
        anchor, target_gt, rel_type = sample_anchor_target_with_relation(vehicles)
    except RuntimeError:
        return None
    
    # Build graph features (same as data collection)
    ego_transform = ego_vehicle.get_transform()
    node_feats, candidate_mask = build_node_features(
        ego_transform, vehicles, anchor.id, target_gt.id
    )
    edge_index, edge_attr = build_edges(vehicles)
    lang_feat = build_lang_feat_from_rel_type(rel_type)
    geom = build_geom_label(ego_transform, target_gt.get_transform())
    
    # Find target index in vehicles list
    target_gt_idx = None
    for i, v in enumerate(vehicles):
        if v.id == target_gt.id:
            target_gt_idx = i
            break
    
    if target_gt_idx is None:
        return None
    
    # Convert to torch.Tensor format (add batch dimension)
    sample = {
        "vehicles": vehicles,  # Keep original list for index mapping
        "node_feats": torch.from_numpy(node_feats).unsqueeze(0).float(),  # (1, N, F)
        "edge_index": torch.from_numpy(edge_index).long(),  # (2, E)
        "edge_attr": torch.from_numpy(edge_attr).unsqueeze(0).float(),  # (1, E, F_e)
        "lang_feat": torch.from_numpy(lang_feat).unsqueeze(0).float(),  # (1, D_l)
        "candidate_mask": torch.from_numpy(candidate_mask).unsqueeze(0).bool(),  # (1, N)
        "target_index_gt": int(target_gt_idx),  # Ground truth for evaluation
        "rel_type": int(rel_type),
        "geom": torch.from_numpy(geom).unsqueeze(0).float(),  # (1, 2)
    }
    
    return sample

