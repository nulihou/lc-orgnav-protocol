
"""Collect relation-aware goal selection data from CARLA.

IMPORTANT:
    - This script is designed to be run in a local environment where CARLA is installed
      and a simulator is running on localhost:2000.
    - It will NOT run in this current environment (no CARLA), but the code is structured
      to be directly usable when you have CARLA.

It creates `.npz` samples compatible with `GoalSelectionDataset`, with fields:

    node_feats:    (N, 32)
    edge_index:    (2, E)
    edge_attr:     (E, 8)
    lang_feat:     (32,)
    candidate_mask:(N,)
    target_index:  ()
    rel_type:      ()
    geom:          (2,)  # [distance, theta]

Usage example (from project root):

    python scripts/carla_collect_data.py --num-scenes 200 --output-root data/carla_samples
"""
import os
import math
import random
import argparse
from pathlib import Path

import numpy as np

import carla  # Requires CARLA Python API in your environment

from lc_org_nav.geometry import ego_to_target_geometry

# Make sure these match lc_org_nav/config.py and generate_dummy_data.py.
NODE_FEAT_DIM = 32
EDGE_FEAT_DIM = 8
LANG_FEAT_DIM = 32
NUM_REL_TYPES = 5  # e.g. near, front, behind, left, right

def build_node_features(ego_transform, vehicles, anchor_id, target_id):
    """Build node_feats (N, NODE_FEAT_DIM) and candidate_mask (N,).
    Include both ego- and anchor-relative features. Filter candidates by range and exclude anchor.
    """
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
    # Candidate distance threshold (ego-relative)
    MAX_CAND_DIST = 50.0

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
        # 不泄露 target
        is_target = 0.0

        # Compose features: ego-relative + anchor-relative + speed/flags
        base_feat = np.array(
            [
                # ego-relative
                dx_ego, dy_ego, d_ego,
                math.cos(theta_ego), math.sin(theta_ego),
                # anchor-relative
                dx_a, dy_a, d_a,
                math.cos(theta_a), math.sin(theta_a),
                # others
                speed,
                is_anchor,
                is_target,
            ],
            dtype=np.float32,
        )

        # Pad to NODE_FEAT_DIM
        if base_feat.shape[0] < NODE_FEAT_DIM:
            pad = np.zeros((NODE_FEAT_DIM - base_feat.shape[0],), dtype=np.float32)
            feat_vec = np.concatenate([base_feat, pad], axis=0)
        else:
            feat_vec = base_feat[:NODE_FEAT_DIM]

        node_feats_list.append(feat_vec)
        # Filter candidates: within distance and not the anchor, BUT always keep target
        is_candidate = (d_ego < MAX_CAND_DIST) and (veh.id != anchor_id)
        if veh.id == target_id:
            is_candidate = True
        candidate_mask.append(is_candidate)

    node_feats = np.stack(node_feats_list, axis=0)  # (N, NODE_FEAT_DIM)
    candidate_mask = np.array(candidate_mask, dtype=bool)  # (N,)

    return node_feats, candidate_mask

def build_edges(vehicles):
    """Build a simple fully connected directed graph with geometric edge features.

    Returns:
        edge_index: (2, E) int64
        edge_attr:  (E, EDGE_FEAT_DIM) float32
    """
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

            # Simple directional encoding
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            base_edge = np.array(
                [
                    dx, dy, d,
                    cos_a, sin_a,
                    1.0 if d < 30.0 else 0.0,  # near flag
                    0.0,  # reserved
                    0.0,  # reserved
                ],
                dtype=np.float32,
            )

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
    """Encode relation type as one-hot in a LANG_FEAT_DIM vector."""
    lang_feat = np.zeros((LANG_FEAT_DIM,), dtype=np.float32)
    if 0 <= rel_type < NUM_REL_TYPES:
        lang_feat[rel_type] = 1.0
    return lang_feat.astype(np.float32)

def build_geom_label(ego_transform, target_transform):
    """Compute [distance, theta] from ego to target using helper."""
    d, theta = ego_to_target_geometry(
        ego_transform.location.x,
        ego_transform.location.y,
        math.radians(ego_transform.rotation.yaw),
        target_transform.location.x,
        target_transform.location.y,
    )
    return np.array([d, theta], dtype=np.float32)

def classify_relation(anchor_tf, target_tf, rel_type: int) -> bool:
    """
    Check if target satisfies the given relation type relative to anchor.
    rel_type semantics:
      0: near   (d < 50m)  # ✅ 统一为离线训练协议：90%的near在≤46.7m内
      1: front  (d < 60m and |theta| < 45°)
      2: behind (d < 60m and around 180° ±45°)
      3: left   (d < 60m and 45°~135°)
      4: right  (d < 60m and -135°~-45°)
    """
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
    # ✅ 统一near阈值：从20m调整为50m，与离线训练协议一致
    # 离线数据分析：90%的near在≤46.7m内，95%在≤48.6m内，99%在≤49.7m内
    near_d = 50.0
    far_min_d = 5.0
    far_max_d = 40.0
    front_deg = math.radians(30.0)
    side_min = math.radians(60.0)
    side_max = math.radians(120.0)
    if rel_type == 0:  # near
        return d < near_d
    elif rel_type == 1:  # front
        return (far_min_d < d < far_max_d) and abs(theta) < front_deg
    elif rel_type == 2:  # behind (symmetric around pi/-pi)
        return (far_min_d < d < far_max_d) and abs(abs(theta) - math.pi) < front_deg
    elif rel_type == 3:  # left
        return (far_min_d < d < far_max_d) and (side_min < theta < side_max)
    elif rel_type == 4:  # right
        return (far_min_d < d < far_max_d) and (-side_max < theta < -side_min)
    else:
        return False


def sample_anchor_target_with_relation(vehicles, max_trials: int = 50):
    """
    Randomly sample (anchor, rel_type), then pick a target satisfying that relation.
    Retry several times; raise if failed for this scene.
    """
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
        # Only accept scenes with a unique satisfying target
        if len(candidates) == 1:
            target = candidates[0]
            # 采集即时自检：目标必须满足关系
            assert classify_relation(anchor_tf, target.get_transform(), rel_type), "Target does not satisfy relation at collection time!"
            return anchor, target, rel_type
    raise RuntimeError("Failed to sample anchor/target with desired relation.")


def collect_scene_sample(world, ego_vehicle):
    """Collect a single scene and return a dict ready to be saved as .npz.

    Steps:
        - Retrieve all vehicle actors
        - Select anchor and target among NPC vehicles
        - Build node_feats, edge_index, edge_attr, lang_feat, candidate_mask, target_index, rel_type, geom
    """
    actors = world.get_actors().filter("vehicle.*")
    vehicles = [v for v in actors if v.id != ego_vehicle.id]
    if len(vehicles) < 3:
        raise RuntimeError("Not enough NPC vehicles to form a relational scene.")

    # 基于关系选择 anchor/target/rel_type，确保语义一致
    anchor = None
    target = None
    rel_type = None
    anchor, target, rel_type = sample_anchor_target_with_relation(vehicles)

    # Build graph nodes in the same order as `vehicles`
    ego_transform = ego_vehicle.get_transform()
    node_feats, candidate_mask = build_node_features(ego_transform, vehicles, anchor.id, target.id)
    edge_index, edge_attr = build_edges(vehicles)
    lang_feat = build_lang_feat_from_rel_type(rel_type)
    geom = build_geom_label(ego_transform, target.get_transform())

    # Determine indices in vehicles list
    target_index = None
    anchor_index = None
    for i, v in enumerate(vehicles):
        if v.id == target.id:
            target_index = i
        if v.id == anchor.id:
            anchor_index = i
    if target_index is None:
        raise RuntimeError("Failed to find target vehicle index.")
    if anchor_index is None:
        raise RuntimeError("Failed to find anchor vehicle index.")

    sample = {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "lang_feat": lang_feat,
        "candidate_mask": candidate_mask,
        "target_index": np.array(target_index, dtype=np.int64),
        "anchor_index": np.array(anchor_index, dtype=np.int64),
        "rel_type": np.array(rel_type, dtype=np.int64),
        "geom": geom,
    }
    return sample

def save_sample(sample, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        node_feats=sample["node_feats"],
        edge_index=sample["edge_index"],
        edge_attr=sample["edge_attr"],
        lang_feat=sample["lang_feat"],
        candidate_mask=sample["candidate_mask"],
        target_index=sample["target_index"],
        rel_type=sample["rel_type"],
        geom=sample["geom"],
    )

def setup_ego_and_npcs(world, num_npcs: int = 40):
    """Spawn an ego vehicle and a number of NPC vehicles with autopilot on."""
    blueprint_library = world.get_blueprint_library()

    ego_bp = blueprint_library.filter("vehicle.tesla.model3")
    ego_bp = ego_bp[0] if ego_bp else blueprint_library.filter("vehicle.*")[0]

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    if not spawn_points:
        raise RuntimeError("No spawn points found in the map.")

    ego_transform = spawn_points[0]
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    # Traffic manager setup
    try:
        client = world.get_client()
    except AttributeError:
        # CARLA API fallback: obtain client via map's underlying API if available
        # If not available, default to standard port
        import carla as _carla
        client = _carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)

    ego_vehicle.set_autopilot(True, tm.get_port())

    # Spawn NPCs
    npc_vehicles = []
    for sp in spawn_points[1: 1 + num_npcs]:
        veh_bp = random.choice(blueprint_library.filter("vehicle.*"))
        npc = world.try_spawn_actor(veh_bp, sp)
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npc_vehicles.append(npc)
    return ego_vehicle, npc_vehicles, tm

def main():
    parser = argparse.ArgumentParser(description="Collect CARLA relational navigation data for LC-ORGNav")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--num-scenes", type=int, default=100, help="Number of scenes to collect")
    parser.add_argument("--output-root", type=str, default="data/carla_samples", help="Output root directory")
    parser.add_argument("--town", type=str, default="", help="Optional: load specific town (e.g., Town10HD_Opt, Town03)")
    parser.add_argument("--weather", type=str, default="", help="Optional: set weather preset (e.g., ClearNoon, WetCloudyNoon, HardRainNoon)")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    # Optionally load town
    if args.town:
        try:
            world = client.load_world(args.town)
        except RuntimeError:
            world = client.get_world()
    else:
        world = client.get_world()

    # Optionally set weather
    if args.weather:
        try:
            preset = getattr(carla.WeatherParameters, args.weather)
            world.set_weather(preset)
        except AttributeError:
            pass

    # Try to enable synchronous mode for more deterministic data
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

    ego_vehicle = None
    npc_vehicles = []
    tm = None

    try:
        ego_vehicle, npc_vehicles, tm = setup_ego_and_npcs(world, num_npcs=40)

        root = Path(args.output_root).resolve()
        samples_dir = root / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Let the world run a few ticks before collecting
        for _ in range(20):
            world.tick()

        collected = 0
        attempts = 0
        max_attempts = args.num_scenes * 5

        while collected < args.num_scenes and attempts < max_attempts:
            attempts += 1
            world.tick()

            try:
                sample = collect_scene_sample(world, ego_vehicle)
            except RuntimeError as e:
                # Skip scenes that don't have enough vehicles etc.
                print(f"[WARN] Skipping scene (reason: {e})")
                continue

            out_path = samples_dir / f"scene_{collected:05d}.npz"
            save_sample(sample, out_path)
            print(f"[INFO] Saved scene {collected} to {out_path}")
            collected += 1

        # Write list files (train/val split)
        if collected > 0:
            rel_root = root.relative_to(Path(".").resolve())
            all_paths = [rel_root / "samples" / f"scene_{i:05d}.npz" for i in range(collected)]
            # simple split: 80% train, 20% val
            split_idx = int(0.8 * collected)
            train_paths = all_paths[:split_idx]
            val_paths = all_paths[split_idx:]

            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            train_list = data_dir / "train_list.txt"
            val_list = data_dir / "val_list.txt"
            with open(train_list, "w", encoding="utf-8") as f:
                for p in train_paths:
                    f.write(str(p) + "\n")
            with open(val_list, "w", encoding="utf-8") as f:
                for p in val_paths:
                    f.write(str(p) + "\n")

            print(f"[INFO] Wrote train list: {train_list}")
            print(f"[INFO] Wrote val   list: {val_list}")

    finally:
        # Clean up actors
        actors_to_destroy = []
        if ego_vehicle is not None:
            actors_to_destroy.append(ego_vehicle)
        actors_to_destroy.extend(npc_vehicles)
        for a in actors_to_destroy:
            try:
                a.destroy()
            except RuntimeError:
                pass

        # Restore asynchronous mode if we changed it
        try:
            settings = world.get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
        except RuntimeError:
            pass

if __name__ == "__main__":
    main()
