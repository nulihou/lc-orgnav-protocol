"""Coherent position corruption and label-blind deployment interface checks."""
from __future__ import annotations

import hashlib
import math

import numpy as np


def load_graph(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in ("node_feats", "edge_index", "edge_attr", "lang_feat", "candidate_mask", "node_spawn_indices")}


def clone(graph):
    return {k: v.copy() for k, v in graph.items()}


def graph_hash(graph):
    h = hashlib.sha256()
    for key in ("node_feats", "edge_index", "edge_attr", "lang_feat", "candidate_mask", "node_spawn_indices"):
        x = np.ascontiguousarray(graph[key])
        h.update(key.encode()); h.update(str(x.shape).encode()); h.update(x.dtype.str.encode()); h.update(x.tobytes())
    return h.hexdigest()


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def orientation(dx, dy, cos_b, sin_b):
    usable = np.hypot(dx, dy) > 1e-4
    angles = wrap(np.arctan2(dy[usable], dx[usable]) - np.arctan2(sin_b[usable], cos_b[usable]))
    if not len(angles):
        raise ValueError("Cannot recover coordinate-frame orientation")
    yaw = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    if np.max(np.abs(wrap(angles-yaw))) > 1e-4:
        raise ValueError("Inconsistent bearing/frame schema")
    return yaw


def recover_frame(graph):
    x = graph["node_feats"].astype(np.float64)
    if x.shape[1] != 32 or graph["edge_attr"].shape[1] != 8:
        raise ValueError("Unsupported feature dimensions")
    ego = np.flatnonzero(np.hypot(x[:, 0], x[:, 1]) < 1e-5)
    anchors = np.flatnonzero(x[:, 11] > .5)
    if len(ego) != 1 or len(anchors) != 1 or np.any(x[:, 12] != 0):
        raise ValueError("Expected one ego, one anchor and zero target indicator")
    ego_idx, anchor_idx = int(ego[0]), int(anchors[0])
    yaw_ego = orientation(x[:, 0], x[:, 1], x[:, 3], x[:, 4])
    yaw_anchor = orientation(x[:, 5], x[:, 6], x[:, 8], x[:, 9])
    frame = {"ego_index": ego_idx, "anchor_index": anchor_idx, "ego_yaw": yaw_ego, "anchor_yaw": yaw_anchor}
    reconstructed = rebuild_geometry(graph, x[:, 0:2], frame)
    if not np.allclose(reconstructed["node_feats"][:, :10], graph["node_feats"][:, :10], atol=1e-4, rtol=1e-5):
        raise ValueError("Zero-noise reconstructed nodes disagree with source schema")
    if not np.allclose(reconstructed["edge_attr"], graph["edge_attr"], atol=1e-4, rtol=1e-5):
        raise ValueError("Zero-noise reconstructed edges disagree with source schema")
    return frame


def rebuild_geometry(graph, positions, frame):
    output = clone(graph)
    positions = np.asarray(positions, dtype=np.float64)
    ego_pos = positions[frame["ego_index"]]
    anchor_pos = positions[frame["anchor_index"]]
    ego = positions-ego_pos
    anchor = positions-anchor_pos
    for delta, offset, yaw in ((ego, 0, frame["ego_yaw"]), (anchor, 5, frame["anchor_yaw"])):
        distance = np.hypot(delta[:, 0], delta[:, 1])
        angle = np.arctan2(delta[:, 1], delta[:, 0])-yaw
        output["node_feats"][:, offset:offset+3] = np.column_stack([delta, distance])
        output["node_feats"][:, offset+3:offset+5] = np.column_stack([np.cos(angle), np.sin(angle)])
    src, dst = graph["edge_index"]
    delta = positions[dst]-positions[src]
    distance = np.hypot(delta[:, 0], delta[:, 1])
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    output["edge_attr"] = np.column_stack([delta, distance, np.cos(angle), np.sin(angle),
                                           distance < 30, np.zeros(len(src)), np.zeros(len(src))]).astype(np.float32)
    return output


def position_noise(graph, sigma, seed, scene_id):
    frame = recover_frame(graph)
    if sigma == 0:
        return clone(graph)
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_id, 1]))
    positions = graph["node_feats"][:, :2].astype(np.float64).copy()
    movable = np.ones(len(positions), bool)
    movable[[frame["ego_index"], frame["anchor_index"]]] = False
    positions[movable] += rng.normal(0, sigma, size=(int(movable.sum()), 2))
    return rebuild_geometry(graph, positions, frame)


def delete_node(graph, index):
    keep = np.ones(len(graph["node_feats"]), bool); keep[index] = False
    mapping = np.full(len(keep), -1, np.int64); mapping[keep] = np.arange(int(keep.sum()))
    edge_keep = keep[graph["edge_index"][0]] & keep[graph["edge_index"][1]]
    output = clone(graph)
    for key in ("node_feats", "candidate_mask", "node_spawn_indices"):
        output[key] = graph[key][keep]
    output["edge_index"] = mapping[graph["edge_index"][:, edge_keep]]
    output["edge_attr"] = graph["edge_attr"][edge_keep]
    return output, mapping


def permute_nodes(graph, order):
    order = np.asarray(order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(len(graph["node_feats"]))):
        raise ValueError("Not a complete node permutation")
    mapping = np.empty(len(order), np.int64); mapping[order] = np.arange(len(order))
    output = clone(graph)
    for key in ("node_feats", "candidate_mask", "node_spawn_indices"):
        output[key] = graph[key][order]
    output["edge_index"] = mapping[graph["edge_index"]]
    return output, mapping


def validate_interface(graph, track, age_ms=0):
    """Never receives or consults a target label or a true target identity."""
    x, ei, ea, mask, q = (graph[k] for k in ("node_feats", "edge_index", "edge_attr", "candidate_mask", "lang_feat"))
    if x.ndim != 2 or x.shape[1] != 32 or ei.shape[0] != 2 or ea.shape != (ei.shape[1], 8) or mask.shape != (len(x),) or q.shape != (32,):
        return "INVALID_GRAPH"
    if not np.isfinite(x).all() or not np.isfinite(ea).all() or not np.isfinite(q).all():
        return "INVALID_GRAPH"
    if ei.size and (ei.min() < 0 or ei.max() >= len(x)):
        return "INVALID_GRAPH"
    if len(set(graph["node_spawn_indices"].tolist())) != len(x):
        return "INVALID_GRAPH"
    if age_ms > 250 or age_ms < 0:
        return "STALE_INPUT"
    if not mask.any():
        return "NO_CANDIDATE"
    if np.sum(q[:5] > .5) != 1 or not np.all((q[:5] == 0) | (q[:5] == 1)):
        return "UNSUPPORTED_QUERY"
    if track == "hard_confusable" and (np.sum(q[6:9] > .5) != 1 or np.any(q[5] != 0)):
        return "UNSUPPORTED_QUERY"
    return "VALID"


def relation_matches(graph, relation=None):
    q, x = graph["lang_feat"], graph["node_feats"]
    relation = int(np.argmax(q[:5])) if relation is None else relation
    d = x[:, 7]; theta = np.arctan2(x[:, 9], x[:, 8])
    if relation == 0:
        valid = d < 50
    elif relation == 1:
        valid = (d > 5) & (d < 40) & (np.abs(theta) < np.pi/6)
    elif relation == 2:
        valid = (d > 5) & (d < 40) & (np.abs(np.abs(theta)-np.pi) < np.pi/6)
    elif relation == 3:
        valid = (d > 5) & (d < 40) & (theta > np.pi/3) & (theta < 2*np.pi/3)
    elif relation == 4:
        valid = (d > 5) & (d < 40) & (theta > -2*np.pi/3) & (theta < -np.pi/3)
    else:
        valid = np.zeros(len(x), bool)
    return np.flatnonzero(graph["candidate_mask"] & valid)


def rule_prediction(graph, fine=False):
    matches = relation_matches(graph)
    candidates = np.flatnonzero(graph["candidate_mask"])
    pool = matches if len(matches) else candidates
    if not len(pool):
        return {"pred_index": None, "raw_max_softmax": None, "margin": None, "relation_fallback": None}
    policy = int(np.argmax(graph["lang_feat"][5:9])) if fine else 0
    x = graph["node_feats"]
    scores = (-x[:, 7] if policy == 0 else x[:, 7] if policy == 1 else
              x[:, 7]*x[:, 9] if policy == 2 else -x[:, 7]*x[:, 9])
    pred = int(pool[int(np.argmax(scores[pool]))])
    sorted_scores = np.sort(scores[pool])
    margin = float(sorted_scores[-1]-sorted_scores[-2]) if len(pool)>1 else None
    return {"pred_index": pred, "raw_max_softmax": None, "margin": margin, "relation_fallback": not bool(len(matches))}


def alternate_query(graph, track, original_target_index):
    """Generate and prove a second unique query target on the unchanged G.

    Eligibility is determined by the frozen task definition only, never by a
    learned-model score. The original rule must first reproduce its stored label.
    """
    original = rule_prediction(graph, fine=(track == "hard_confusable"))
    if original["pred_index"] != original_target_index:
        return None, "original_query_rule_label_mismatch"
    if track == "hard_confusable":
        matches = relation_matches(graph)
        if len(matches) < 2:
            return None, "fewer_than_two_relation_matches"
        if original["margin"] is None or original["margin"] <= 1e-5:
            return None, "original_query_target_not_unique"
        old_policy = int(np.argmax(graph["lang_feat"][5:9]))
        for policy in (1, 2, 3):
            if policy == old_policy:
                continue
            changed = clone(graph); changed["lang_feat"][5:9] = 0; changed["lang_feat"][5+policy] = 1
            candidate = rule_prediction(changed, fine=True)
            if candidate["pred_index"] != original_target_index and candidate["margin"] is not None and candidate["margin"] > 1e-5:
                return (changed, candidate["pred_index"], {"query_change": "policy", "relation_matches": matches.tolist(),
                        "alternate_policy": policy, "alternate_rule_margin": candidate["margin"]}), None
    else:
        old_relation = int(np.argmax(graph["lang_feat"][:5]))
        if len(relation_matches(graph, old_relation)) != 1:
            return None, "original_relation_target_not_unique"
        for relation in (1, 2, 3, 4):
            if relation == old_relation:
                continue
            matches = relation_matches(graph, relation)
            if len(matches) == 1 and int(matches[0]) != original_target_index:
                changed = clone(graph); changed["lang_feat"][:5] = 0; changed["lang_feat"][relation] = 1
                return (changed, int(matches[0]), {"query_change": "relation", "alternate_relation": relation,
                        "alternate_relation_matches": matches.tolist()}), None
    return None, "no_distinct_unique_alternate_target"
