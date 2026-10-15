"""Freeze outcome-blind graph-disjoint R3 partitions from unchanged R2 NPZs.

Test tasks and labels are retained in their original order. Validation wins over
training, and test wins over both. All queries of one canonical input graph or
one base scene belong to one partition. No prediction or rollout is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TRACKS = ("scale300", "hard_confusable")
SOURCE_TRACKS = {"scale300": "scale300_20260522", "hard_confusable": "hard_confusable_20260526"}
SPLITS = ("train", "val", "test")
GRAPH_ARRAYS = ("node_feats", "candidate_mask", "edge_index", "edge_attr")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        digest.update(stable_json([name, arr.dtype.str, list(arr.shape)]).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def canonical_graph(arrays: dict[str, np.ndarray]) -> tuple[str, int]:
    """Permutation-invariant hash of G; query, target and node IDs are excluded.

    These spatial graphs have unique full node-feature rows. Their lexicographic
    ordering is therefore a canonical labeling. Duplicate feature rows fail
    closed rather than silently claiming general graph-isomorphism support.
    """
    x = np.asarray(arrays["node_feats"])
    mask = np.asarray(arrays["candidate_mask"], dtype=np.bool_)
    ei = np.asarray(arrays["edge_index"], dtype=np.int64)
    ea = np.asarray(arrays["edge_attr"])
    if x.ndim != 2 or mask.shape != (len(x),) or ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("Invalid node, candidate-mask or edge-index shape")
    if ea.ndim != 2 or len(ea) != ei.shape[1] or not np.isfinite(x).all() or not np.isfinite(ea).all():
        raise ValueError("Invalid edge attributes or nonfinite geometry")
    if ei.size and (ei.min() < 0 or ei.max() >= len(x)):
        raise ValueError("Edge references an invalid node")
    node_keys = [tuple(row.tolist()) + (bool(mask[i]),) for i, row in enumerate(x)]
    if len(set(node_keys)) != len(x):
        raise ValueError("Canonical node keys are not unique; explicit isomorphism audit required")
    order = np.asarray(sorted(range(len(x)), key=lambda i: node_keys[i]), dtype=np.int64)
    inverse = np.empty(len(x), dtype=np.int64)
    inverse[order] = np.arange(len(x))
    edges = inverse[ei]
    edge_keys = [tuple(edges[:, j].tolist()) + tuple(ea[j].tolist()) for j in range(len(ea))]
    edge_order = np.asarray(sorted(range(len(ea)), key=lambda j: edge_keys[j]), dtype=np.int64)
    target = int(np.asarray(arrays.get("target_index", arrays["target_idx_gt"])).item())
    if not 0 <= target < len(x) or not mask[target]:
        raise ValueError("Protocol target is outside the valid candidate mask")
    if "target_index" in arrays and int(np.asarray(arrays["target_idx_gt"]).item()) != target:
        raise ValueError("Training and protocol target labels disagree")
    graph_hash = array_hash({"node_feats": x[order], "candidate_mask": mask[order],
                             "edge_index": edges[:, edge_order], "edge_attr": ea[edge_order]})
    return graph_hash, int(inverse[target])


def base_scene_hash(scene: dict) -> str:
    start = 2 if scene.get("anchor_node_idx") == 1 else 1
    key = [scene["town"], int(scene["ego_spawn_idx"]), int(scene["anchor_spawn_idx"]),
           sorted(int(v) for v in scene["node_spawn_indices"][start:])]
    return hashlib.sha256(stable_json(key).encode()).hexdigest()


def assign_groups(records: list[dict]) -> None:
    """Join equal canonical G OR equal base scene, independently of query/label."""
    parent = list(range(len(records)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(records):
        for field in ("canonical_graph_sha256", "canonical_base_scene_sha256"):
            key = (field, row[field])
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
    components: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(records):
        components[find(i)].append(row)
    for rows in components.values():
        identities = sorted({(r["canonical_graph_sha256"], r["canonical_base_scene_sha256"]) for r in rows})
        group_id = hashlib.sha256(stable_json(identities).encode()).hexdigest()
        owner = max((r["original_split"] for r in rows), key=lambda s: SPLITS.index(s))
        for row in rows:
            row["group_id"] = group_id
            row["group_owner_split"] = owner
            row["retained"] = row["original_split"] == owner
            row["split"] = row["original_split"] if row["retained"] else None


def summary(rows: list[dict]) -> dict:
    fields = ("num_candidates", "query_type", "rel_type", "query_policy", "target_slot",
              "target_index", "canonical_target_index", "town", "weather")
    return {"tasks": len(rows), "canonical_graphs": len({r["canonical_graph_sha256"] for r in rows}),
            "base_scenes": len({r["canonical_base_scene_sha256"] for r in rows}),
            "independent_groups": len({r["group_id"] for r in rows}),
            "strata": {field: dict(sorted(Counter(str(r[field]) for r in rows).items())) for field in fields},
            "all_targets_in_candidate_mask": all(r["target_in_candidate_mask"] for r in rows),
            "input_dimensions": dict(sorted(Counter(stable_json(r["input_dimensions"]) for r in rows).items()))}


def intersections(partitions: dict[str, list[dict]]) -> dict:
    result = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        result[f"{a}-{b}"] = {field: len({r[field] for r in partitions[a]} & {r[field] for r in partitions[b]})
                                for field in ("group_id", "canonical_graph_sha256", "canonical_base_scene_sha256")}
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_track(workspace: Path, output: Path, track: str) -> dict:
    source = workspace / "_data_classification_20260531/paper_used_data/runs" / SOURCE_TRACKS[track]
    target_dir = output / track
    (target_dir / "lists").mkdir(parents=True, exist_ok=True)
    records, source_files = [], []
    for split in SPLITS:
        scenario_path = source / "scenarios" / f"{split}_scenarios.json"
        source_files.append(scenario_path)
        scenes = json.loads(scenario_path.read_text(encoding="utf-8"))
        for scene in scenes:
            sid = int(scene["id"])
            npz_path = source / "trainable_graphs" / split / f"scen_{sid:03d}.npz"
            protocol_path = source / "protocol_graphs" / split / npz_path.name
            with np.load(npz_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            with np.load(protocol_path, allow_pickle=False) as loaded:
                protocol_arrays = {name: loaded[name] for name in loaded.files}
            for name in (*GRAPH_ARRAYS, "lang_feat", "target_idx_gt", "node_spawn_indices"):
                if not np.array_equal(arrays[name], protocol_arrays[name]):
                    raise ValueError(f"Trainable and protocol NPZ differ in {name}: {npz_path}")
            graph_hash, canonical_target = canonical_graph(arrays)
            label = int(np.asarray(arrays["target_index"]).item())
            if label != int(scene["target_index_gt"]):
                raise ValueError(f"JSON and NPZ target labels differ: {npz_path}")
            if not np.array_equal(arrays["node_spawn_indices"], np.asarray(scene["node_spawn_indices"])):
                raise ValueError(f"JSON and NPZ node identities differ: {npz_path}")
            records.append({"track": track, "sample_id": f"{track}:{split}:{sid:04d}", "scene_id": sid,
                            "original_split": split, "npz_path": str(npz_path.resolve()),
                            "npz_path_absolute": str(npz_path.resolve()),
                            "npz_path_relative_workspace": npz_path.relative_to(workspace).as_posix(),
                            "protocol_npz_path_relative_workspace": protocol_path.relative_to(workspace).as_posix(),
                            "source_npz_sha256": sha256_file(npz_path),
                            "protocol_npz_sha256": sha256_file(protocol_path),
                            "source_scenario_json_relative_workspace": scenario_path.relative_to(workspace).as_posix(),
                            "canonical_graph_sha256": graph_hash, "canonical_base_scene_sha256": base_scene_hash(scene),
                            "target_index": label, "canonical_target_index": canonical_target,
                            "target_spawn_idx": int(scene["node_spawn_indices"][label]),
                            "target_in_candidate_mask": bool(arrays["candidate_mask"][label]),
                            "query_vector": arrays["lang_feat"].tolist(),
                            "query_sha256": array_hash({"lang_feat": arrays["lang_feat"]}),
                            "num_candidates": int(arrays["candidate_mask"].sum()),
                            "rel_type": int(arrays["rel_type"]), "query_type": scene["query_type"],
                            "query_policy": scene.get("query_policy"), "query_policy_id": scene.get("query_policy_id"),
                            "target_slot": scene.get("target_slot", label), "town": scene["town"],
                            "weather": scene.get("weather", "unknown"),
                            "ego_spawn_idx": scene["ego_spawn_idx"], "anchor_spawn_idx": scene["anchor_spawn_idx"],
                            "node_spawn_indices": scene["node_spawn_indices"], "npz_arrays": sorted(arrays),
                            "input_dimensions": {name: list(arrays[name].shape) for name in (*GRAPH_ARRAYS, "lang_feat")}})
    assign_groups(records)
    historical = {s: [r for r in records if r["original_split"] == s] for s in SPLITS}
    retained = {s: [r for r in historical[s] if r["retained"]] for s in SPLITS}
    cross = intersections(retained)
    if any(count for pair in cross.values() for count in pair.values()):
        raise ValueError("Frozen partitions are not graph/base-scene/group disjoint")
    if [r["sample_id"] for r in retained["test"]] != [r["sample_id"] for r in historical["test"]]:
        raise ValueError("Original test task ordering changed")
    if len(retained["test"]) != 300:
        raise ValueError("Expected all 300 historical test tasks to be retained")
    for split, rows in retained.items():
        content = "".join(stable_json(r) + "\n" for r in rows)
        for name in (f"{split}_manifest.jsonl", f"{split}.jsonl"):
            (target_dir / name).write_text(content, encoding="utf-8")
        (target_dir / "lists" / f"{split}.txt").write_text("".join(r["npz_path"] + "\n" for r in rows), encoding="utf-8")
    removed = [r for r in records if not r["retained"]]
    write_json(target_dir / "removed_historical_samples.json", removed)
    duplicate_groups = defaultdict(list)
    for row in records:
        duplicate_groups[row["group_id"]].append(row)
    duplicates = [{"group_id": key, "members": value} for key, value in sorted(duplicate_groups.items()) if len(value) > 1]
    write_json(target_dir / "historical_duplicate_groups.json", duplicates)
    labels_by_graph = defaultdict(set)
    for row in records:
        labels_by_graph[row["canonical_graph_sha256"]].add((row["query_sha256"], row["canonical_target_index"]))
    labels_by_query = defaultdict(set)
    for row in records:
        labels_by_query[(row["canonical_graph_sha256"], row["query_sha256"])].add(row["canonical_target_index"])
    conflicts = [{"graph_sha256": key[0], "query_sha256": key[1], "canonical_target_indices": sorted(value)}
                for key, value in labels_by_query.items() if len(value) > 1]
    if conflicts:
        raise ValueError(f"Conflicting labels on the same (G,q): {len(conflicts)}")
    audit = {"track": track, "partition_rule": "test > validation > training; entire canonical-G/base-scene groups",
             "outcome_blind": True, "predictions_or_rollouts_read": False,
             "historical": {s: summary(rows) for s, rows in historical.items()},
             "historical_cross_split_intersections": intersections(historical),
             "frozen": {s: summary(rows) for s, rows in retained.items()},
             "removed_counts": {s: len(historical[s])-len(retained[s]) for s in SPLITS},
             "frozen_cross_split_intersections": cross, "all_frozen_intersections_zero": True,
             "unchanged_original_test_tasks_and_order": True, "unchanged_npz_arrays": True,
             "canonical_graphs_with_multiple_query_target_pairs": sum(len(v)>1 for v in labels_by_graph.values()),
             "same_graph_and_query_label_conflicts": conflicts,
             "source_scenario_json_sha256": {str(p.relative_to(workspace).as_posix()): sha256_file(p) for p in source_files}}
    write_json(target_dir / "audit.json", audit)
    protocol = f"""# Frozen R3 {track} protocol (2026-10-02)

The test task sequence and all NPZ tensors/labels are unchanged from R2. This is
an outcome-blind correction: no fitted predictions or online outcomes are read.
Precedence is test, then validation, then training. A task is removed from a
lower-priority partition if its full canonical graph OR query/label-independent
base scene appears in a higher-priority partition. All queries of a retained
group remain in one partition; within-partition query diversity is preserved.

| Partition | Historical tasks | Frozen tasks | Frozen graph clusters | Removed |
|---|---:|---:|---:|---:|
""" + "".join(f"| {s} | {len(historical[s])} | {len(retained[s])} | {audit['frozen'][s]['independent_groups']} | {audit['removed_counts'][s]} |\n" for s in SPLITS) + """

Use lists/train.txt, lists/val.txt and lists/test.txt with the existing dataset
loader. Each line is an absolute path to an unchanged original trainable NPZ.
The JSONL manifests also retain workspace-relative paths for relocation and
SHA-256 source checks. All original graph/query/target arrays remain readable.
The short train.jsonl/val.jsonl/test.jsonl names are exact manifest aliases.

Canonical G consists of the complete node_feats, candidate_mask, edge_index and
edge_attr tensors. Query vector, targets, labels, IDs and source split do not
enter the G hash. Node rows are ordered by complete feature values and candidate
status; edge rows are sorted after remapping. Nonunique node keys fail closed.
The base-scene key is (town, ego spawn, anchor spawn, sorted candidate spawns),
independent of candidate order, target and relation/policy query. Groups join
either matching G hashes or matching base-scene hashes across all partitions.

Validation alone selects models/hyperparameters. Test is read for final scoring
after selection; test labels must never guide model selection, scene filtering
or retry decisions. No missing sample is replaced and no denominator is padded.
Training seeds do not create independent test scenes. Closed-loop seeds are
repeated measurements nested within group_id; statistical clusters use this ID.
Hard-confusable retains 300 query tasks on 299 independent graph groups, including
the two different queries on one graph. Scale-300 retains 300 graph groups.

This correction removes historical train/validation/test graph overlap. It does
not establish unseen maps, weather, perception, tracking or real-world validity;
all source tasks still originate in the same controlled CARLA protocol. Map,
query, policy, candidate-count and target-slot distributions are in audit.json.
Historical duplicate groups and every removed task are archived next to it.
"""
    (target_dir / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    freeze_paths = [p for p in sorted(target_dir.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"]
    freeze = {"algorithm": "SHA-256", "relative_to": str(target_dir.resolve()),
              "artifacts": {p.relative_to(target_dir).as_posix(): sha256_file(p) for p in freeze_paths},
              "source_npz_sha256": {r["npz_path_relative_workspace"]: r["source_npz_sha256"] for r in records},
              "source_protocol_npz_sha256": {r["protocol_npz_path_relative_workspace"]: r["protocol_npz_sha256"] for r in records}}
    write_json(target_dir / "SHA256SUMS.json", freeze)
    return {"track": track, "frozen_counts": {s: len(retained[s]) for s in SPLITS},
            "frozen_groups": {s: audit["frozen"][s]["independent_groups"] for s in SPLITS},
            "removed": audit["removed_counts"], "freeze_manifest_sha256": sha256_file(target_dir/"SHA256SUMS.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if not output.is_relative_to(workspace / "R3_revision_20261002/02_protocol"):
        raise ValueError("Output must remain inside the authorized R3 02_protocol directory")
    report = [freeze_track(workspace, output, track) for track in TRACKS]
    write_json(output / "FREEZE_SUMMARY.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
