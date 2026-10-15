"""Independently verify persisted R3 manifests, hashes and immutable source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    args = parser.parse_args()
    workspace, root = args.workspace.resolve(), args.protocol_root.resolve()
    report = {}
    for track in ("scale300", "hard_confusable"):
        track_root = root / track
        freeze = json.loads((track_root / "SHA256SUMS.json").read_text(encoding="utf-8"))
        audit = json.loads((track_root / "audit.json").read_text(encoding="utf-8"))
        for relative, expected in audit["source_scenario_json_sha256"].items():
            assert digest(workspace / relative) == expected, (track, "scenario_json_hash", relative)
        for relative, expected in freeze["artifacts"].items():
            assert digest(track_root / relative) == expected, (track, "artifact_hash", relative)
        for field in ("source_npz_sha256", "source_protocol_npz_sha256"):
            for relative, expected in freeze[field].items():
                assert digest(workspace / relative) == expected, (track, "source_hash", relative)
        partitions = {}
        for split in ("train", "val", "test"):
            manifest = track_root / f"{split}_manifest.jsonl"
            assert manifest.read_bytes() == (track_root / f"{split}.jsonl").read_bytes()
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            listed = (track_root / "lists" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            assert listed == [r["npz_path"] for r in rows], (track, split, "list_manifest_order")
            for row in rows:
                path = Path(row["npz_path"])
                assert path.resolve() == (workspace / row["npz_path_relative_workspace"]).resolve()
                with np.load(path, allow_pickle=False) as data:
                    assert int(data["target_index"]) == row["target_index"]
                    assert bool(data["candidate_mask"][row["target_index"]])
                    assert np.array_equal(data["lang_feat"], np.asarray(row["query_vector"]))
                    assert data["node_feats"].shape[1] == 32 and data["edge_attr"].shape[1] == 8
            partitions[split] = rows
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            for field in ("group_id", "canonical_graph_sha256", "canonical_base_scene_sha256"):
                assert not ({r[field] for r in partitions[a]} & {r[field] for r in partitions[b]}), (track, a, b, field)
        test = partitions["test"]
        original_path = workspace / test[0]["source_scenario_json_relative_workspace"]
        original_scenes = json.loads(original_path.read_text(encoding="utf-8"))
        assert [r["scene_id"] for r in test] == [s["id"] for s in original_scenes]
        assert [r["target_index"] for r in test] == [s["target_index_gt"] for s in original_scenes]
        assert len(test) == 300
        report[track] = {"checks_passed": True, "artifacts_hashed": len(freeze["artifacts"]),
                         "original_scenario_jsons_hashed": len(audit["source_scenario_json_sha256"]),
                         "original_trainable_sources_hashed": len(freeze["source_npz_sha256"]),
                         "original_protocol_sources_hashed": len(freeze["source_protocol_npz_sha256"]),
                         "loaded_retained_npz_samples": sum(len(v) for v in partitions.values()),
                         "frozen_counts": {s: len(v) for s, v in partitions.items()},
                         "test_labels_and_order_unchanged": True, "cross_graph_base_group_overlaps": 0}
    (root / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
