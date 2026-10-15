"""Replay frozen selector scores into the unchanged shared CARLA controller.

This evaluates target acquisition, not sensor perception or online tracking.
The full graph/query fingerprint must match the graph used for prediction.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from scripts.online_multi import eval_gs_proto_driven as evaluation


def fingerprint(node, edges, attrs, query, mask):
    digest = hashlib.sha256()
    for array, dtype in [(node, "<f4"), (edges, "<i8"), (attrs, "<f4"),
                         (query, "<f4"), (mask, "u1")]:
        value = np.asarray(array, dtype=dtype)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class FrozenScores:
    def __init__(self, protocol_dir, predictions):
        self.scores = {}
        self.record_count = 0
        for line in Path(predictions).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = int(row["scene_id"])
            graph = np.load(str(Path(protocol_dir) / ("scen_%03d.npz" % scene_id)),
                            allow_pickle=False)
            key = fingerprint(graph["node_feats"], graph["edge_index"],
                              graph["edge_attr"], graph["lang_feat"], graph["candidate_mask"])
            scores = np.asarray(row["logits"], dtype=np.float32)
            mask = graph["candidate_mask"].astype(bool)
            if scores.shape != mask.shape:
                raise ValueError("score shape mismatch for scene %d" % scene_id)
            if not np.isfinite(scores[mask]).all():
                raise ValueError("nonfinite candidate scores")
            chosen = int(np.argmax(np.where(mask, scores, -np.inf)))
            if chosen != int(row["pred_index"]):
                raise ValueError("prediction/index mismatch")
            if key in self.scores and not np.array_equal(self.scores[key], scores):
                raise ValueError("identical graph/query has conflicting predictions")
            self.scores[key] = scores
            self.record_count += 1

    def eval(self):
        return self

    def forward_single(self, node_feats, edge_index, edge_attr, lang_feat, candidate_mask):
        arrays = [x.detach().cpu().numpy() for x in
                  [node_feats, edge_index, edge_attr, lang_feat, candidate_mask]]
        key = fingerprint(*arrays)
        if key not in self.scores:
            raise ValueError("evaluation input has no frozen prediction")
        return {"goal_logits": torch.from_numpy(self.scores[key].copy())}


def main():
    start = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed-config", required=True)
    p.add_argument("--runs-per-scene", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--controller-profile", default="improved_v20")
    p.add_argument("--method", required=True)
    p.add_argument("--port", type=int, default=2010)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("existing results will not be overwritten: " + args.output)
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    model = FrozenScores(args.protocol_dir, args.predictions)
    import carla
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30.0)
    world = evaluation.setup_world(client, scenarios[0]["town"], scenarios[0]["weather"])
    results = evaluation.run_evaluation(
        client=client, scenarios=scenarios, world=world, model=model,
        device=torch.device("cpu"), max_steps=args.max_steps,
        output_path=args.output, protocol_dir=args.protocol_dir,
        seed_list=evaluation.load_seed_config(args.seed_config),
        num_runs_per_scene=args.runs_per_scene, tm_port=args.port + 6000,
        fixed_delta_seconds=0.05, no_rendering=True,
        controller_profile=args.controller_profile, selector_profile="neural",
        model_type="r3_frozen_" + args.method)
    expected = len(scenarios) * args.runs_per_scene
    emitted = [json.loads(line) for line in Path(args.output).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(emitted) != expected:
        raise RuntimeError("incomplete episode denominator: %d/%d" % (len(emitted), expected))
    metadata = vars(args)
    metadata.update(expected_episodes=expected, emitted_episodes=len(emitted),
                    valid_episodes=sum(row.get("valid", True) is not False for row in emitted),
                    elapsed_seconds=time.monotonic() - start,
                    predictions_sha256=hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
                    scope="frozen-input one-shot target selection; no sensor/tracker validation")
    Path(args.output + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
