"""Freeze explicit Oracle and protocol-rule reference scores for CARLA."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.online_multi.eval_gs_proto_driven import choose_target_index

p = argparse.ArgumentParser()
p.add_argument("--protocol-dir", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--rule", choices=["rule_semantic", "fine_rank_rule"], required=True)
a = p.parse_args()
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
rows = {"oracle": [], a.rule: []}
for path in sorted(Path(a.protocol_dir).glob("scen_*.npz")):
    z = np.load(str(path), allow_pickle=False)
    scene = int(path.stem.split("_")[-1])
    choice = choose_target_index(
        torch.zeros(len(z["candidate_mask"])), z["candidate_mask"].astype(bool),
        z["node_feats"], z["lang_feat"], selector_profile=a.rule)
    for method, chosen in [("oracle", int(z["target_idx_gt"])), (a.rule, int(choice["chosen_idx"]))]:
        logits = np.zeros(len(z["candidate_mask"]), dtype=np.float32)
        logits[chosen] = 1.0
        rows[method].append({"scene_id": scene, "pred_index": chosen,
                             "target_index": int(z["target_idx_gt"]), "logits": logits.tolist()})
for method, values in rows.items():
    destination = out / (method + "_scores.jsonl")
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
    print(method, len(values), "correct", sum(r["pred_index"] == r["target_index"] for r in values),
          "sha256", hashlib.sha256(destination.read_bytes()).hexdigest())
