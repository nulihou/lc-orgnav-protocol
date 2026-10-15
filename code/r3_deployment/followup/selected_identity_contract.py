"""Score-blind identity-loss contract on already registered physical deletions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(previous_id, live_ids, validator_status, held_score=None, threshold=None):
    """This function cannot receive the true semantic target identity."""
    if validator_status != "VALID":
        return {"status": validator_status, "selected_id": None}
    if previous_id is None:
        return {"status": "NO_PREVIOUS_SELECTION", "selected_id": None}
    if previous_id not in live_ids:
        return {"status": "TARGET_LOST", "selected_id": None}
    if threshold is not None and held_score is not None and held_score < threshold:
        return {"status": "AMBIGUOUS", "selected_id": None}
    return {"status": "SELECTED_HELD_ID", "selected_id": previous_id}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(root, track, method):
    original=root/"04_results/deployment"/track/method
    output=root/"04_results/deployment/identity_contract"/track/method
    output.mkdir(parents=True,exist_ok=True)
    if (output/"RUN_STARTED.json").exists():
        raise RuntimeError(f"Follow-up already started; refuse overwrite/retry: {output}")
    protocol=root/"02_protocol/SELECTED_IDENTITY_DISAPPEARANCE_FOLLOWUP_20261002.md"
    raw=read_rows(original/"case_records.jsonl")
    meta=json.loads((original/"summary.json").read_text(encoding="utf-8"))
    clean={r["sample_id"]:r for r in raw if r["condition"]=="clean"}
    manifest=read_rows(root/"02_protocol"/track/"test.jsonl")
    candidate_ids={}
    for row in manifest:
        with np.load(row["npz_path"],allow_pickle=False) as arrays:
            candidate_ids[row["sample_id"]]=arrays["node_spawn_indices"][arrays["candidate_mask"]].tolist()
    conditions=[r for r in raw if r["condition"] in ("target_missing","distractor_missing")]
    threshold=meta["validation_threshold"]["selected_threshold"]
    registration={"protocol_sha256":sha(protocol),"followup_code_sha256":sha(Path(__file__)),
        "static_records_sha256":sha(original/"case_records.jsonl"),"checkpoint_sha256":meta["checkpoint_sha256"],
        "track":track,"method":method,"cases":len(conditions),"new_model_inference":False,"new_training":False,
        "ground_truth_available_to_contract":False,"clean_validation_threshold":threshold,"reruns_permitted":False}
    (output/"RUN_STARTED.json").write_text(json.dumps(registration,indent=2)+"\n",encoding="utf-8")
    results=[]
    for case in conditions:
        first=clean[case["sample_id"]]; previous=first["selected_track_id"]
        live=[identifier for identifier in candidate_ids[case["sample_id"]] if identifier!=case["removed_track_id"]]
        assert previous in candidate_ids[case["sample_id"]] or previous is None
        scores=case.get("candidate_logits"); held_score=None
        if scores is not None and previous in live:
            scores=np.asarray(scores,dtype=float); assert len(scores)==len(live)
            probabilities=np.exp(scores-scores.max());probabilities/=probabilities.sum()
            held_score=float(probabilities[live.index(previous)])
        for name,cutoff in [("binding_only",None),("validation_score_threshold",threshold),
                            ("fixed_score_0.5",.5),("fixed_score_0.7",.7),("fixed_score_0.9",.9)]:
            if scores is None and name != "binding_only":
                continue
            decision=contract(previous,live,"VALID" if case["status"]=="SELECTED" else case["status"],held_score,cutoff)
            # Ground truth appears only after contract status/selection has been computed.
            truth=case["true_track_id_evaluation_only"]; issued=decision["selected_id"]
            results.append({"track":track,"method":method,"sample_id":case["sample_id"],"group_id":case["group_id"],
                "physical_deletion_condition":case["condition"],"corruption_seed":case["corruption_seed"],
                "point":name,"threshold":cutoff,"previous_clean_selected_id":previous,
                "previous_clean_semantic_correct":first["correct_true_identity"],"removed_id":case["removed_track_id"],
                "live_candidate_ids":live,"held_identity_current_softmax":held_score,
                "selected_id":issued,"status":decision["status"],"true_target_for_scoring_only":truth,
                "semantic_correct":issued==truth,"semantic_wrong_nonnull":issued is not None and issued!=truth,
                "source_input_sha256":case["input_sha256"],"source_npz_sha256":case["source_npz_sha256"],
                "no_truth_used_in_decision":True,"static_two_payload_test_not_tracking":True})
    with (output/"case_records.jsonl").open("w",encoding="utf-8") as handle:
        for row in results:handle.write(json.dumps(row,separators=(",",":"))+"\n")
    summary={"registration":registration,"conditions":{}}
    for condition in ("target_missing","distractor_missing"):
        summary["conditions"][condition]={}
        for point in sorted({r["point"] for r in results}):
            subset=[r for r in results if r["physical_deletion_condition"]==condition and r["point"]==point]
            n=len(subset);accepted=sum(r["selected_id"] is not None for r in subset);wrong=sum(r["semantic_wrong_nonnull"] for r in subset)
            summary["conditions"][condition][point]={"n":n,"graph_groups":len({r['group_id'] for r in subset}),
                "accepted":accepted,"coverage":accepted/n,"wrong_nonnull":wrong,"wrong_nonnull_rate":wrong/n,
                "accepted_error_rate":wrong/accepted if accepted else None,"semantic_correct":sum(r["semantic_correct"] for r in subset),
                "initial_semantic_wrong":sum(not r["previous_clean_semantic_correct"] for r in subset),
                "target_lost_null":sum(r["status"]=="TARGET_LOST" for r in subset),
                "null_count":n-accepted,"status_counts":dict(Counter(r["status"] for r in subset)),
                "all_missing_previous_ids_return_null":all(r["selected_id"] is None for r in subset if r["previous_clean_selected_id"] not in r["live_candidate_ids"]),
                "accepted_decisions_keep_previous_identity":all(r["selected_id"]==r["previous_clean_selected_id"] for r in subset if r["selected_id"] is not None)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (output/"SHA256SUMS.json").write_text(json.dumps({p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="SHA256SUMS.json"},indent=2)+"\n",encoding="utf-8")
    print(track,method,json.dumps(summary["conditions"]["target_missing"]["binding_only"]),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--track",choices=("scale300","hard_confusable"),required=True);parser.add_argument("--methods",nargs="+",required=True)
    args=parser.parse_args()
    for method in args.methods:run(args.root.resolve(),args.track,method)


if __name__=="__main__":main()
