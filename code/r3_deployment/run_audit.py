"""Registered static observation audit; each model's formal run executes once."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r3_selectors import load_checkpoint, predict_arrays
from r3_selectors.pipeline import assemble_batch, forward_logits
from transforms import (alternate_query, clone, delete_node, graph_hash, load_graph, permute_nodes,
                        position_noise, recover_frame, relation_matches, rule_prediction, validate_interface)

ROOT_SEED = 20261002
REPLICATES = tuple(range(ROOT_SEED, ROOT_SEED+10))
THRESHOLDS = (.5, .7, .9)
MODEL_NAMES = ("graph_relation_gate", "matched_geometry_mlp", "query_geometry_attention", "drivetransformer_task_adapted")
METHODS = (*MODEL_NAMES, "coarse_rule", "fine_rank_rule")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def batch_predict(bundle, graphs, batch_size=128):
    results = []
    for offset in range(0, len(graphs), batch_size):
        part = graphs[offset:offset+batch_size]
        samples = [{"nodes": g["node_feats"], "edge_index": g["edge_index"], "edge_attr": g["edge_attr"],
                    "query": g["lang_feat"], "candidates": g["candidate_mask"],
                    "target": int(np.flatnonzero(g["candidate_mask"])[0])} for g in part]
        try:
            batch = assemble_batch(samples, bundle["normalization"], bundle["device"])
            with torch.no_grad():
                logits = forward_logits(bundle["model"], batch)
                masked = logits.masked_fill(~batch["candidates"], -1e9)
                probabilities = masked.softmax(-1).cpu().numpy()
                logits = logits.cpu().numpy()
            part_results = []
            for j, graph in enumerate(part):
                indices = np.flatnonzero(graph["candidate_mask"])
                score = logits[j, indices]
                if not np.isfinite(score).all():
                    raise ValueError("Nonfinite model logits")
                pred = int(indices[np.argmax(score)])
                ordered = np.sort(score)
                part_results.append({"pred_index": pred, "raw_max_softmax": float(probabilities[j, pred]),
                                "margin": float(ordered[-1]-ordered[-2]) if len(indices)>1 else None,
                                "relation_fallback": None, "candidate_logits": score.tolist()})
            results.extend(part_results)
        except Exception as exc:
            # No resampling/retry: every graph in this failed inference batch is
            # retained as INVALID_GRAPH/exception in the registered denominator.
            results.extend({"pred_index": None, "raw_max_softmax": None, "margin": None,
                            "relation_fallback": None, "error": repr(exc)} for _ in part)
    return results


def predict(method, bundle, graphs):
    if method in ("coarse_rule", "fine_rank_rule"):
        return [rule_prediction(g, fine=method == "fine_rank_rule") for g in graphs]
    return batch_predict(bundle, graphs)


def prepare_cases(records, track):
    cases = []
    def add(row, graph, condition, seed=None, age_ms=0, removed_id=None):
        cases.append({"record": row, "graph": graph, "condition": condition, "seed": seed,
                      "age_ms": age_ms, "removed_id": removed_id, "input_sha256": graph_hash(graph),
                      "validator_status": validate_interface(graph, track, age_ms)})
    for row in records:
        graph = load_graph(row["npz_path"]); recover_frame(graph)
        target = row["target_index"]
        add(row, graph, "clean")
        for sigma in (.25, .5, 1., 2.):
            for seed in REPLICATES:
                add(row, position_noise(graph, sigma, seed, row["scene_id"]), f"position_noise_sigma_{sigma:g}m", seed)
        decoys = np.flatnonzero(graph["candidate_mask"]); decoys = decoys[decoys != target]
        for seed in REPLICATES:
            rng = np.random.default_rng(np.random.SeedSequence([seed, row["scene_id"], 2]))
            removed = int(rng.choice(decoys)); changed, _ = delete_node(graph, removed)
            add(row, changed, "distractor_missing", seed, removed_id=int(graph["node_spawn_indices"][removed]))
        changed, _ = delete_node(graph, target)
        add(row, changed, "target_missing", removed_id=int(graph["node_spawn_indices"][target]))
        changed = clone(graph); changed["candidate_mask"][:] = False
        add(row, changed, "empty_candidates")
        changed = clone(graph); changed["lang_feat"][:9] = 0
        add(row, changed, "unknown_query_zero_relation")
        changed = clone(graph); changed["lang_feat"][:5] = 0; changed["lang_feat"][:2] = 1
        add(row, changed, "unknown_query_multiple_relation")
        for seed in REPLICATES:
            rng = np.random.default_rng(np.random.SeedSequence([seed, row["scene_id"], 3]))
            changed, _ = permute_nodes(graph, rng.permutation(len(graph["node_feats"])))
            add(row, changed, "node_order_permutation", seed)
        add(row, graph, "timestamp_valid_age_0ms", age_ms=0)
        add(row, graph, "timestamp_stale_age_500ms", age_ms=500)
    return cases


def validation_threshold(method, bundle, records):
    if method not in MODEL_NAMES:
        return {"selected_threshold": None, "score_wrapper_applicable": False, "reason": "Deterministic rule has no softmax score"}
    graphs = [load_graph(r["npz_path"]) for r in records]
    predictions = predict(method, bundle, graphs)
    points = []
    for threshold in THRESHOLDS:
        accepted = [i for i,p in enumerate(predictions) if p["pred_index"] is not None and p["raw_max_softmax"] >= threshold]
        correct = sum(predictions[i]["pred_index"] == records[i]["target_index"] for i in accepted)
        points.append({"threshold": threshold, "n": len(records), "accepted": len(accepted), "correct_accepted": correct,
                       "coverage": len(accepted)/len(records), "accepted_accuracy": correct/len(accepted) if accepted else None})
    eligible = [p for p in points if p["accepted"] and p["accepted_accuracy"] >= .95]
    chosen = max(eligible, key=lambda p: (p["coverage"], p["threshold"])) if eligible else points[-1]
    return {"selected_threshold": chosen["threshold"], "score_wrapper_applicable": True,
            "selection_rule": "Fixed .5/.7/.9 candidates; maximize clean-validation coverage with >=95% accepted accuracy; ties use higher threshold. If none qualify use .9 and disclose failure.",
            "validation_requirement_met": bool(eligible), "validation_points": points,
            "validation_only": True, "target_missing_validation_used": False, "calibrated_uncertainty": False}


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(ROOT_SEED)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def condition_summary(records, selected_threshold):
    result = {}
    for condition in sorted({r["condition"] for r in records}):
        part = [r for r in records if r["condition"] == condition]
        n = len(part); covered = [r for r in part if r["selected_track_id"] is not None]
        true_present = condition not in ("target_missing", "empty_candidates", "unknown_query_zero_relation",
                                         "unknown_query_multiple_relation", "timestamp_stale_age_500ms")
        correct = sum(r["correct_true_identity"] for r in part)
        clustered = defaultdict(list)
        for row in part:
            clustered[row["group_id"]].append(float(row["correct_true_identity"]))
        group_mean = [float(np.mean(v)) for v in clustered.values()]
        conf = [r["raw_max_softmax"] for r in part if r["raw_max_softmax"] is not None]
        point = {"n": n, "graph_clusters": len(clustered), "correct": correct,
                 "full_denominator_accuracy": correct/n if true_present else None,
                 "accepted": len(covered), "coverage": len(covered)/n,
                 "wrong_nonnull_commands": sum(not r["correct_true_identity"] for r in covered),
                 "wrong_nonnull_command_rate": sum(not r["correct_true_identity"] for r in covered)/n,
                 "accepted_error_rate": sum(not r["correct_true_identity"] for r in covered)/len(covered) if covered else None,
                 "null_count": n-len(covered), "inference_exception_count": sum(r["inference_exception"] is not None for r in part),
                 "status_counts": dict(Counter(r["status"] for r in part)),
                 "clean_original_id_agreement": sum(r["same_original_identity_as_clean"] for r in part)/n,
                 "raw_max_softmax_quantiles": np.quantile(conf, [0,.25,.5,.75,1]).tolist() if conf else None,
                 "score_is_calibrated_uncertainty": False, "threshold_points": []}
        if true_present:
            point.update({"scene_cluster_mean_accuracy": float(np.mean(group_mean)),
                          "scene_cluster_bootstrap_ci95": bootstrap(group_mean), "bootstrap_samples": 10000})
        if conf:
            for threshold in THRESHOLDS:
                accepted = [r for r in part if r["selected_track_id"] is not None and r["raw_max_softmax"] >= threshold]
                correct_accepted = sum(r["correct_true_identity"] for r in accepted)
                point["threshold_points"].append({"threshold": threshold, "accepted": len(accepted),
                    "coverage": len(accepted)/n, "wrong_nonnull_command_rate": (len(accepted)-correct_accepted)/n,
                    "accepted_error_rate": (len(accepted)-correct_accepted)/len(accepted) if accepted else None,
                    "full_denominator_accuracy": correct_accepted/n if true_present else None,
                    "selected_by_clean_validation": threshold == selected_threshold})
        fallback = [r["relation_fallback"] for r in part if r["relation_fallback"] is not None]
        point["relation_gate_fallback_rate"] = float(np.mean(fallback)) if fallback else None
        point["relation_gate_note"] = "Rule relation-mask fallback; learned soft feature gates do not filter candidates and have no hard-gate fallback."
        result[condition] = point
    return result


def query_flip(method, bundle, records, track, output_dir):
    pairs, exclusions = [], []
    for row in records:
        graph = load_graph(row["npz_path"])
        generated, reason = alternate_query(graph, track, row["target_index"])
        if generated is None:
            exclusions.append({"sample_id": row["sample_id"], "reason": reason}); continue
        changed, target, proof = generated
        # The graph tensors are fixed; only the query and task's evaluation label differ.
        for key in ("node_feats", "edge_index", "edge_attr", "candidate_mask"):
            assert np.array_equal(graph[key], changed[key])
        zero = clone(graph); zero["lang_feat"][:] = 0
        pairs.append({"row": row, "a": graph, "b": changed, "zero": zero, "target_a": row["target_index"],
                      "target_b": target, "proof": proof})
    normal = predict(method, bundle, [p[q] for p in pairs for q in ("a", "b")])
    no_query = predict(method, bundle, [p["zero"] for p in pairs])
    results = []; counts = defaultdict(lambda: Counter(tasks=0, correct=0, pairs=0, both_correct=0, switched=0, invalid=0))
    for i, pair in enumerate(pairs):
        a, b = normal[2*i], normal[2*i+1]
        zero = no_query[i]
        for condition, pred_a, pred_b in (("normal", a, b), ("no_query", zero, zero), ("query_swap", b, a)):
            ca, cb = pred_a["pred_index"] == pair["target_a"], pred_b["pred_index"] == pair["target_b"]
            count = counts[condition]; count.update(tasks=2, correct=int(ca)+int(cb), pairs=1,
                both_correct=int(ca and cb), switched=int(pred_a["pred_index"] is not None and pred_b["pred_index"] is not None and pred_a["pred_index"] != pred_b["pred_index"]),
                invalid=int(pred_a["pred_index"] is None)+int(pred_b["pred_index"] is None))
            results.append({"sample_id": pair["row"]["sample_id"], "group_id": pair["row"]["group_id"],
                "condition": condition, "target_a_index": pair["target_a"], "target_b_index": pair["target_b"],
                "pred_a": pred_a["pred_index"], "pred_b": pred_b["pred_index"], "correct_a": ca, "correct_b": cb,
                "query_a": pair["a"]["lang_feat"].tolist(), "query_b": pair["b"]["lang_feat"].tolist(),
                "proof": pair["proof"], "source_npz_sha256": pair["row"]["source_npz_sha256"],
                "graph_sha256": pair["row"]["canonical_graph_sha256"], "structural_validator_bypassed_for_raw_diagnostic": condition == "no_query"})
    with (output_dir / "query_flip_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in results: handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
    dump(output_dir / "query_flip_exclusions.json", exclusions)
    return {"eligible_pairs": len(pairs), "original_task_denominator": len(records), "excluded": len(exclusions),
            "family": "same-relation policy flip" if track == "hard_confusable" else "relation flip with unique relation target",
            "same_graph_not_original_queryflip_benchmark": True, "selection_uses_model_outcomes": False,
            "conditions": {condition: {**dict(c), "task_accuracy": c["correct"]/c["tasks"] if c["tasks"] else None,
                 "both_correct_rate": c["both_correct"]/c["pairs"] if c["pairs"] else None,
                 "switch_rate": c["switched"]/c["pairs"] if c["pairs"] else None} for condition,c in counts.items()}}


def aggregate(track_root):
    summaries = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in track_root.glob("*/summary.json")}
    dump(track_root / "summary.json", summaries)
    text = "# Static deployment input audit\n\nAll values below include registered failures/abstentions. Input perturbations are synthetic static graph conditions, not detector/tracker or driving safety evidence.\n\n"
    text += "| Method | Clean accuracy | sigma 2m accuracy | Target-missing wrong issuance | Permutation original-ID agreement |\n|---|---:|---:|---:|---:|\n"
    for name, s in summaries.items():
        c=s["conditions"]
        text+=f"| {name} | {c['clean']['full_denominator_accuracy']:.3%} | {c['position_noise_sigma_2m']['full_denominator_accuracy']:.3%} | {c['target_missing']['wrong_nonnull_command_rate']:.3%} | {c['node_order_permutation']['clean_original_id_agreement']:.3%} |\n"
    text += "\nTarget missing keeps decoys and does not disclose the absent true identity to the adapter/model. All threshold .5/.7/.9 coverage/error points and the clean-validation selected threshold are retained in JSON; raw maximum softmax is not calibrated uncertainty. Rules have no softmax, so the wrapper does not apply. Empty/unknown/stale rejections are explicit structural validator behavior. Node permutation tests representation invariance, not temporal ID continuity or reassociation. Query-flip eligibility uses deterministic task definitions, never model scores; No-Query bypasses the structural validator for a raw ranking diagnostic.\n"
    (track_root / "REPORT.md").write_text(text, encoding="utf-8")


def run_method(root, track, method, cases, test, validation):
    track_root=root/"04_results/deployment"/track; output=track_root/method
    output.mkdir(parents=True, exist_ok=True)
    if (output/"RUN_STARTED.json").exists():
        raise RuntimeError(f"Formal run already started; refusing to overwrite or retry {output}")
    bundle=None; checkpoint_hash=None; batch_check=None
    if method in MODEL_NAMES:
        path=root/"04_results/offline"/track/method/"seed_42/checkpoint.pt"
        bundle=load_checkpoint(path, "cpu"); checkpoint_hash=sha(path)
        # Bounded preflight on eight validation samples, independent of test outcomes.
        sample_graphs=[load_graph(r["npz_path"]) for r in validation[:8]]
        batched=predict(method,bundle,sample_graphs)
        for graph, result in zip(sample_graphs,batched):
            single=predict_arrays(bundle,graph["node_feats"],graph["edge_index"],graph["edge_attr"],graph["lang_feat"],graph["candidate_mask"])
            assert single["pred_index"]==result["pred_index"]
            assert abs(max(single["probabilities"])-result["raw_max_softmax"])<1e-5
        batch_check={"validation_samples":8,"single_vs_batch_choices_equal":True,"max_softmax_tolerance":1e-5}
    threshold=validation_threshold(method,bundle,validation)
    started={"track":track,"method":method,"training_seed":42 if bundle else None,"root_seed":ROOT_SEED,
             "checkpoint_sha256":checkpoint_hash,"registration_sha256":sha(track_root/"REGISTRATION.json"),
             "cases":len(cases),"reruns_permitted":False,"validation_threshold":threshold,"batch_preflight":batch_check}
    dump(output/"RUN_STARTED.json",started)
    good=[i for i,case in enumerate(cases) if case["validator_status"]=="VALID"]
    t=time.perf_counter(); predictions=predict(method,bundle,[cases[i]["graph"] for i in good])
    by_index=dict(zip(good,predictions)); clean_ids={}; results=[]
    for i,case in enumerate(cases):
        row,g=case["record"],case["graph"]; p=by_index.get(i,{"pred_index":None,"raw_max_softmax":None,"margin":None,"relation_fallback":None})
        pred=p["pred_index"]; selected=None if pred is None else int(g["node_spawn_indices"][pred])
        truth=int(row["target_spawn_idx"])
        if case["condition"]=="clean": clean_ids[row["sample_id"]]=selected
        status=case["validator_status"] if case["validator_status"]!="VALID" else "INFERENCE_EXCEPTION" if p.get("error") else "SELECTED"
        results.append({"track":track,"method":method,"training_seed":42 if bundle else None,
            "checkpoint_sha256":checkpoint_hash,"sample_id":row["sample_id"],"scene_id":row["scene_id"],"group_id":row["group_id"],
            "source_npz_sha256":row["source_npz_sha256"],"input_sha256":case["input_sha256"],"schema_version":"32node8edge_v1",
            "condition":case["condition"],"corruption_seed":case["seed"],"declared_age_ms":case["age_ms"],
            "removed_track_id":case["removed_id"],"true_track_id_evaluation_only":truth,"selected_track_id":selected,
            "chosen_index":pred,"status":status,"correct_true_identity":selected==truth,
            "same_original_identity_as_clean":selected==clean_ids.get(row["sample_id"]),
            "raw_max_softmax":p["raw_max_softmax"],"margin":p["margin"],"candidate_logits":p.get("candidate_logits"),
            "relation_match_count":len(relation_matches(g)),"relation_fallback":p["relation_fallback"],"inference_exception":p.get("error"),
            "score_is_calibrated_uncertainty":False,"output_expiry_ms":250})
    with (output/"case_records.jsonl").open("w",encoding="utf-8") as handle:
        for result in results:handle.write(json.dumps(result,ensure_ascii=False,separators=(",",":"))+"\n")
    summary={**started,"wall_seconds":time.perf_counter()-t,"conditions":condition_summary(results,threshold["selected_threshold"]),
             "query_flip":query_flip(method,bundle,test,track,output)}
    dump(output/"summary.json",summary)
    dump(output/"SHA256SUMS.json",{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="SHA256SUMS.json"})
    aggregate(track_root)
    print(json.dumps({"track":track,"method":method,"cases":len(results),"clean_accuracy":summary['conditions']['clean']['full_denominator_accuracy'],
                      "sigma2_accuracy":summary['conditions']['position_noise_sigma_2m']['full_denominator_accuracy'],
                      "target_missing_wrong_issuance":summary['conditions']['target_missing']['wrong_nonnull_command_rate'],
                      "query_flip":summary['query_flip']['conditions']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--track",choices=("scale300","hard_confusable"),required=True)
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS));args=p.parse_args()
    root=args.root.resolve();track_root=root/"04_results/deployment"/args.track;track_root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    test_path=root/"02_protocol"/args.track/"test.jsonl"; val_path=root/"02_protocol"/args.track/"val.jsonl"
    test,validation=rows(test_path),rows(val_path)
    registration={"version":"static_audit_v1_20261002","root_seed":ROOT_SEED,"corruption_seeds":list(REPLICATES),
        "primary_track":"hard_confusable","secondary_track":"scale300","track":args.track,
        "sigma_m":[.25,.5,1,2],"replicates":10,"permutations":10,"softmax_thresholds":list(THRESHOLDS),
        "validation_threshold_selection":"clean val only; max coverage >=95% accepted precision; fallback .9 with explicit failure",
        "models_fixed_before_test":list(MODEL_NAMES),"training_seed":42,"no_results_used_to_select_cases":True,
        "test_manifest_sha256":sha(test_path),"val_manifest_sha256":sha(val_path),
        "deployment_spec_sha256":sha(root/"07_audit/deployment_spec.md"),
        "audit_source_sha256":{q.name:sha(q) for q in Path(__file__).parent.glob("*.py")},
        "query_flip_generation":"first unique different target under alternative policy (hard) or relation (scale), determined solely by fine geometry rule; disclose all exclusions",
        "no_query":"raw ranker diagnostic bypasses query structural validator","invalid_cases_remain_in_denominator":True,
        "bootstrap_seed":ROOT_SEED,"bootstrap_samples":10000,"unit":"group_id; average task/replicate values within each group first"}
    registration_path=track_root/"REGISTRATION.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text(encoding="utf-8"))==registration,"Registration/source changed after formal run"
    else:dump(registration_path,registration)
    cases=prepare_cases(test,args.track)
    assert len(cases)==len(test)*67
    schema={"all_source_graphs_schema_checked":True,"tasks":len(test),"graph_groups":len({r['group_id'] for r in test}),
            "cases":len(cases),"condition_counts":dict(Counter(c['condition'] for c in cases)),
            "structural_status_counts":dict(Counter(c['validator_status'] for c in cases)),
            "zero_noise_node_edge_reconstruction_tolerance":{"atol":1e-4,"rtol":1e-5},"zero_sigma_returns_exact_original_tensors":True}
    dump(track_root/"SCHEMA_AND_CASES.json",schema)
    for method in args.methods:run_method(root,args.track,method,cases,test,validation)


if __name__=="__main__":main()
