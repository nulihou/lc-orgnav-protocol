"""Verify complete audit packages and export compact, adverse-result tables."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS=("graph_relation_gate","matched_geometry_mlp","query_geometry_attention","drivetransformer_task_adapted","coarse_rule","fine_rank_rule")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();output=root/"04_results/deployment"
    static_rows=[];threshold_rows=[];query_rows=[];contract_rows=[];verification={}
    for track in ("scale300","hard_confusable"):
        track_root=output/track
        registration=json.loads((track_root/"REGISTRATION.json").read_text(encoding="utf-8"))
        for name,expected in registration["audit_source_sha256"].items():
            assert digest(root/"03_code/r3_deployment"/name)==expected
        assert digest(root/"02_protocol"/track/"test.jsonl")==registration["test_manifest_sha256"]
        assert digest(root/"02_protocol"/track/"val.jsonl")==registration["val_manifest_sha256"]
        total_cases=0
        for method in METHODS:
            directory=track_root/method;s=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
            hashes=json.loads((directory/"SHA256SUMS.json").read_text(encoding="utf-8"))
            for name,expected in hashes.items():assert digest(directory/name)==expected
            assert s["registration_sha256"]==digest(track_root/"REGISTRATION.json")
            if s["checkpoint_sha256"]:
                assert digest(root/"04_results/offline"/track/method/"seed_42/checkpoint.pt")==s["checkpoint_sha256"]
            assert s["cases"]==20100;total_cases+=s["cases"]
            c=s["conditions"]
            for condition,p in c.items():
                static_rows.append({"track":track,"method":method,"condition":condition,"n":p["n"],"graph_clusters":p["graph_clusters"],
                    "correct":p["correct"],"accuracy":p["full_denominator_accuracy"],"coverage":p["coverage"],"wrong_nonnull_rate":p["wrong_nonnull_command_rate"],
                    "accepted_error_rate":p["accepted_error_rate"],"null_count":p["null_count"],"inference_exceptions":p["inference_exception_count"],
                    "clean_ID_agreement":p["clean_original_id_agreement"],"bootstrap_ci95":p.get("scene_cluster_bootstrap_ci95")})
                for point in p["threshold_points"]:
                    threshold_rows.append({"track":track,"method":method,"condition":condition,**point})
            assert c["target_missing"]["n"]==300
            assert c["empty_candidates"]["status_counts"]=={"NO_CANDIDATE":300}
            assert c["unknown_query_zero_relation"]["status_counts"]=={"UNSUPPORTED_QUERY":300}
            assert c["unknown_query_multiple_relation"]["status_counts"]=={"UNSUPPORTED_QUERY":300}
            assert c["timestamp_stale_age_500ms"]["status_counts"]=={"STALE_INPUT":300}
            for condition,p in s["query_flip"]["conditions"].items():
                query_rows.append({"track":track,"method":method,"condition":condition,
                    "eligible_pairs":s["query_flip"]["eligible_pairs"],"excluded_source_tasks":s["query_flip"]["excluded"],**p})
            followup=output/"identity_contract"/track/method
            f=json.loads((followup/"summary.json").read_text(encoding="utf-8"))
            hashes=json.loads((followup/"SHA256SUMS.json").read_text(encoding="utf-8"))
            for name,expected in hashes.items():assert digest(followup/name)==expected
            assert f["registration"]["static_records_sha256"]==digest(directory/"case_records.jsonl")
            assert f["registration"]["protocol_sha256"]==digest(root/"02_protocol/SELECTED_IDENTITY_DISAPPEARANCE_FOLLOWUP_20261002.md")
            assert f["registration"]["followup_code_sha256"]==digest(root/"03_code/r3_deployment/followup/selected_identity_contract.py")
            for condition,points in f["conditions"].items():
                for point,p in points.items():
                    assert p["all_missing_previous_ids_return_null"] and p["accepted_decisions_keep_previous_identity"]
                    contract_rows.append({"track":track,"method":method,"condition":condition,"point":point,**p})
        verification[track]={"methods":len(METHODS),"static_case_records":total_cases,"source_and_checkpoint_and_output_hashes_verified":True,
            "structural_rejections_verified":True,"identity_loss_no_substitution_verified":True,"rerun_existing_formal_cases":False}
    def table(name,values):
        fields=list(dict.fromkeys(k for row in values for k in row))
        with (output/name).open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(values)
    table("static_condition_metrics.csv",static_rows);table("raw_score_coverage_error.csv",threshold_rows)
    table("same_graph_query_flip_metrics.csv",query_rows);table("identity_contract_metrics.csv",contract_rows)
    (output/"INDEPENDENT_PACKAGE_VERIFICATION.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")
    text="# Completed static-input audit and identity contract follow-up\n\nAll six methods were evaluated once on every registered payload for each track. There are 241,200 per-method static case records across the two tracks; repeated corruptions are clustered within the 300 Scale and 299 Hard base graph groups. No inference/adapter failure was resampled. These are synthetic static observation inputs.\n\n"
    text+="| Track / method | Clean correct / 300 | sigma2m correct / 3000 | Raw target-missing wrong / 300 | Identity-contract held wrong / 300 |\n|---|---:|---:|---:|---:|\n"
    for track in ("scale300","hard_confusable"):
        for method in METHODS:
            s=json.loads((output/track/method/"summary.json").read_text());f=json.loads((output/"identity_contract"/track/method/"summary.json").read_text())
            c=s["conditions"];binding=f["conditions"]["target_missing"]["binding_only"]
            text+=f"| {track} / {method} | {c['clean']['correct']} | {c['position_noise_sigma_2m']['correct']} | {c['target_missing']['wrong_nonnull_commands']} | {binding['wrong_nonnull']} |\n"
    text+="\nThe forced-choice target-missing failures remain in the formal audit. Fixed score thresholds and a clean-validation selected threshold do not establish reliable absent-target rejection. The separately registered identity contract only rejects when the *previous predicted ID* is absent from the live candidate payload; an initially wrong ID that stays live can remain wrong. Semantic truth is used for scoring only. This is a static two-payload binding check, not tracker loss/reacquisition or driving safety validation.\n\nEmpty candidates, unsupported queries and stale declared metadata produce null outputs via explicit schema logic, not learned recognition. Static permutations preserve original-ID choices but do not test temporal ID switches. Hard same-G policy flips use 296 of 300 source tasks; Scale relation flips use 131, with all exclusions disclosed independently of model scores. Paired scene-cluster intervals and all score/error/coverage points are saved beside this report.\n"
    (output/"REPORT.md").write_text(text,encoding="utf-8")
    freeze_path=output/"DEPLOYMENT_SHA256.json"
    files=[p for d in (output,root/"03_code/r3_deployment") for p in d.rglob('*') if p.is_file() and p!=freeze_path and '__pycache__' not in p.parts]
    files.append(root/"02_protocol/SELECTED_IDENTITY_DISAPPEARANCE_FOLLOWUP_20261002.md")
    payload={"algorithm":"SHA-256","files_relative_to_R3":{p.relative_to(root).as_posix():digest(p) for p in sorted(files)}}
    freeze_path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(verification,indent=2));print('deployment_package_sha256',digest(freeze_path))


if __name__=="__main__":main()
