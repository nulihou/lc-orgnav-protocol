"""Read-only model comparison and checksum verification of completed static runs."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--track",required=True)
    args=parser.parse_args();track_root=args.root.resolve()/"04_results/deployment"/args.track
    data={}
    for directory in sorted(track_root.iterdir()):
        if not directory.is_dir() or not (directory/"summary.json").exists():continue
        checks=json.loads((directory/"SHA256SUMS.json").read_text(encoding="utf-8"))
        for name,expected in checks.items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected
        with (directory/"case_records.jsonl").open(encoding="utf-8") as handle:
            data[directory.name]=[json.loads(line) for line in handle]
    reference=data["graph_relation_gate"];outputs={}
    for name,records in data.items():
        if name=="graph_relation_gate":continue
        assert len(records)==len(reference)
        grouped=defaultdict(lambda:defaultdict(list))
        for a,b in zip(reference,records):
            for field in ("sample_id","group_id","condition","corruption_seed","input_sha256","true_track_id_evaluation_only"):
                assert a[field]==b[field],(name,field)
            metric="wrong_nonnull_command" if a["condition"]=="target_missing" else "original_ID_agreement" if a["condition"]=="node_order_permutation" else "full_denominator_accuracy"
            field="same_original_identity_as_clean" if metric=="original_ID_agreement" else "correct_true_identity"
            if metric=="wrong_nonnull_command":
                va=float(a["selected_track_id"] is not None and not a["correct_true_identity"])
                vb=float(b["selected_track_id"] is not None and not b["correct_true_identity"])
            else:va,vb=float(a[field]),float(b[field])
            grouped[(a["condition"],metric)][a["group_id"]].append(va-vb)
        comparison={}
        for (condition,metric),groups in grouped.items():
            values=np.asarray([np.mean(groups[key]) for key in sorted(groups)])
            rng=np.random.default_rng(20261002)
            sampled=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
            comparison[condition]={"metric":metric,"graph_clusters":len(values),"graph_minus_comparator_scene_mean":float(values.mean()),
                                   "paired_scene_cluster_bootstrap_ci95":np.quantile(sampled,[.025,.975]).tolist(),
                                   "bootstrap_samples":10000,"seed":20261002}
        outputs[name]=comparison
    payload={"reference":"graph_relation_gate","all_methods_received_identical_corrupted_inputs":True,
             "every_per_method_artifact_sha256_verified":True,"comparisons":outputs}
    (track_root/"PAIRED_COMPARISONS.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    text="# Paired static-input comparison\n\nMean differences first average all task/replicate observations within each canonical graph group. The 95% intervals use 10,000 paired graph-cluster bootstrap samples. Raw model errors and abstentions remain in the denominator.\n\n| Comparator | Clean graph minus comparator | sigma2m graph minus comparator | sigma2m paired CI95 |\n|---|---:|---:|---|\n"
    for name,conditions in outputs.items():
        clean=conditions['clean'];noise=conditions['position_noise_sigma_2m'];ci=noise['paired_scene_cluster_bootstrap_ci95']
        text+=f"| {name} | {100*clean['graph_minus_comparator_scene_mean']:.2f} pp | {100*noise['graph_minus_comparator_scene_mean']:.2f} pp | [{100*ci[0]:.2f}, {100*ci[1]:.2f}] pp |\n"
    text+="\nSynthetic position-noise results do not establish sensor/tracker accuracy or real driving benefit. A gain over a deterministic rule under corruption is restricted to these frozen graphs, levels and seeds. Target-missing wrong issuance and static order invariance are separate metrics, not acquisition or safety outcomes.\n"
    (track_root/"PAIRED_REPORT.md").write_text(text,encoding="utf-8")
    print(args.track,'verified methods',list(data),'comparisons',len(outputs))


if __name__=="__main__":main()
