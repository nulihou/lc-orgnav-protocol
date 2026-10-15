"""Report all frozen runs; group-level uncertainty is primary."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

NAMES=['graph_relation_gate','matched_geometry_mlp','query_geometry_attention','drivetransformer_task_adapted']
SEEDS=[7,21,42]


def exact_p(wins, losses):
    n=wins+losses
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.0


def group_comparison(a,b,repeats=10000):
    lookup={r['sample_id']:r for r in b}
    if set(lookup)!={r['sample_id'] for r in a}: raise ValueError('Paired test tasks differ')
    groups={}; task_wins=task_losses=0
    for r in a:
        other=lookup[r['sample_id']]
        if r['target_index']!=other['target_index'] or r['group_id']!=other['group_id']:
            raise ValueError('Paired target/group differs')
        difference=int(r['correct'])-int(other['correct'])
        task_wins+=int(difference==1); task_losses+=int(difference==-1)
        groups.setdefault(r['group_id'],[]).append(difference)
    diffs=np.array([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(20261002)
    bootstrap=diffs[rng.integers(0,len(diffs),size=(repeats,len(diffs)))].mean(1)
    return {'groups':len(diffs),'tasks':len(a),'mean_group_accuracy_delta_pp':float(diffs.mean()*100),
            'group_bootstrap_ci95_pp':(np.quantile(bootstrap,[.025,.975])*100).tolist(),
            'group_wins':int((diffs>0).sum()),'group_losses':int((diffs<0).sum()),
            'group_sign_test_p':exact_p(int((diffs>0).sum()),int((diffs<0).sum())),
            'secondary_task_mcnemar_p':exact_p(task_wins,task_losses),
            'task_a_only':task_wins,'task_b_only':task_losses}


def seed_averaged_group_comparison(predictions,name,repeats=10000):
    groups={}
    for seed in SEEDS:
        a=predictions['graph_relation_gate',seed]; b=predictions[name,seed]
        lookup={r['sample_id']:r for r in b}
        for r in a:
            groups.setdefault(r['group_id'],{}).setdefault(seed,[]).append(
                int(r['correct'])-int(lookup[r['sample_id']]['correct']))
    if any(set(v)!=set(SEEDS) for v in groups.values()):raise ValueError('Incomplete seed/group pairing')
    # Average query tasks within each group, then average training seeds.
    diffs=np.array([np.mean([np.mean(v[s]) for s in SEEDS]) for v in groups.values()])
    rng=np.random.default_rng(20261002)
    bootstrap=diffs[rng.integers(0,len(diffs),size=(repeats,len(diffs)))].mean(1)
    return {'groups':len(diffs),'seeds':SEEDS,'seed_averaging_before_group_bootstrap':True,
            'mean_group_accuracy_delta_pp':float(diffs.mean()*100),
            'group_bootstrap_ci95_pp':(np.quantile(bootstrap,[.025,.975])*100).tolist(),
            'group_wins':int((diffs>0).sum()),'group_losses':int((diffs<0).sum()),
            'group_sign_test_p':exact_p(int((diffs>0).sum()),int((diffs<0).sum())),
            'interpretation':'Group CI estimates variation across test groups, conditional on the three fixed training seeds; seed sample std separately reports training variability.'}


def holm(values):
    ordered=sorted(values,key=values.get); m=len(ordered); adjusted={}; previous=0.0
    for i,key in enumerate(ordered):
        previous=max(previous,min(1.0,(m-i)*values[key]));adjusted[key]=previous
    return adjusted


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    source=args.root/'04_results/offline'; rows=[]; incomplete=[]; paired={}; strata=[]
    for track in ['scale300','hard_confusable']:
        predictions={}
        manifest=[json.loads(l) for l in (args.root/'02_protocol'/track/'test.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        strata_lookup={r['sample_id']:r for r in manifest}
        for name in NAMES:
            seedrows=[]
            for seed in SEEDS:
                run=source/track/name/f'seed_{seed}'
                if not (run/'summary.json').exists():
                    incomplete.append(str(run));continue
                s=json.loads((run/'summary.json').read_text(encoding='utf-8'))
                preds=[json.loads(l) for l in (run/'test_predictions.jsonl').read_text().splitlines()]
                if sum(r['correct'] for r in preds)!=s['test']['correct']:raise ValueError('Prediction summary mismatch')
                predictions[name,seed]=preds
                row={'track':track,'model':name,'seed':seed,'test_correct':s['test']['correct'],
                    'test_n':s['test']['total'],'test_accuracy_percent':s['test']['accuracy']*100,
                    'validation_accuracy_percent':s['validation']['accuracy']*100,'best_epoch':s['best_epoch'],
                    'parameters':s['trainable_parameters'],
                    'active_selector_parameters':s['trainable_parameters']-(50311 if name=='graph_relation_gate' else 0),
                    'dormant_auxiliary_parameters':50311 if name=='graph_relation_gate' else 0,
                    'training_seconds':s['train_seconds'],
                    'forward_median_ms':s['timing']['gpu_only_model_forward_ms']['median'],
                    'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256']}
                rows.append(row);seedrows.append(row)
                bystratum={}
                for r in preds:
                    m=strata_lookup[r['sample_id']]
                    for key,value in [('K',m['num_candidates']),('relation',m.get('query_type',m.get('rel_type'))),('policy',m.get('query_policy'))]:
                        if value is not None:bystratum.setdefault((key,str(value)),[]).append(int(r['correct']))
                for (key,value),correct in bystratum.items():
                    strata.append({'track':track,'model':name,'seed':seed,'stratum':key,'value':value,
                                   'n':len(correct),'correct':sum(correct),'accuracy_percent':100*np.mean(correct)})
            if len(seedrows)==3:
                accuracies=np.array([r['test_accuracy_percent'] for r in seedrows])
                paired[f'{track}/{name}/mean']={'mean_percent':float(accuracies.mean()),
                  'sample_std_percent':float(accuracies.std(ddof=1)), 'seeds':SEEDS,
                  'not_independent_test_replications':True}
        for name in NAMES[1:]:
            for seed in SEEDS:
                if ('graph_relation_gate',seed) in predictions and (name,seed) in predictions:
                    paired[f'{track}/graph_vs_{name}/seed_{seed}']=group_comparison(predictions['graph_relation_gate',seed],predictions[name,seed])
            if all(('graph_relation_gate',s) in predictions and (name,s) in predictions for s in SEEDS):
                paired[f'{track}/graph_vs_{name}/seed_averaged']=seed_averaged_group_comparison(predictions,name)
        # These reporting families supplement the frozen training protocol;
        # inferential details were not part of its original preregistration.
        for seedlabel in [f'seed_{s}' for s in SEEDS]+['seed_averaged']:
            keys=[f'{track}/graph_vs_{name}/{seedlabel}' for name in NAMES[1:]]
            if not all(k in paired for k in keys):continue
            adjusted=holm({k:paired[k]['group_sign_test_p'] for k in keys})
            for key in keys:
                paired[key]['group_sign_test_holm_p']=adjusted[key]
                paired[key]['holm_family']=f'{track}/{seedlabel}: 3 graph-versus-baseline contrasts; supplemental analysis'
            if seedlabel!='seed_averaged':
                secondary=holm({k:paired[k]['secondary_task_mcnemar_p'] for k in keys})
                for key in keys:paired[key]['secondary_task_mcnemar_holm_p']=secondary[key]
    dest=source/'all_results.json';dest.write_text(json.dumps({'runs':rows,'comparisons':paired,'strata':strata,
                        'incomplete':incomplete,'primary_uncertainty':'graph/base-scene connected group',
                        'no_test_seed_selection':True,
                        'inferential_analysis_registration':'Supplement added after formal training began; do not describe as originally preregistered statistics.'},indent=2),encoding='utf-8')
    if rows:
        with (source/'all_runs.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# R3 matched baseline results','',
        'All rows use newly graph-disjoint train/validation manifests; historical R2 tables are a separate cohort.',
        'DriveTransformer task-block adapted is a static protocol adapter, not a full-stack SOTA reproduction.',
        'No seed is selected by test accuracy. Group bootstrap and group sign tests are primary; task McNemar is secondary.',
        '', '| Track | Model | Seed | Validation % | Test correct/N | Test % | Best epoch | Parameters |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["track"]}|{r["model"]}|{r["seed"]}|{r["validation_accuracy_percent"]:.2f}|{r["test_correct"]}/{r["test_n"]}|{r["test_accuracy_percent"]:.2f}|{r["best_epoch"]}|{r["parameters"]}|')
    lines+=['','## Three-seed descriptive summaries','',
            '| Track/model | Mean test % | Sample standard deviation (percentage points) |',
            '|---|---:|---:|']
    for key,v in paired.items():
        if key.endswith('/mean'):lines.append(f'|{key[:-5]}|{v["mean_percent"]:.2f}|{v["sample_std_percent"]:.2f}|')
    lines+=['','The three training seeds share one test cohort; the standard deviation measures training variability, not three independent datasets.',
            '', '## Seed-averaged group comparisons','',
            'Positive differences favor Graph+Relation-Gate. Query tasks are averaged inside each graph/base-scene group, then the three training seeds are averaged before test-group resampling. Holm families contain the three graph-versus-baseline contrasts separately for each track. These inferential reporting details supplement the frozen training protocol; they were not originally preregistered.',
            '', '| Comparison | Groups | Delta pp | Group bootstrap 95% CI pp | Group sign Holm p |',
            '|---|---:|---:|---:|---:|']
    for key,v in paired.items():
        if key.endswith('seed_averaged'):
            ci=v['group_bootstrap_ci95_pp']
            lines.append(f'|{key}|{v["groups"]}|{v["mean_group_accuracy_delta_pp"]:+.2f}|[{ci[0]:+.2f}, {ci[1]:+.2f}]|{v.get("group_sign_test_holm_p",float("nan")):.5g}|')
    lines+=['',f'Incomplete runs: {len(incomplete)}.']
    (source/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(rows),'incomplete':len(incomplete),'summary':str(source/'SUMMARY.md')}))


if __name__=='__main__':main()
