"""Audit every unique physical rollout, then map shared outcomes to selectors."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

METHODS = ['graph_relation_gate', 'matched_geometry_mlp',
           'query_geometry_attention', 'drivetransformer_task_adapted']
METRICS = ['semantic_8m', 'semantic_4m', 'selected_8m', 'wrong_reach_8m', 'valid']


def sign_p(values):
    pos = sum(v > 1e-9 for v in values)
    neg = sum(v < -1e-9 for v in values)
    n = pos + neg
    return min(1., 2. * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / 2.**n) if n else 1.


def ci(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261002)
    boot = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def holm(values):
    ordered = sorted(values, key=values.get)
    previous = 0.
    out = {}
    for i, key in enumerate(ordered):
        previous = max(previous, min(1., (len(values)-i)*values[key]))
        out[key] = previous
    return out


def main():
    root = Path(__file__).resolve().parents[2]
    all_summary = {}
    csvrows = []
    for track in ['scale300', 'hard_confusable']:
        directory = root / '04_results/closed_loop' / track
        manifest = json.loads((directory/'execution_library/manifest.json').read_text())
        assignments = {r['scene_id']: r for r in manifest['assignments']}
        files = [(directory/'reference_execution.jsonl',
                  {sid: a['reference_choice'] for sid, a in assignments.items()})]
        for tier, size in enumerate(manifest['tier_sizes']):
            scores = directory/'execution_library'/('tier_%02d_scores.jsonl'%tier)
            choices = {r['scene_id']: r['pred_index'] for r in
                       (json.loads(x) for x in scores.read_text().splitlines() if x.strip())}
            files.append((directory/('tier_%02d_execution.jsonl'%tier), choices))
        library = {}
        hashes = {}
        for path, choices in files:
            metadata = json.loads(Path(str(path)+'.metadata.json').read_text())
            count = 0
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.open(encoding='utf-8'):
                if not line.strip():
                    continue
                r = json.loads(line)
                sid, run = int(r['scene_id']), int(r['run_id'])
                chosen = int(choices[sid])
                valid = r.get('valid', True) is not False
                if valid and int(r['chosen_idx_graph']) != chosen:
                    raise ValueError('Executed actor differs from frozen command')
                key = (sid, run, chosen)
                if key in library:
                    raise ValueError('A physical command was executed more than once')
                reference = assignments[sid]['reference_choice']
                reach8 = valid and bool(r.get('success_8m', False))
                reach4 = valid and bool(r.get('success_4m', False))
                correct = chosen == reference
                row = dict(scene_id=sid, run_id=run, chosen_idx=chosen,
                           intended_correct=correct, valid=valid,
                           semantic_8m=correct and reach8, semantic_4m=correct and reach4,
                           selected_8m=reach8, wrong_reach_8m=not correct and reach8,
                           collision=r.get('collision') if valid else None,
                           dynamic_actor_collision=r.get('dynamic_actor_collision') if valid else None,
                           static_only_collision=r.get('static_only_collision') if valid else None,
                           lane_invasion=r.get('lane_invasion') if valid else None,
                           source_file=path.name)
                library[key] = row
                count += 1
            if count != len(choices)*3 or count != metadata['emitted_episodes']:
                raise ValueError('Physical denominator is incomplete')
        if len(library) != manifest['expected_unique_physical_rollouts']:
            raise ValueError('Registered execution library is incomplete')
        methodrows = {}
        for method in METHODS + ['oracle_rule_reference']:
            selected = []
            for sid, a in assignments.items():
                chosen = a['reference_choice'] if method == 'oracle_rule_reference' else a['method_choices'][method]
                for run in [1, 2, 3]:
                    row = library[sid, run, chosen].copy()
                    row.update(track=track, method=method)
                    selected.append(row)
                    csvrows.append(row)
            by_scene = {sid: [r for r in selected if r['scene_id']==sid] for sid in assignments}
            metrics = {}
            for metric in METRICS:
                values = [sum(r[metric] for r in group)/3. for group in by_scene.values()]
                metrics[metric] = dict(count=sum(r[metric] for r in selected), n=len(selected),
                                       rate=float(np.mean(values)), cluster_ci95=ci(values))
            safety = {}
            for flag in ['collision', 'dynamic_actor_collision', 'static_only_collision', 'lane_invasion']:
                available = [r[flag] for r in selected if r[flag] is not None]
                safety[flag] = {'count':sum(available), 'logged_valid_n':len(available),
                                'rate_among_logged_valid':float(np.mean(available)) if available else None,
                                'not_a_safety_gain_test':True}
            methodrows[method] = dict(n=180, groups=60,
                intended_proto_correct=sum(r['intended_correct'] for r in selected), metrics=metrics,
                safety_diagnostic=safety,
                selected_rows=selected, cluster_means={sid:{m:sum(r[m] for r in g)/3. for m in METRICS}
                                                     for sid,g in by_scene.items()})
        graph = methodrows['graph_relation_gate']['cluster_means']
        comparisons = {}
        for method in METHODS[1:]:
            other = methodrows[method]['cluster_means']
            differences = [graph[sid]['semantic_8m'] - other[sid]['semantic_8m'] for sid in assignments]
            comparisons[method] = dict(delta_pp=100*float(np.mean(differences)),
                cluster_ci95_pp=[100*x for x in ci(differences)],
                cluster_sign_p=sign_p(differences), groups=60)
        adjusted = holm({m:c['cluster_sign_p'] for m,c in comparisons.items()})
        for m in comparisons:
            comparisons[m]['cluster_sign_holm_p']=adjusted[m]
        for m in methodrows:
            methodrows[m].pop('selected_rows')
            methodrows[m].pop('cluster_means')
        all_summary[track] = dict(unique_physical_rollouts=len(library),
            methods=methodrows, graph_vs_baselines=comparisons, source_sha256=hashes,
            shared_outcomes_not_independent_physical_runs=True,
            statistical_unit='60 pre-registered base graph clusters; three rollout seeds averaged',
            no_failed_condition_retry=True)
    out = root/'04_results/closed_loop'
    (out/'all_results.json').write_text(json.dumps(all_summary,indent=2),encoding='utf-8')
    with (out/'paired_selector_outcomes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines=['# R3 shared-controller execution audit','',
           'Each identical scene, seed and selected actor is physically executed once. Reused outcomes are paired observations, not additional runs.', '',
           '| Track | Method | Planned ProtoAcc | Sem8 count/N | Sem8 % [cluster CI] | Sem4 % | Wrong reach8 % | Valid |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for track,s in all_summary.items():
        for method,r in s['methods'].items():
            a=r['metrics']['semantic_8m']; lo,hi=a['cluster_ci95']
            lines.append(f'|{track}|{method}|{r["intended_proto_correct"]}/180|{a["count"]}/180|{a["rate"]*100:.1f} [{lo*100:.1f},{hi*100:.1f}]|{r["metrics"]["semantic_4m"]["rate"]*100:.1f}|{r["metrics"]["wrong_reach_8m"]["rate"]*100:.1f}|{r["metrics"]["valid"]["count"]}/180|')
        lines.append(f'\n{track}: {s["unique_physical_rollouts"]} unique command execution attempts (including actor-association rejections); 180 paired observations per method; 60 independent groups.\n')
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({t:s['unique_physical_rollouts'] for t,s in all_summary.items()}))


if __name__=='__main__':
    main()
