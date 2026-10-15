"""Build outcome-blind unique-command execution tiers from fixed seed-42 scores."""
import argparse
import hashlib
import json
from pathlib import Path

METHODS = ['graph_relation_gate', 'matched_geometry_mlp',
           'query_geometry_attention', 'drivetransformer_task_adapted']


def read_lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--track', choices=['scale300', 'hard_confusable'], required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol = root / '02_protocol' / args.track / 'closed_loop'
    out = root / '04_results' / 'closed_loop' / args.track / 'execution_library'
    if out.exists():
        raise FileExistsError('Do not overwrite a registered execution library')
    scenarios = json.loads((protocol / 'scenarios_60.json').read_text())
    ids = {int(s['id']) for s in scenarios}
    reference_path = out.parent / 'frozen_reference_scores' / 'oracle_scores.jsonl'
    reference = {int(r['scene_id']): r for r in read_lines(reference_path)}
    models = {}
    hashes = {}
    for method in METHODS:
        path = root / '04_results' / 'offline' / args.track / method / 'seed_42' / 'test_predictions.jsonl'
        models[method] = {int(r['scene_id']): r for r in read_lines(path)}
        hashes[method] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not ids.issubset(models[method]):
            raise ValueError('Incomplete frozen predictions for ' + method)
    tiers = []
    assignments = []
    for s in scenarios:
        sid = int(s['id'])
        gt = int(reference[sid]['pred_index'])
        choices = {m: int(models[m][sid]['pred_index']) for m in METHODS}
        extra = sorted(set(choices.values()) - {gt})
        for slot, chosen in enumerate(extra):
            while len(tiers) <= slot:
                tiers.append([])
            # Scores encode an already chosen actor; no label is read by the selector replay.
            logits = [-1e9] * len(reference[sid]['logits'])
            logits[chosen] = 0.0
            tiers[slot].append((s, {'scene_id': sid, 'pred_index': chosen, 'logits': logits}))
        assignments.append({'scene_id': sid, 'reference_choice': gt,
                            'method_choices': choices, 'extra_choices': extra})
    out.mkdir(parents=True)
    for slot, rows in enumerate(tiers):
        (out / ('tier_%02d_scenarios.json' % slot)).write_text(
            json.dumps([s for s, _ in rows], indent=2), encoding='utf-8')
        (out / ('tier_%02d_scores.jsonl' % slot)).write_text(
            ''.join(json.dumps(r) + '\n' for _, r in rows), encoding='utf-8')
    manifest = {'track': args.track, 'model_seed': 42, 'model_predictions_sha256': hashes,
                'scenarios_sha256': hashlib.sha256((protocol / 'scenarios_60.json').read_bytes()).hexdigest(),
                'reference_commands': len(scenarios), 'additional_commands': sum(len(x) for x in tiers),
                'expected_unique_physical_rollouts': (len(scenarios) + sum(len(x) for x in tiers)) * 3,
                'paired_outcomes_per_method': len(scenarios) * 3,
                'tier_sizes': [len(x) for x in tiers], 'assignments': assignments,
                'no_execution_outcomes_read': True}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'assignments'}, indent=2))


if __name__ == '__main__':
    main()
