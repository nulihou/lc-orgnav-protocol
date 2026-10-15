"""Audit completed artifacts without changing models, scores or checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path


NAMES = ['graph_relation_gate', 'matched_geometry_mlp',
         'query_geometry_attention', 'drivetransformer_task_adapted']
SEEDS = [7, 21, 42]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line]


def audit(root):
    rows = []
    missing = []
    protocol_hash = digest(root / '02_protocol' / 'BASELINE_PROTOCOL.md')
    core_hash_sets = set()
    for track in ['scale300', 'hard_confusable']:
        manifest_paths = {s: root / '02_protocol' / track / f'{s}.jsonl' for s in ['train', 'val', 'test']}
        manifests = {s: jsonlines(p) for s, p in manifest_paths.items()}
        for name in NAMES:
            for seed in SEEDS:
                run = root / '04_results' / 'offline' / track / name / f'seed_{seed}'
                if not (run / 'summary.json').exists():
                    missing.append(str(run))
                    continue
                summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
                config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
                history = jsonlines(run / 'history.jsonl')
                errors = []
                if summary['status'] != 'complete' or summary['smoke_train_only']:
                    errors.append('run not a complete formal fit')
                if len(history) != 80 or [h['epoch'] for h in history] != list(range(1, 81)):
                    errors.append('not all 80 epochs retained')
                if any(not math.isfinite(h[k]) for h in history for k in ['train_loss', 'train_ce', 'train_rank']):
                    errors.append('nonfinite training history')
                selected = max(history, key=lambda h: (h['validation']['accuracy'], -h['validation']['ce']))
                if selected['epoch'] != summary['best_epoch'] or selected['epoch'] != history[-1]['selected_best_epoch']:
                    errors.append('selected epoch does not match validation-only rule')
                if summary['validation'] != selected['validation']:
                    errors.append('selected validation metrics differ')
                if summary['protocol_sha256'] != protocol_hash:
                    errors.append('registered protocol changed')
                if summary['manifest_sha256'] != {s: digest(p) for s, p in manifest_paths.items()}:
                    errors.append('manifest hash changed')
                if summary['sample_counts'] != {s: len(m) for s, m in manifests.items()}:
                    errors.append('manifest sample count changed')
                if summary['checkpoint_sha256'] != digest(run / 'checkpoint.pt'):
                    errors.append('checkpoint hash changed')
                if summary['test_evaluations_for_selection'] != 0 or summary['test_evaluations_after_training'] != 1:
                    errors.append('declared test use differs from protocol')
                if config['model_config']['name'] != name or config['training_seed'] != seed:
                    errors.append('run name/seed mismatch')
                for filename, expected in summary['code_sha256'].items():
                    if digest(filename) != expected:
                        errors.append('core code changed: ' + filename)
                core_hash_sets.add(tuple(sorted(summary['code_sha256'].items())))
                for split, filename in [('val', 'validation_predictions.jsonl'), ('test', 'test_predictions.jsonl')]:
                    preds = jsonlines(run / filename)
                    if [p['sample_id'] for p in preds] != [m['sample_id'] for m in manifests[split]]:
                        errors.append(split + ' task IDs/order differ from frozen manifest')
                    if len(preds) != summary['validation' if split == 'val' else 'test']['total']:
                        errors.append(split + ' prediction count mismatch')
                    for pred, record in zip(preds, manifests[split]):
                        if pred['target_index'] != record['target_index'] or pred['group_id'] != record['group_id']:
                            errors.append(split + ' target/group differs')
                            break
                        if pred['seed'] != seed or pred['model_name'] != name:
                            errors.append(split + ' model/seed differs')
                            break
                        if len(pred['logits']) != len(pred['candidate_mask']) or not pred['candidate_mask'][pred['pred_index']]:
                            errors.append(split + ' logits/index/candidate contract invalid')
                            break
                    if sum(int(p['correct']) for p in preds) != summary['validation' if split == 'val' else 'test']['correct']:
                        errors.append(split + ' aggregate correct count mismatch')
                rows.append({'track': track, 'model': name, 'seed': seed, 'passed': not errors,
                             'errors': errors, 'best_epoch': summary['best_epoch'],
                             'test_correct': summary['test']['correct'], 'test_n': summary['test']['total'],
                             'checkpoint_sha256': summary['checkpoint_sha256']})
    result = {'protocol_sha256': protocol_hash, 'completed': len(rows), 'expected': 24,
              'missing': missing, 'core_source_hash_variants': len(core_hash_sets), 'runs': rows,
              'passed': len(rows) == 24 and not missing and all(r['passed'] for r in rows) and len(core_hash_sets) == 1,
              'scope': 'Artifact and selection-rule audit; declaration counters are not an independent execution trace.'}
    dest = root / '07_audit' / 'baseline_training_freeze_audit.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['completed', 'expected', 'core_source_hash_variants', 'passed']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root)
