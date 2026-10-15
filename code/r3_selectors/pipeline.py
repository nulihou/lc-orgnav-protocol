"""Train-only preprocessing, validation selection, single final test evaluation.

Run with the 03_code directory on PYTHONPATH. All outputs stay under R3.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .models import ModelConfig, MODEL_REGISTRY, build_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    here = Path(__file__).resolve().parent
    original = here.parent / 'lc_org_nav' / 'model.py'
    return {str(p): sha256(p) for p in [here / '__init__.py', here / 'models.py', here / 'pipeline.py', original]}


def read_array_sample(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        def arr(key):
            a = np.asarray(z[key])
            expected = {'node_feats': 2, 'edge_attr': 2, 'edge_index': 2,
                        'lang_feat': 1, 'candidate_mask': 1}[key]
            if a.ndim == expected + 1 and a.shape[0] == 1:
                a = a[0]
            return a
        return {
            'nodes': arr('node_feats').astype(np.float32),
            'edge_index': arr('edge_index').astype(np.int64),
            'edge_attr': arr('edge_attr').astype(np.float32),
            'query': arr('lang_feat').astype(np.float32),
            'candidates': arr('candidate_mask').astype(bool),
            'target': int(z['target_index'] if 'target_index' in z else z['target_idx_gt'])
        }


def assemble_batch(samples, norm=None, device='cpu'):
    n = max(len(s['nodes']) for s in samples)
    b = len(samples)
    arrays = {
        'nodes': np.zeros((b, n, 32), np.float32),
        'edges': np.zeros((b, n, n, 8), np.float32),
        'query': np.zeros((b, 32), np.float32),
        'candidates': np.zeros((b, n), bool),
        'valid': np.zeros((b, n), bool),
        'edge_mask': np.zeros((b, n, n), bool),
        'targets': np.zeros(b, np.int64),
    }
    for i, s in enumerate(samples):
        nn = len(s['nodes'])
        arrays['nodes'][i, :nn] = s['nodes']
        arrays['query'][i] = s['query']
        arrays['candidates'][i, :nn] = s['candidates']
        arrays['valid'][i, :nn] = True
        src, dst = s['edge_index']
        arrays['edges'][i, src, dst] = s['edge_attr']
        arrays['edge_mask'][i, src, dst] = True
        arrays['targets'][i] = s['target']
        if not arrays['candidates'][i, s['target']]:
            raise ValueError('Target is not a valid candidate: refusing denominator change')
    if norm:
        arrays['nodes'] = (arrays['nodes'] - np.asarray(norm['node_mean'], np.float32)) / np.asarray(norm['node_std'], np.float32)
        arrays['edges'] = (arrays['edges'] - np.asarray(norm['edge_mean'], np.float32)) / np.asarray(norm['edge_std'], np.float32)
        arrays['nodes'] *= arrays['valid'][..., None]
        arrays['edges'] *= arrays['edge_mask'][..., None]
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


class ManifestDataset:
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        self.samples = []
        for record in self.records:
            npz_path = record.get('npz_path', record.get('npz_path_absolute'))
            if record.get('source_npz_sha256') and sha256(npz_path) != record['source_npz_sha256']:
                raise ValueError('Frozen NPZ hash changed: refusing to train')
            s = read_array_sample(npz_path)
            if s['target'] != int(record['target_index']):
                raise ValueError('Manifest and NPZ labels differ')
            self.samples.append(s)
        if not self.samples:
            raise ValueError('Empty manifest')
        self.hash = sha256(self.manifest)
        self.tensors = None

    def prepare(self, norm, device):
        self.tensors = assemble_batch(self.samples, norm, device)
        return self

    def __len__(self):
        return len(self.samples)


def train_normalization(dataset):
    nodes = np.concatenate([s['nodes'] for s in dataset.samples], 0).astype(np.float64)
    edges = np.concatenate([s['edge_attr'] for s in dataset.samples], 0).astype(np.float64)
    def stats(x):
        mean, std = x.mean(0), x.std(0)
        std[std < 1e-6] = 1.0
        return mean.tolist(), std.tolist()
    nm, ns = stats(nodes); em, es = stats(edges)
    return {'node_mean': nm, 'node_std': ns, 'edge_mean': em, 'edge_std': es,
            'source': 'train manifest only', 'train_manifest_sha256': dataset.hash}


def forward_logits(model, batch):
    return model(**{k: batch[k] for k in ['nodes', 'edges', 'query', 'candidates', 'valid', 'edge_mask']})


def common_loss(logits, candidates, targets, rank_weight=0.2, margin=1.0):
    masked = logits.masked_fill(~candidates, -1e9)
    ce = F.cross_entropy(masked, targets)
    target = logits.gather(1, targets[:, None])
    negatives = candidates.clone()
    negatives.scatter_(1, targets[:, None], False)
    per_negative = F.relu(margin + logits - target) * negatives
    per_graph = per_negative.sum(1) / negatives.sum(1).clamp_min(1)
    rank = per_graph.mean()
    return ce + rank_weight * rank, ce, rank


def evaluate(model, dataset, batch_size):
    model.eval()
    correct = 0; total_ce = 0.0; all_logits = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = {k: v[start:start + batch_size] for k, v in dataset.tensors.items()}
            logits = forward_logits(model, batch)
            masked = logits.masked_fill(~batch['candidates'], -1e9)
            correct += int((masked.argmax(-1) == batch['targets']).sum().item())
            total_ce += float(F.cross_entropy(masked, batch['targets'], reduction='sum').item())
            all_logits.append(logits.cpu())
    return {'correct': correct, 'total': len(dataset), 'accuracy': correct / len(dataset),
            'ce': total_ce / len(dataset)}, torch.cat(all_logits, 0)


def write_predictions(path, dataset, logits, seed, model_name):
    with Path(path).open('w', encoding='utf-8') as out:
        for record, sample, row in zip(dataset.records, dataset.samples, logits):
            n = len(sample['nodes'])
            row = row[:n]
            masked = row.masked_fill(~torch.tensor(sample['candidates']), -1e9)
            probabilities = masked.softmax(-1)
            pred = int(masked.argmax().item())
            rec = {
                'sample_id': record['sample_id'],
                'scene_id': int(record.get('scene_id', record.get('original_scene_id', -1))),
                'pred_index': pred, 'chosen_idx': pred, 'target_index': sample['target'],
                'correct': pred == sample['target'], 'gs_correct': pred == sample['target'],
                'logits': row.tolist(), 'probabilities': probabilities.tolist(),
                'candidate_mask': sample['candidates'].tolist(),
                'canonical_group_id': record['group_id'], 'group_id': record['group_id'],
                'canonical_graph_sha256': record.get('canonical_graph_sha256'),
                'seed': seed, 'model_name': model_name, 'track': record['track'],
                'npz_path': record.get('npz_path', record.get('npz_path_absolute')),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')


def latency(model, dataset, repetitions=100):
    b = {k: v[:1] for k, v in dataset.tensors.items()}
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            forward_logits(model, b)
        if b['nodes'].is_cuda:
            torch.cuda.synchronize()
        durations = []
        for _ in range(repetitions):
            start = time.perf_counter()
            forward_logits(model, b)
            if b['nodes'].is_cuda:
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    return {'repetitions': repetitions, 'batch_size': 1, 'gpu_only_model_forward_ms':
            {'median': float(np.median(durations)), 'p95': float(np.percentile(durations, 95))},
            'excludes': 'NPZ loading, graph reconstruction, normalization, perception and controller'}


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = build_model(saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict']); model.eval()
    return {'model': model, 'normalization': saved['normalization'],
            'metadata': saved['metadata'], 'model_config': saved['model_config'],
            'device': str(device), 'checkpoint_path': str(path)}


def predict_arrays(bundle, node_feats, edge_index, edge_attr, lang_feat, candidate_mask):
    # No target is supplied to the model or used to compute a prediction.
    candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    dummy_target = int(np.flatnonzero(candidate_mask)[0])
    sample = {'nodes': np.asarray(node_feats, np.float32).reshape(-1, 32),
              'edge_index': np.asarray(edge_index, np.int64).reshape(2, -1),
              'edge_attr': np.asarray(edge_attr, np.float32).reshape(-1, 8),
              'query': np.asarray(lang_feat, np.float32).reshape(32),
              'candidates': candidate_mask, 'target': dummy_target}
    batch = assemble_batch([sample], bundle['normalization'], bundle['device'])
    with torch.no_grad():
        logits = forward_logits(bundle['model'], batch)[0]
        masked = logits.masked_fill(~batch['candidates'][0], -1e9)
    return {'pred_index': int(masked.argmax().item()), 'chosen_idx': int(masked.argmax().item()),
            'logits': logits.cpu().tolist(), 'probabilities': masked.softmax(-1).cpu().tolist()}


def predict_npz(bundle, path):
    s = read_array_sample(path)
    return predict_arrays(bundle, s['nodes'], s['edge_index'], s['edge_attr'], s['query'], s['candidates'])


def fit_one(cfg, seed, datasets, normalization, out_dir, args):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'summary.json').exists():
        print(f'SKIP COMPLETE {out_dir}', flush=True)
        return json.loads((out_dir / 'summary.json').read_text())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(cfg).to(args.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {'track': args.track, 'training_seed': seed, 'epochs': args.epochs,
                'batch_size': args.batch_size, 'learning_rate': args.lr,
                'weight_decay': 1e-5, 'rank_weight': 0.2, 'rank_margin': 1.0,
                'selection': 'validation accuracy, then lower validation CE; no test or closed-loop selection',
                'manifest_sha256': {k: v.hash for k, v in datasets.items()},
                'sample_counts': {k: len(v) for k, v in datasets.items()},
                'code_sha256': code_hashes(), 'trainable_parameters': params,
                'device': args.device, 'torch_version': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
                'test_read_for_selection': False,
                'protocol_sha256': sha256(args.root / '02_protocol' / 'BASELINE_PROTOCOL.md'),
                'source_npz_hashes_verified': True}
    (out_dir / 'run_config.json').write_text(json.dumps({'model_config': asdict(cfg), **metadata}, indent=2), encoding='utf-8')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    generator = torch.Generator().manual_seed(seed + 1000)
    history = []; best = (-1.0, float('-inf')); best_epoch = 0
    start_time = time.perf_counter()
    history_file = (out_dir / 'history.jsonl').open('w', encoding='utf-8')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter(); model.train()
        order = torch.randperm(len(datasets['train']), generator=generator).to(args.device)
        losses = []; ces = []; ranks = []
        for begin in range(0, len(order), args.batch_size):
            ids = order[begin:begin + args.batch_size]
            batch = {k: v[ids] for k, v in datasets['train'].tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, batch)
            loss, ce, rank = common_loss(logits, batch['candidates'], batch['targets'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss {cfg.name} {seed} {epoch}; fail retained')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item())); ces.append(float(ce.item())); ranks.append(float(rank.item()))
        val, _ = evaluate(model, datasets['val'], args.batch_size)
        selection = (val['accuracy'], -val['ce'])
        if selection > best:
            best = selection; best_epoch = epoch
            torch.save({'model_config': asdict(cfg), 'state_dict': model.state_dict(),
                        'normalization': normalization, 'metadata': metadata,
                        'best_epoch': epoch, 'validation': val}, out_dir / 'checkpoint.pt')
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
                'train_ce': float(np.mean(ces)), 'train_rank': float(np.mean(ranks)),
                'validation': val, 'seconds': time.perf_counter() - epoch_start,
                'learning_rate': optimizer.param_groups[0]['lr'], 'selected_best_epoch': best_epoch}
        history.append(row); history_file.write(json.dumps(row) + '\n'); history_file.flush()
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f'{args.track} {cfg.name} seed{seed} ep{epoch}/{args.epochs} val={100*val["accuracy"]:.2f}% bestep={best_epoch} sec={row["seconds"]:.2f}', flush=True)
    history_file.close()
    train_seconds = time.perf_counter() - start_time
    bundle = load_checkpoint(out_dir / 'checkpoint.pt', args.device)
    val, val_logits = evaluate(bundle['model'], datasets['val'], args.batch_size)
    test_dataset = datasets['train'] if args.smoke else datasets['test']
    test, test_logits = evaluate(bundle['model'], test_dataset, args.batch_size)
    write_predictions(out_dir / 'validation_predictions.jsonl', datasets['val'], val_logits, seed, cfg.name)
    write_predictions(out_dir / ('smoke_train_predictions.jsonl' if args.smoke else 'test_predictions.jsonl'), test_dataset, test_logits, seed, cfg.name)
    timing = latency(bundle['model'], test_dataset)
    summary = {'model_config': asdict(cfg), **metadata, 'best_epoch': best_epoch,
               'validation': val, 'test': test, 'train_seconds': train_seconds, 'timing': timing,
               'checkpoint': str((out_dir / 'checkpoint.pt').resolve()),
               'checkpoint_sha256': sha256(out_dir / 'checkpoint.pt'),
               'status': 'complete', 'test_evaluations_for_selection': 0,
               'test_evaluations_after_training': 0 if args.smoke else 1,
               'smoke_train_only': args.smoke}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    label = 'smoke_train' if args.smoke else 'test'
    print(f'COMPLETE {args.track} {cfg.name} seed{seed} {label}={test["correct"]}/{test["total"]} time={train_seconds:.1f}s', flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--track', choices=['scale300', 'hard_confusable'], required=True)
    p.add_argument('--models', nargs='+', default=list(MODEL_REGISTRY))
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 21, 42])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='One epoch, separate output; never enters formal results')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.smoke:
        args.epochs = 1; args.seeds = [7]
    datasets = {split: ManifestDataset(args.root / '02_protocol' / args.track / f'{split}.jsonl') for split in ['train', 'val', 'test']}
    groups = {k: {r['group_id'] for r in v.records} for k, v in datasets.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if groups[a] & groups[b]:
            raise ValueError(f'Group overlap {a}/{b}; refuse training')
    norm = train_normalization(datasets['train'])
    for dataset in datasets.values():
        dataset.prepare(norm, args.device)
    # Capacity chosen before test evaluation; this is deliberately not a tuning sweep.
    dimensions = {'graph_relation_gate': 128, 'matched_geometry_mlp': 160,
                  'query_geometry_attention': 96, 'drivetransformer_task_adapted': 112}
    output = args.root / '04_results' / ('smoke' if args.smoke else 'offline') / args.track
    output.mkdir(parents=True, exist_ok=True)
    (output / 'normalization.json').write_text(json.dumps(norm, indent=2), encoding='utf-8')
    for name in args.models:
        cfg = ModelConfig(name=name, hidden_dim=dimensions[name])
        for seed in args.seeds:
            dest = output / name / f'seed_{seed}'
            try:
                fit_one(cfg, seed, datasets, norm, dest, args)
            except Exception as exc:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / 'failure.json').write_text(json.dumps({'error': repr(exc), 'model': name, 'seed': seed, 'track': args.track}, indent=2), encoding='utf-8')
                raise


if __name__ == '__main__':
    main()
