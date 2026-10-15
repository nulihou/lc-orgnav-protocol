"""Check original graph equivalence, padding and node permutation on TRAIN inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .models import ModelConfig, build_model
from .pipeline import ManifestDataset, assemble_batch, forward_logits, train_normalization


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(7)
    ds = ManifestDataset(args.root / '02_protocol/scale300/train.jsonl')
    norm = train_normalization(ds)
    samples = [ds.samples[0], ds.samples[100], ds.samples[2000]]
    output = {}
    for name, h in [('graph_relation_gate',128), ('matched_geometry_mlp',160),
                    ('query_geometry_attention',96), ('drivetransformer_task_adapted',112)]:
        m = build_model(ModelConfig(name,h)).eval()
        b = assemble_batch(samples,norm)
        with torch.no_grad():
            batch_logits = forward_logits(m,b)
            individual = []
            for i,s in enumerate(samples):
                single = forward_logits(m,assemble_batch([s],norm))[0]
                individual.append(float((batch_logits[i,:len(single)]-single).abs().max()))
            s = samples[-1]; perm = np.random.default_rng(7).permutation(len(s['nodes']))
            inverse = np.argsort(perm)
            shuffled = dict(s); shuffled['nodes']=s['nodes'][perm]
            shuffled['candidates']=s['candidates'][perm]
            shuffled['edge_index']=inverse[s['edge_index']]
            shuffled['target']=int(inverse[s['target']])
            clean = forward_logits(m,assemble_batch([s],norm))[0]
            shuffled_logits = forward_logits(m,assemble_batch([shuffled],norm))[0]
            permutation_delta = float((clean-shuffled_logits[inverse]).abs().max())
            result = {'padding_max_abs': max(individual), 'permutation_max_abs': permutation_delta,
                      'parameters': sum(p.numel() for p in m.parameters())}
            if name == 'graph_relation_gate':
                sb=assemble_batch([s],norm); src,dst=s['edge_index']
                original=m.backbone.forward_single(sb['nodes'][0],torch.tensor(s['edge_index']),
                    sb['edges'][0,src,dst],sb['query'][0],sb['candidates'][0])['goal_logits']
                result['original_graph_max_abs']=float((original-clean).abs().max())
            if max(result[k] for k in result if k.endswith('_abs')) > 1e-4:
                raise AssertionError((name,result))
            output[name]=result
    dest=args.root/'07_audit/selector_implementation_checks.json'
    dest.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
