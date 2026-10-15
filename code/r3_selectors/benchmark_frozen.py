"""Interleave frozen seed-42 model timings on identical static train inputs.

This is model-only timing, not a full driving-stack FPS/safety measurement.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .pipeline import ManifestDataset, assemble_batch, forward_logits, load_checkpoint
from .summarize_results import NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--repeats',type=int,default=200)
    args=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rows=[]
    for track in ['scale300','hard_confusable']:
        dataset=ManifestDataset(args.root/'02_protocol'/track/'train.jsonl')
        sample=max(dataset.samples,key=lambda s:len(s['nodes']))
        models={}
        for name in NAMES:
            path=args.root/'04_results/offline'/track/name/'seed_42/checkpoint.pt'
            bundle=load_checkpoint(path,args.device)
            batch=assemble_batch([sample],bundle['normalization'],args.device)
            models[name]=(bundle['model'],batch)
        durations={name:[] for name in NAMES}
        rng=np.random.default_rng(20261002)
        with torch.no_grad():
            for _ in range(20):
                for model,batch in models.values():forward_logits(model,batch)
            for _ in range(args.repeats):
                for name in rng.permutation(NAMES):
                    model,batch=models[name]
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                    start=time.perf_counter();forward_logits(model,batch)
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                    durations[name].append((time.perf_counter()-start)*1000)
        for name,values in durations.items():
            rows.append({'track':track,'model':name,'training_seed':42,'n_nodes':len(sample['nodes']),
                'batch_size':1,'repeats':args.repeats,'median_ms':float(np.median(values)),
                'p95_ms':float(np.quantile(values,.95)),
                'parameters':sum(p.numel() for p in models[name][0].parameters())})
    result={'rows':rows,'device':args.device,'torch_version':torch.__version__,'platform':platform.platform(),
        'gpu':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
        'timing_scope':'Normalized prepared tensor model forward only; excludes loading, graph reconstruction, preprocessing, sensing, tracking and control.',
        'ordering':'randomly interleaved four models under same static input shape, frozen seed42 checkpoints',
        'resource_exclusivity':'Not guaranteed; this shared workstation may run other authorized CARLA or analysis processes. Do not infer exclusive-device latency or stack FPS.'}
    dest=args.root/'04_results/offline'/f'common_{args.device.replace(":","_")}_latency.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
