"""CPU checkpoint replay and test prediction integrity, without model selection."""
import argparse
import json
from pathlib import Path
import torch
from .pipeline import ManifestDataset, assemble_batch, forward_logits, load_checkpoint, sha256
from .summarize_results import NAMES,SEEDS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);checks=[]
    for track in ['scale300','hard_confusable']:
        dataset=ManifestDataset(args.root/'02_protocol'/track/'test.jsonl')
        for name in NAMES:
            for seed in SEEDS:
                run=args.root/'04_results/offline'/track/name/f'seed_{seed}'
                if not (run/'summary.json').exists():continue
                summary=json.loads((run/'summary.json').read_text())
                if sha256(run/'checkpoint.pt')!=summary['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                bundle=load_checkpoint(run/'checkpoint.pt','cpu')
                predictions=[json.loads(l) for l in (run/'test_predictions.jsonl').read_text().splitlines()]
                if len(predictions)!=len(dataset):raise ValueError('Denominator changed')
                errors=[];disagreements=[]
                with torch.no_grad():
                    for start in range(0,len(dataset),64):
                        samples=dataset.samples[start:start+64]
                        batch=assemble_batch(samples,bundle['normalization'],'cpu')
                        logits=forward_logits(bundle['model'],batch)
                        chosen=logits.masked_fill(~batch['candidates'],-1e9).argmax(-1)
                        for offset,s in enumerate(samples):
                            i=start+offset;r=predictions[i];m=dataset.records[i];n=len(s['nodes'])
                            if r['sample_id']!=m['sample_id'] or r['target_index']!=s['target'] or r['group_id']!=m['group_id']:
                                raise ValueError('ID, target or group changed')
                            expected=torch.tensor(r['logits'])
                            errors.append(float((logits[offset,:n]-expected).abs().max()))
                            if int(chosen[offset])!=r['pred_index']:disagreements.append(r['sample_id'])
                checks.append({'track':track,'model':name,'seed':seed,'n':len(dataset),
                    'max_cpu_gpu_logit_abs_delta':max(errors),'prediction_disagreements':disagreements,
                    'predictions_sha256':sha256(run/'test_predictions.jsonl'),
                    'checkpoint_sha256':sha256(run/'checkpoint.pt'),
                    'manifest_hash_matches':bundle['metadata']['manifest_sha256']['test']==dataset.hash})
                # Keep any CPU/GPU difference visible; CUDA frozen predictions
                # remain authoritative for the preregistered replay cohort.
    out=args.root/'07_audit/frozen_prediction_integrity.json'
    out.write_text(json.dumps({'checks':checks,'cuda_frozen_predictions_authoritative':True,
                              'no_model_or_checkpoint_selection':True},indent=2),encoding='utf-8')
    print(json.dumps({'runs_checked':len(checks),'disagreements':sum(len(r['prediction_disagreements']) for r in checks),'artifact':str(out)}))


if __name__=='__main__':main()
