"""Independent mean recount from packaged predictions; Python standard library only.
Run from any working directory. No checkpoint, GPU, original repository or network needed.
The original bootstrap intervals are retained in results.json, not recomputed here.
"""
from pathlib import Path
import json,hashlib,math,collections
H=Path(__file__).resolve().parent
th=json.loads((H/'p15_thresholds_frozen_copy.json').read_text())
results=json.loads((H/'results.json').read_text())
hashes=json.loads((H/'PREDICTION_HASHES.json').read_text())
primary={'B':'G3noK','a':'cal','b':'G3noK','c':'cal','M1':'G3noK','M12':'exist','FULL':'ltt'}
assert len(hashes)==168
acc=collections.defaultdict(lambda: [0,0,0,0])
for filename,digest in hashes.items():
 p=H/'predictions'/filename;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,filename
 track,arm,model,seed=p.stem.split('__');arm='b' if arm=='P14b' else arm;e=th[f'{track}|{arm}|{model}@{seed}'];op=primary[arm]
 for line in p.open():
  r=json.loads(line);o=r['out']
  if op=='cal':rejected=1-o['null_p']<e['cal_thr']
  elif op in ('exist','ltt'):rejected=o['pexist']<e[op+'_thr']
  else:
   z=sum(c*o[f] for c,f in zip(e['coef'],['maxp','margin','entropy','toplogit','logit_margin']))+e['intercept']
   rejected=1/(1+math.exp(-max(-700,min(700,z))))<e['thr']
  correct=o['pred']==r['target'];v=acc[f"{track}|{arm}|{model}|{r['condition']}"]
  v[0]+=1;v[1]+=rejected;v[2]+=correct;v[3]+=(correct and not rejected)
checked=0
for key,(n,rejected,correct,retained) in acc.items():
 ref=results['metrics'][key];assert n==ref['n_tasks']*3
 pairs=[(100*rejected/n,'rejection_ci95')]
 if not key.endswith('|deleted'):pairs.extend([(100*correct/n,'raw_accuracy_ci95'),(100*retained/n,'accepted_correct_ci95')])
 for value,field in pairs:
  assert abs(value-ref[field][0])<.000051,(key,field,value,ref[field]);checked+=1
print(json.dumps({'prediction_files_verified':len(hashes),'condition_rows':len(acc),'mean_metrics_checked':checked,'mismatches':0},indent=2))
