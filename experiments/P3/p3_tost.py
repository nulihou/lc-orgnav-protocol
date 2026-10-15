"""P3 analysis (frozen protocol PROTOCOL.md). Run only after PREDICTION_HASHES.json lists every new test_predictions.jsonl."""
import json,hashlib
from pathlib import Path
import numpy as np
import os;REPO=Path(os.environ.get('LCORGNAV_R3_ROOT','R3_revision_20261002'));H=Path(__file__).resolve().parent
OLD=REPO/'04_results/offline';NEW=H/'root/04_results/offline'
M=['graph_relation_gate','matched_geometry_mlp','query_geometry_attention','drivetransformer_task_adapted']
SEEDS_OLD=[7,21,42];SEEDS_NEW=[1,3,5,11,13,17,19]
reg=json.load(open(H/'PREDICTION_HASHES.json'))
def preds(t,m,s):
    p=(OLD if s in SEEDS_OLD else NEW)/t/m/f'seed_{s}'/'test_predictions.jsonl'
    if s in SEEDS_NEW: assert hashlib.sha256(p.read_bytes()).hexdigest()==reg[f'{t}/{m}/seed_{s}'],p
    return {json.loads(l)['sample_id']:int(json.loads(l)['pred_index']) for l in open(p,encoding='utf-8') if l.strip()}
out={}
for t in('scale300','hard_confusable'):
    man=[json.loads(l) for l in open(REPO/'02_protocol'/t/'test_manifest.jsonl',encoding='utf-8-sig') if l.strip()]
    sid=[r['sample_id'] for r in man];ti=np.array([int(r['target_index']) for r in man]);gid=[r['group_id'] for r in man]
    u,inv=np.unique(gid,return_inverse=True)
    C={}
    for m in M:
        C[m]=np.array([[preds(t,m,s)[x]==ti[i] for i,x in enumerate(sid)] for s in SEEDS_OLD+SEEDS_NEW],float)  # seeds x samples
    res={m:dict(per_seed_acc=[round(100*a,2) for a in C[m].mean(1)],mean=round(100*C[m].mean(),2),sd=round(100*C[m].mean(1).std(ddof=1),2)) for m in M}
    rng=np.random.default_rng(20261002);idx=rng.integers(0,len(u),(10000,len(u)))
    def gm(v):  # group-mean of per-sample values
        s=np.bincount(inv,v);c=np.bincount(inv);return s/c,c
    from math import comb
    cmp={}
    for m in M[1:]:
        d=C['graph_relation_gate'].mean(0)-C[m].mean(0)  # per-sample seed-averaged diff
        g,c=gm(d);est=float((g*c).sum()/c.sum())
        bs=(g[idx]*c[idx]).sum(1)/c[idx].sum(1)
        ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))];ci90=[float(np.percentile(bs,5)),float(np.percentile(bs,95))]
        pos=int((g>0).sum());neg=int((g<0).sum());n=pos+neg;k=min(pos,neg)
        p=min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0
        cmp[m]=dict(diff_pp=round(100*est,2),ci95_pp=[round(100*x,2) for x in ci95],ci90_pp=[round(100*x,2) for x in ci90],
                    tost_equiv_2pp=bool(ci90[0]>-0.02 and ci90[1]<0.02),tost_equiv_1pp=bool(ci90[0]>-0.01 and ci90[1]<0.01),sign_p=p,groups_pos_neg=[pos,neg])
    ps=sorted(cmp,key=lambda m:cmp[m]['sign_p']);run=0
    for i,m in enumerate(ps): run=max(run,min(1.0,(len(ps)-i)*cmp[m]['sign_p']));cmp[m]['sign_p_holm']=run
    out[t]=dict(methods=res,graph_vs=cmp)
json.dump(out,open(H/'p3_results.json','w'),indent=1);print(json.dumps(out,indent=1))
