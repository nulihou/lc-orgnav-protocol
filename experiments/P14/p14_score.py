"""P14 scoring. Asserts PREDICTION_HASHES.json first. Gate/threshold fitting uses train/val prediction files only.
Operating points: 'argmax' (null models: abstain iff null is the argmax; others never abstain), 'cal' (null models: abstain iff
1-P(null) < t, t chosen on VAL present with the P9 rule for FRR<=2%), 'G3noK' (P12a head: logistic regression on
[maxp, margin, entropy, toplogit, logit_margin] fit on TRAIN present vs deleted, threshold on VAL by the same rule).
Empty candidate set = abstain. Item values are averaged over seeds 7/21/42; scene/group-cluster bootstrap 2000, seed 20261014;
paired differences variant - baseline; Holm within each family (16 contrasts). Preset criterion (P9): CAR>=90% at FRR<=2% on both
Town10 tracks (primary op 'cal' for null models; 'G3noK' for others)."""
import json
from p14_common import *
from sklearn.linear_model import LogisticRegression
P=H/'preds';reg=json.load(open(H/'PREDICTION_HASHES.json'))
for k,h in reg.items():assert sha(P/k)==h,k
L=lambda n:[json.loads(l) for l in open(P/f'preds_p14_{n}.jsonl')]
F5=['maxp','margin','entropy','toplogit','logit_margin']
def pick(sp,tg=0.02):
    sp=np.sort(np.asarray(sp,float));k=int(np.floor(tg*len(sp)));return float(sp[k]) if k<len(sp) else float('inf')
TH={}
for track in TRACKS:
    tr=L(f't10_{track}_train');va=L(f't10_{track}_val');keys=[k for k in va[0]['out']]
    for k in keys:
        X=np.array([[r['out'][k][f] for f in F5] for r in tr if k in r['out']]);y=np.array([r['cond']=='present' for r in tr if k in r['out']],int)
        c=LogisticRegression(C=1.0,max_iter=1000).fit(X,y);vp=[r for r in va if r['cond']=='present' and k in r['out']]
        t=pick(c.predict_proba(np.array([[r['out'][k][f] for f in F5] for r in vp]))[:,1])
        e=dict(coef=c.coef_[0].tolist(),intercept=float(c.intercept_[0]),thr=t)
        if va[0]['out'][k]['null_p'] is not None:e['cal_thr']=pick([1-r['out'][k]['null_p'] for r in vp])
        TH[f'{track}|{k}']=e
json.dump(TH,open(H/'p14_thresholds.json','w'),indent=1)
def abst(r,k,op,track):
    if k not in r['out']:return True
    o=r['out'][k];e=TH[f'{track}|{k}']
    if op=='argmax':return bool(o['argnull'])
    if op=='cal':return (1-o['null_p'])<e['cal_thr'] if o['null_p'] is not None else False
    z=float(np.dot(e['coef'],[o[f] for f in F5])+e['intercept']);return 1/(1+np.exp(-z))<e['thr']
rng=np.random.default_rng(20261014)
def cl(R):
    u=sorted({r['cluster'] for r in R});ix={c:i for i,c in enumerate(u)};return np.array([ix[r['cluster']] for r in R]),len(u)
BI={}
def boot(v,R):
    g,n=cl(R);key=(len(R),n,id(R))
    if key not in BI:BI[key]=rng.integers(0,n,(2000,n))
    v=np.asarray(v,float);s=np.bincount(g,v,n);c=np.bincount(g,None,n);b=BI[key];bs=s[b].sum(1)/np.maximum(c[b].sum(1),1)
    return [round(100*v.mean(),2),round(100*np.percentile(bs,2.5),2),round(100*np.percentile(bs,97.5),2)],bs
def pdiff(a,b,R):
    d=np.asarray(a,float)-np.asarray(b,float);pt,bs=boot(d,R);p=2*min((bs<=0).mean(),(bs>=0).mean())
    return dict(diff=pt[0],ci=pt[1:],p_boot=float(max(p,1/2000)))
def trk(name,r):return r['route'] if name=='t2c_kw' else ('scale300' if 'scale300' in name else 'hard_confusable')
def vals(name,R,v,m,metric,op=None):
    """seed-mean item values. metric: acc (forced node accuracy on present; empty set = wrong), abst (abstention indicator)."""
    out=[]
    for r in R:
        xs=[]
        for s in S3:
            k=f'{v}|{m}@{s}'
            if metric=='acc':xs.append(float(k in r['out'] and r['out'][k]['pred']==r['label']))
            elif metric=='ret':xs.append(float(k in r['out'] and (not abst(r,k,op,trk(name,r))) and r['out'][k]['pred']==r['label']))
            else:xs.append(float(abst(r,k,op,trk(name,r))))
        out.append(np.mean(xs))
    return out
D={n:L(n) for n in['t10_scale300_test','t10_hard_confusable_test','t10_hard_rot','town05','nus','t2c_kw']}
avail=['B']+[v for v in VARIANTS if any(k.startswith(v+'|') for k in D['t10_hard_confusable_test'][0]['out'])]
OPS=lambda v:['argmax','cal','G3noK'] if v in('a','c') else ['G3noK']
res={'variants_available':avail,'desc':{},'families':{}}
for n,R in D.items():
    Pr=[r for r in R if r['cond'] in('present',) or r['cond'].startswith('rot')];Dl=[r for r in R if r['cond']=='deleted']
    for v in avail:
        for m in M:
            e={}
            if n=='t10_hard_rot':
                for a in(90,180,270):X=[r for r in R if r['cond']==f'rot{a}'];e[f'acc_rot{a}']=boot(vals(n,X,v,m,'acc'),X)[0]
                e['acc_rot_pooled']=boot(vals(n,R,v,m,'acc'),R)[0]
            else:
                e['acc']=boot(vals(n,Pr,v,m,'acc'),Pr)[0]
                for op in OPS(v):
                    e[f'CAR_{op}']=boot(vals(n,Dl,v,m,'abst',op),Dl)[0];e[f'FRR_{op}']=boot(vals(n,Pr,v,m,'abst',op),Pr)[0]
                    e[f'retained_acc_{op}']=boot(vals(n,Pr,v,m,'ret',op),Pr)[0]
            res['desc'][f'{n}|{v}|{m}']=e
def crit(v,m):
    op='cal' if v in('a','c') else 'G3noK';ok=True
    for t in TRACKS:
        e=res['desc'][f't10_{t}_test|{v}|{m}'];ok=ok and e[f'CAR_{op}'][0]>=90 and e[f'FRR_{op}'][0]<=2
    return dict(op=op,meets_CAR90_FRR2_both_tracks=bool(ok))
res['criterion']={f'{v}|{m}':crit(v,m) for v in avail for m in M}
def fam(v,spec):
    C={}
    for m in M:
        for lab,n,metric,opv,opb in spec:
            R=D[n];X=[r for r in R if (r['cond']=='deleted')==(metric=='abst')]
            C[f'{m}|{lab}']=pdiff(vals(n,X,v,m,metric,opv),vals(n,X,'B',m,metric,opb),X)
    ps=sorted(C.items(),key=lambda kv:kv[1]['p_boot']);run=0
    for i,(k,c) in enumerate(ps):run=max(run,min(1.0,c['p_boot']*(len(ps)-i)));c['p_holm']=run;c['at_floor']=c['p_boot']<=1/2000
    return C
SA=[('CAR_hard','t10_hard_confusable_test','abst','cal','G3noK'),('CAR_scale','t10_scale300_test','abst','cal','G3noK'),('acc_hard','t10_hard_confusable_test','acc',None,None),('acc_scale','t10_scale300_test','acc',None,None)]
SB=[('acc_rot_pooled','t10_hard_rot','acc',None,None),('acc_town05','town05','acc',None,None),('acc_nus','nus','acc',None,None),('acc_hard','t10_hard_confusable_test','acc',None,None)]
SC=[SA[0],SA[1],SB[0],SB[2]]
for v,sp in(('a',SA),('b',SB),('c',SC)):
    if v in avail:res['families'][f'P14{v}']=fam(v,sp)
json.dump(res,open(H/'p14_results.json','w'),indent=1);print(json.dumps(res['criterion'],indent=1));print(json.dumps(res['families'],indent=1)[:4000])
