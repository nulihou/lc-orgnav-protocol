"""P15 scoring (run after P15_DONE.txt and P14_DONE.txt). Asserts P15 and P14 prediction hashes first.
Merges B and P14 a/b/c outputs (P14 preds) with P15 outputs by (set,id,cond). Threshold fitting uses TRAIN/VAL prediction files only.
Operating points (abstain iff score < threshold; empty candidate set = abstain):
  G3noK  : P12a logistic head on [maxp,margin,entropy,toplogit,logit_margin], fit TRAIN present vs deleted, P9 threshold on VAL present (all models).
  cal    : P14 a/c, score 1-P(null), P9 threshold on VAL present.
  exist  : P15 M12/FULL, score P(exist), P9 threshold on VAL present (FULL: calibration half only).
  ltt    : P15 M12/FULL, Learn-then-Test, FRR<=2% w.p.>=0.9, on VAL present (FULL: calibration half, disjoint from training/selection).
  crc    : conformal risk control E[FRR]<=2%, same calibration data as ltt.
Primary op: B,P14b,M1 -> G3noK; P14a/c -> cal; M12 -> exist; FULL -> ltt.
Preset criterion: CAR>=90% and FRR<=2% (point estimates, Town10HD test) on BOTH tracks at the primary op.
Statistics: seed-mean item values; group/scene-cluster bootstrap 2000, seed 20261014; Holm within each family of 16 contrasts;
TOST-style non-inferiority on clean forced accuracy vs B: one-sided 95% lower bound of paired difference > -delta (2pp primary, 1pp sensitivity)."""
import json
from p15_common import *
from sklearn.linear_model import LogisticRegression
# Env overrides exist ONLY for the smoke harness (fake files built from TRAIN graphs); the real run uses the defaults.
TEST_ROOT=os.environ.get('P15_SCORE_SMOKE_ROOT')
OUTD=Path(TEST_ROOT) if TEST_ROOT else H;P15P=OUTD/'preds15' if TEST_ROOT else H/'preds';P14P=OUTD/'preds14' if TEST_ROOT else P14/'preds'
r15=json.load(open(OUTD/'PREDICTION_HASHES_P15.json'));r14=json.load(open((OUTD if TEST_ROOT else P14)/'PREDICTION_HASHES.json'))
for k,h in r15.items():assert sha(P15P/k)==h,k
for k,h in r14.items():assert sha(P14P/k)==h,k
def L(n):
    A=[json.loads(l) for l in open(P15P/f'preds_p15_{n}.jsonl')];B={(r['id'],r['cond']):r for r in (json.loads(l) for l in open(P14P/f'preds_p14_{n}.jsonl'))}
    for r in A:
        b=B[(r['id'],r['cond'])];assert b['label']==r['label'] and b['ncand']==r['ncand'];r['out'].update(b['out'])
    assert len(A)==len(B);return A
F5=['maxp','margin','entropy','toplogit','logit_margin']
VARS=['B','a','b','c','M1','M12','FULL'];PRIMARY={'B':'G3noK','a':'cal','b':'G3noK','c':'cal','M1':'G3noK','M12':'exist','FULL':'ltt'}
TH={}
for track in TRACKS:
    tr=L(f't10_{track}_train');va=L(f't10_{track}_val');keys=sorted({k for r in va for k in r['out']})
    for k in keys:
        v=k.split('|')[0];X=np.array([[r['out'][k][f] for f in F5] for r in tr if k in r['out']]);y=np.array([r['cond']=='present' for r in tr if k in r['out']],int)
        c=LogisticRegression(C=1.0,max_iter=1000).fit(X,y);vp=[r for r in va if r['cond']=='present' and k in r['out']]
        e=dict(coef=c.coef_[0].tolist(),intercept=float(c.intercept_[0]),thr=p9_threshold(c.predict_proba(np.array([[r['out'][k][f] for f in F5] for r in vp]))[:,1]))
        if v in('a','c'):e['cal_thr']=p9_threshold([1-r['out'][k]['null_p'] for r in vp])
        if v in('M12','FULL'):
            cp=[r['out'][k]['pexist'] for r in vp if (v!='FULL' or in_cal(r['cluster']))];e['n_cal']=len(cp)
            e['exist_thr']=p9_threshold(cp);e['ltt_thr']=ltt_threshold(cp);e['crc_thr']=crc_threshold(cp)
        TH[f'{track}|{k}']=e
json.dump(TH,open(OUTD/'p15_thresholds.json','w'),indent=1)
def abst(r,k,op,track):
    if k not in r['out']:return True
    o=r['out'][k];e=TH[f'{track}|{k}']
    if op=='cal':return (1-o['null_p'])<e['cal_thr']
    if op in('exist','ltt','crc'):return o['pexist']<e[f'{op}_thr']
    z=float(np.dot(e['coef'],[o[f] for f in F5])+e['intercept']);return 1/(1+np.exp(-z))<e['thr']
rng=np.random.default_rng(20261014);BI={}
def cl(R):
    u=sorted({r['cluster'] for r in R});ix={c:i for i,c in enumerate(u)};return np.array([ix[r['cluster']] for r in R]),len(u)
def boot(v,R):
    g,n=cl(R);key=(len(R),n,tuple(sorted({r['cluster'] for r in R}))[:3],R[0]['id'],R[-1]['id'])
    if key not in BI:BI[key]=rng.integers(0,n,(2000,n))
    v=np.asarray(v,float);s=np.bincount(g,v,n);c=np.bincount(g,None,n);b=BI[key];bs=s[b].sum(1)/np.maximum(c[b].sum(1),1)
    return [round(100*v.mean(),2),round(100*np.percentile(bs,2.5),2),round(100*np.percentile(bs,97.5),2)],bs
def pdiff(a,b,R):
    d=np.asarray(a,float)-np.asarray(b,float);pt,bs=boot(d,R);p=2*min((bs<=0).mean(),(bs>=0).mean())
    return dict(diff=pt[0],ci=pt[1:],p_boot=float(max(p,1/2000)))
def trk(name,r):return r['route'] if name=='t2c_kw' else ('scale300' if 'scale300' in name else 'hard_confusable')
def vals(name,R,v,m,metric,op=None):
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
SETS=['t10_scale300_test','t10_hard_confusable_test','t10_hard_rot','town05','nus','t2c_kw'];D={n:L(n) for n in SETS}
avail=[v for v in VARS if any(k.startswith(v+'|') for r in D['t10_hard_confusable_test'][:2] for k in r['out'])]
def OPS(v):return {'B':['G3noK'],'b':['G3noK'],'M1':['G3noK'],'a':['cal','G3noK'],'c':['cal','G3noK'],'M12':['exist','ltt','crc','G3noK'],'FULL':['ltt','crc','exist','G3noK']}[v]
res={'variants_available':avail,'primary_op':PRIMARY,'desc':{},'kstrata':{},'weak_strata':{},'families':{},'tost':{},'sanity':{}}
STRATA={'K8':lambda s:s['K']==8,'rel3':lambda s:s['rel']==3,'rel4':lambda s:s['rel']==4,'farthest':lambda s:s['policy']==1,'adist_ge35':lambda s:(s['adist'] or 0)>=35}
for n,R in D.items():
    Pr=[r for r in R if r['cond']=='present' or r['cond'].startswith('rot')];Dl=[r for r in R if r['cond']=='deleted']
    for v in avail:
        for m in M:
            e={}
            if n=='t10_hard_rot':
                for a in(90,180,270):X=[r for r in R if r['cond']==f'rot{a}'];e[f'acc_rot{a}']=boot(vals(n,X,v,m,'acc'),X)[0]
                e['acc_rot_pooled']=boot(vals(n,R,v,m,'acc'),R)[0]
            else:
                e['acc']=boot(vals(n,Pr,v,m,'acc'),Pr)[0]
                for op in OPS(v):
                    e[f'CAR_{op}']=boot(vals(n,Dl,v,m,'abst',op),Dl)[0];e[f'FRR_{op}']=boot(vals(n,Pr,v,m,'abst',op),Pr)[0];e[f'retained_acc_{op}']=boot(vals(n,Pr,v,m,'ret',op),Pr)[0]
                op=PRIMARY[v];ks={}
                for K in sorted({r['st']['K'] for r in Pr}):
                    P_=[r for r in Pr if r['st']['K']==K];D_=[r for r in Dl if r['st']['K']==K]
                    ks[str(K)]=dict(n=len(P_),acc=round(100*np.mean(vals(n,P_,v,m,'acc')),2),FRR=round(100*np.mean(vals(n,P_,v,m,'abst',op)),2),CAR=round(100*np.mean(vals(n,D_,v,m,'abst',op)),2) if D_ else None)
                res['kstrata'][f'{n}|{v}|{m}|{op}']=ks
                if n in('town05','nus'):
                    ws={}
                    for sn,f in STRATA.items():
                        X=[r for r in Pr if f(r['st'])]
                        if X:ws[sn]=dict(n=len(X),acc=boot(vals(n,X,v,m,'acc'),X)[0])
                    res['weak_strata'][f'{n}|{v}|{m}']=ws
            res['desc'][f'{n}|{v}|{m}']=e
def crit(v,m):
    op=PRIMARY[v];ok=True
    for t in TRACKS:
        e=res['desc'][f't10_{t}_test|{v}|{m}'];ok=ok and e[f'CAR_{op}'][0]>=90 and e[f'FRR_{op}'][0]<=2
    return dict(op=op,meets_CAR90_FRR2_both_tracks=bool(ok))
res['criterion']={f'{v}|{m}':crit(v,m) for v in avail for m in M}
def fam(v,ref,spec):
    C={}
    for m in M:
        for lab,n,metric in spec:
            R=D[n];X=[r for r in R if (r['cond']=='deleted')==(metric in('CAR',))] if metric!='FRR' else [r for r in R if r['cond']=='present']
            if metric in('CAR','FRR'):a=vals(n,X,v,m,'abst',PRIMARY[v]);b=vals(n,X,ref,m,'abst',PRIMARY[ref])
            else:a=vals(n,X,v,m,'acc');b=vals(n,X,ref,m,'acc')
            C[f'{m}|{lab}']=pdiff(a,b,X)
    ps=sorted(C.items(),key=lambda kv:kv[1]['p_boot']);run=0
    for i,(k,c) in enumerate(ps):run=max(run,min(1.0,c['p_boot']*(len(ps)-i)));c['p_holm']=run;c['at_floor']=c['p_boot']<=1/2000
    return C
HT,ST='t10_hard_confusable_test','t10_scale300_test'
SPEC_PRIMARY=[('CAR_hard',HT,'CAR'),('CAR_scale',ST,'CAR'),('acc_rot_pooled','t10_hard_rot','acc'),('acc_nus','nus','acc')]
FAMS=[('F1_primary_FULL_vs_B','FULL','B',SPEC_PRIMARY),('F2_FULL_vs_P14c','FULL','c',SPEC_PRIMARY),
      ('F3_M1_vs_B','M1','B',[('acc_rot_pooled','t10_hard_rot','acc'),('acc_town05','town05','acc'),('acc_nus','nus','acc'),('acc_hard',HT,'acc')]),
      ('F4_M12_vs_P14a','M12','a',[('CAR_hard',HT,'CAR'),('CAR_scale',ST,'CAR'),('FRR_hard',HT,'FRR'),('FRR_scale',ST,'FRR')]),
      ('F5_FULL_vs_M12','FULL','M12',[('CAR_hard',HT,'CAR'),('CAR_scale',ST,'CAR'),('acc_nus','nus','acc'),('acc_town05','town05','acc')])]
for name,v,ref,sp in FAMS:
    if v in avail and ref in avail:res['families'][name]=fam(v,ref,sp)
for v in('M1','M12','FULL'):
    if v not in avail:continue
    for n in(HT,ST):
        X=[r for r in D[n] if r['cond']=='present']
        for m in M:
            d=np.asarray(vals(n,X,v,m,'acc'))-np.asarray(vals(n,X,'B',m,'acc'));pt,bs=boot(d,X);lo=100*float(np.percentile(bs,5))
            res['tost'][f'{n}|{v}|{m}']=dict(diff=pt[0],lower95_onesided=round(lo,2),noninferior_2pp=bool(lo>-2),noninferior_1pp=bool(lo>-1))
for v in('M1','M12','FULL'):
    if v not in avail:continue
    X=[r for r in D['t10_hard_rot']];same=0;tot=0
    base={(r['id']):r for r in D[HT] if r['cond']=='present'}
    for r in X:
        for m in M:
            for s in S3:
                k=f'{v}|{m}@{s}';tot+=1;same+=int(r['out'][k]['pred']==base[r['id']]['out'][k]['pred'])
    res['sanity'][f'rotation_prediction_identity_{v}']=dict(identical=same,total=tot)
json.dump(res,open(OUTD/'p15_results.json','w'),indent=1)
print(json.dumps(res['criterion'],indent=1));print(json.dumps(res['families'],indent=1)[:3000]);print(json.dumps(res['sanity'],indent=1))