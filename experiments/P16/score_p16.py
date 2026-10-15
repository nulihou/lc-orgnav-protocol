from run_p16 import *
import collections
H=Path(__file__).resolve().parent
F=validate();hashes=json.loads((H/'PREDICTION_HASHES.json').read_text())
assert len(hashes)==168,len(hashes)
for p,h in hashes.items():assert sha(H/'predictions'/p)==h,p
TH=json.loads((E/'P15_structured_remedy/p15_thresholds.json').read_text())
def abst(o,t,a,m,s):
 e=TH[f'{t}|{a}|{m}@{s}'];op=PRIMARY[a]
 if op=='cal':return 1-o['null_p'] < e['cal_thr']
 if op in ['exist','ltt']:return o['pexist'] < e[op+'_thr']
 z=float(np.dot(e['coef'],[o[f] for f in ['maxp','margin','entropy','toplogit','logit_margin']])+e['intercept'])
 return 1/(1+np.exp(-np.clip(z,-700,700))) < e['thr']
res={'scope':'post hoc, fixed three-seed mean; group-bootstrap descriptive intervals','metrics':{},'parity':{},'control_aware_check':{},'rereso':{}}
rng=np.random.default_rng(20261015);cache={}
def ci(values,groups):
 u,ix=np.unique(groups,return_inverse=True);n=len(u);key=tuple(u)
 if key not in cache:cache[key]=rng.integers(0,n,(5000,n))
 sums=np.bincount(ix,weights=np.asarray(values,float),minlength=n);cnt=np.bincount(ix,minlength=n);b=cache[key]
 bs=sums[b].sum(1)/cnt[b].sum(1)
 return [round(100*np.mean(values),4),round(100*np.percentile(bs,2.5),4),round(100*np.percentile(bs,97.5),4)]
for t in TRACKS:
 old={}
 for prefix,folder in [('p14','P14_remedies'),('p15','P15_structured_remedy')]:
  for line in (E/folder/'preds'/f'preds_{prefix}_t10_{t}_test.jsonl').open():
   r=json.loads(line);old.setdefault((r['id'],r['cond']),{}).update(r['out'])
 for a in ARMS:
  for m in C.M:
   seeds={s:[json.loads(l) for l in (H/'predictions'/f'{t}__{a}__{m}__{s}.jsonl').open()] for s in C.S3}
   meta=seeds[7];par={'raw_identity_mismatches':0,'primary_decision_mismatches':0,'comparisons':0}
   rejected={};ranked={};retained={}
   for cond in ['present','deleted','matching','other']:
    ix=[i for i,r in enumerate(meta) if r['condition']==cond]
    if not ix:continue
    groups=[meta[i]['group'] for i in ix];reject=[];rank=[];ret=[];perseed={}
    for s,rows in seeds.items():
     rr=np.array([abst(rows[i]['out'],t,a,m,s) for i in ix],float)
     cc=np.array([rows[i]['out']['pred']==rows[i]['target'] for i in ix],float)
     reject.append(rr);rank.append(cc);ret.append((1-rr)*cc)
     perseed[s]={'rejection':100*rr.mean(),'rank_accuracy':100*cc.mean() if cond!='deleted' else None,'accepted_correct':100*((1-rr)*cc).mean() if cond!='deleted' else None}
     for i in ix:
      r=rows[i]
      if cond in ['present','deleted']:
       oo=old[(r['id'],cond)][f'{a}|{m}@{s}'];par['comparisons']+=1
       par['raw_identity_mismatches']+=int(oo['pred']!=r['out']['pred'])
       par['primary_decision_mismatches']+=int(abst(oo,t,a,m,s)!=abst(r['out'],t,a,m,s))
    rv=np.mean(reject,axis=0);cv=np.mean(rank,axis=0);av=np.mean(ret,axis=0)
    rejected[cond]=(rv,groups);ranked[cond]=cv;retained[cond]=av
    res['metrics'][f'{t}|{a}|{m}|{cond}']={'n_tasks':len(ix),'n_groups':len(set(groups)),'rejection_ci95':ci(rv,groups),'raw_accuracy_ci95':ci(cv,groups) if cond!='deleted' else None,'accepted_correct_ci95':ci(av,groups) if cond!='deleted' else None,'per_seed':perseed}
   if 'matching' in rejected:
    vd,gd=rejected['deleted'];vm,gm=rejected['matching'];assert gd==gm
    res['metrics'][f'{t}|{a}|{m}|paired_gap']={'CAR_minus_matching_FRR_ci95':ci(vd-vm,gd)}
   ri=[i for i,r in enumerate(meta) if r['condition']=='deleted' and r['rereso_target'] is not None]
   if ri:
    vv=np.mean([[seeds[s][i]['out']['pred']==meta[i]['rereso_target'] for i in ri] for s in C.S3],axis=0)
    res['rereso'][f'{t}|{a}|{m}']={'eligible':len(ri),'raw_rereso_accuracy_ci95':ci(vv,[meta[i]['group'] for i in ri])}
   res['parity'][f'{t}|{a}|{m}']=par
for a in ARMS:
 for m in C.M:
  orig=True;ctrl=True
  for t in TRACKS:
   car=res['metrics'][f'{t}|{a}|{m}|deleted']['rejection_ci95'][0]
   frr=res['metrics'][f'{t}|{a}|{m}|present']['rejection_ci95'][0]
   orig=orig and car>=90 and frr<=2
   ctrl=ctrl and car>=90
   for cond in ['present','matching','other']:
    key=f'{t}|{a}|{m}|{cond}'
    if key in res['metrics']:ctrl=ctrl and res['metrics'][key]['rejection_ci95'][0]<=2
  res['control_aware_check'][f'{a}|{m}']={'original_point_criterion':bool(orig),'additional_control_check':bool(ctrl)}
(H/'results.json').write_text(json.dumps(res,indent=2))
print('PARITY', {k:sum(v[k] for v in res['parity'].values()) for k in ['raw_identity_mismatches','primary_decision_mismatches','comparisons']})
print('CONTROL',json.dumps(res['control_aware_check'],indent=1))
for a in ARMS:
 m='graph_relation_gate'
 print(a,{cond:res['metrics'][f'hard_confusable|{a}|{m}|{cond}']['rejection_ci95'] for cond in ['present','deleted','matching','other']})
