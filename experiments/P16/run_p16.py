import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,argparse,datetime,gc
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;R3=V.parent;WS=R3.parent;E=R3/'07_experiments_r3_20261013'
sys.path[:0]=[str(E/'P15_structured_remedy'),str(E/'P14_remedies'),str(R3/'03_code')]
import p14_common as C
import p15_common as D
import torch
torch.set_num_threads(4)
ARMS=['B','a','b','c','M1','M12','FULL'];TRACKS=C.TRACKS
PRIMARY={'B':'G3noK','a':'cal','b':'G3noK','c':'cal','M1':'G3noK','M12':'exist','FULL':'ltt'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ckpt(t,a,m,s):
 return C.base_ckpt(t,m,s) if a=='B' else (C.run_dir(a,t,m,s)/'checkpoint.pt' if a in ['a','b','c'] else D.run_dir(a,t,m,s)/'checkpoint.pt')
def resolve(r):
 p=Path(r['npz_path'])
 if not p.exists():p=WS/'_archive_20261015_project/backups_tmp_and_data_classification'/Path(r['npz_path_relative_workspace'])
 assert p.exists(),str(p)
 assert sha(p)==r['source_npz_sha256'],str(p)
 return p
def items(t):
 out=[];sources=[];unavailable=[]
 for r in C.manifest(t,'test'):
  p=resolve(r);sources.append({'id':r['sample_id'],'path':str(p),'sha256':sha(p)})
  g=C.load_g(p);target=int(r['target_index']);matches=set(map(int,C.T.relation_matches(g)));cand=list(map(int,np.flatnonzero(g['candidate_mask'])))
  assert target in matches
  conditions=[('present',None),('deleted',target)]
  for cond,pool in [('matching',[i for i in cand if i!=target and i in matches]),('other',[i for i in cand if i!=target and i not in matches])]:
   if pool:conditions.append((cond,pool[0]))
   else:unavailable.append({'id':r['sample_id'],'condition':cond})
  pair={}
  for cond,drop in conditions:
   x,mp=(g,np.arange(len(g['node_feats']))) if drop is None else C.T.delete_node(g,drop)
   rematches=C.T.relation_matches(x);rp=C.T.rule_prediction(x,fine=t=='hard_confusable')
   unique=bool(len(rematches) and (len(rematches)==1 or (rp['margin'] is not None and rp['margin']>1e-5)))
   meta=dict(id=r['sample_id'],group=r['group_id'],condition=cond,target=None if cond=='deleted' else int(mp[target]),K=int(x['candidate_mask'].sum()),nmatch=len(rematches),original_K=len(cand),original_target_spawn=int(g['node_spawn_indices'][target]),removed_spawn=None if drop is None else int(g['node_spawn_indices'][drop]),graph_sha256=C.T.graph_hash(x),rereso_target=rp['pred_index'] if unique else None)
   assert x['candidate_mask'].any()
   if cond!='deleted':assert x['candidate_mask'][meta['target']]
   pair[cond]=meta;out.append((meta,x))
  if 'matching' in pair:assert (pair['matching']['K'],pair['matching']['nmatch'])==(pair['deleted']['K'],pair['deleted']['nmatch'])
 return out,sources,unavailable
def freeze():
 assert not (H/'FREEZE.json').exists(),'Freeze exists; do not overwrite'
 files={str(p):sha(p) for p in H.glob('*.py')};files[str(H/'PROTOCOL.md')]=sha(H/'PROTOCOL.md')
 for p in [E/'P14_remedies/p14_common.py',E/'P15_structured_remedy/p15_common.py',R3/'03_code/r3_deployment/transforms.py',R3/'03_code/r3_selectors/models.py',R3/'03_code/r3_selectors/pipeline.py',R3/'03_code/lc_org_nav/model.py',E/'P15_structured_remedy/p15_thresholds.json']:
  files[str(p)]=sha(p)
 inventory={}
 for t in TRACKS:
  I,src,un=items(t);inventory[t]={'source':src,'unavailable':un,'conditions':{c:sum(x['condition']==c for x,g in I) for c in ['present','deleted','matching','other']}}
  (H/f'inputs_{t}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x,g in I))
  files[str(H/f'inputs_{t}.jsonl')]=sha(H/f'inputs_{t}.jsonl')
  for a in ARMS:
   for m in C.M:
    for s in C.S3:p=ckpt(t,a,m,s);files[str(p)]=sha(p)
 (H/'FREEZE.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'post hoc audit; freeze before P16 prediction','files':files,'inventory':inventory},indent=2))
 print({t:inventory[t]['conditions'] for t in TRACKS},flush=True)
def validate():
 f=json.loads((H/'FREEZE.json').read_text())
 for p,h in f['files'].items():assert sha(p)==h,p
 return f
def predict():
 validate();pred=H/'predictions';pred.mkdir(exist_ok=True)
 for t in TRACKS:
  I,_,_=items(t);meta=[x for x,g in I];graphs=[g for x,g in I]
  for a in ARMS:
   for m in C.M:
    for s in C.S3:
     fn=pred/f'{t}__{a}__{m}__{s}.jsonl'
     if fn.exists():continue
     path=ckpt(t,a,m,s)
     if a in ['M1','M12','FULL']:b=D.load_p15(path);out=D.predict_many(b,graphs,bs=64)
     else:b=C.load_any(path);out=C.predict_many(b,graphs,bs=64)
     assert len(out)==len(meta)
     fn.write_text(''.join(json.dumps(dict(x,out=o))+'\n' for x,o in zip(meta,out)))
     del b,out;gc.collect();torch.cuda.empty_cache()
     print(t,a,m,s,'done',flush=True)
 (H/'PREDICTION_HASHES.json').write_text(json.dumps({p.name:sha(p) for p in sorted(pred.glob('*.jsonl'))},indent=2))
 print('PREDICTIONS_HASHED',flush=True)
if __name__=='__main__':
 stage=argparse.ArgumentParser();stage.add_argument('stage',choices=['freeze','predict']);args=stage.parse_args()
 (freeze if args.stage=='freeze' else predict)()
