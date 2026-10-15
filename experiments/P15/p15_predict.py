"""P15 prediction (label-blind models; labels and strata stored only for scoring). Runs only after all P15 training (P15_FINAL=1).
Models: P15 arms M1/M12/FULL (complete ones), 4 selectors x seeds 7/21/42. Baseline (B) and P14 a/b/c predictions are NOT
recomputed: they are read at scoring time from P14 preds (hash-checked against P14 PREDICTION_HASHES.json).
Item sets and ids are identical to p14_predict.py. Output preds/preds_p15_<set>.jsonl, then PREDICTION_HASHES_P15.json.
Strata per item (from the original present graph): K, rel (argmax q[0:5]), policy (argmax q[5:9] or -1), adist (target anchor distance)."""
import sys,json
from p15_common import *
assert os.environ.get('P15_FINAL')=='1','P15 TEST GUARD: run via run_p15.ps1 after training'
OUT=H/'preds';OUT.mkdir(exist_ok=True)
def strata(g,t):
    q=g['lang_feat'];return dict(K=int(g['candidate_mask'].sum()),rel=int(np.argmax(q[:5])),policy=int(np.argmax(q[5:9])) if q[5:9].any() else -1,
                                adist=float(g['node_feats'][t,7]) if t is not None else None)
def models(track):
    L=[]
    for a in ARMS:
        if all((run_dir(a,track,m,s)/'summary.json').exists() for m in M for s in S3):L+=[(a,m,s,run_dir(a,track,m,s)/'checkpoint.pt') for m in M for s in S3]
    return L
def items_split(track,split):
    I=[]
    for r in manifest(track,split):
        npz=r.get('npz_path',r.get('npz_path_absolute'));g=load_g(npz);t=int(r['target_index']);st=strata(g,t)
        I.append(dict(id=r['sample_id'],cluster=r['group_id'],cond='present',label=t,g=g,st=st));I.append(dict(id=r['sample_id'],cluster=r['group_id'],cond='deleted',label=None,g=delete(g,t),st=st))
    return I
def items_rot(track='hard_confusable'):
    I=[]
    for r in manifest(track,'test'):
        npz=r.get('npz_path',r.get('npz_path_absolute'));g=load_g(npz);t=int(r['target_index']);st=strata(g,t)
        for a in(90,180,270):I.append(dict(id=r['sample_id'],cluster=r['group_id'],cond=f'rot{a}',label=t,g=rot(g,math.radians(a)),st=st))
    return I
def items_town05():
    I=[]
    for p in [l.strip() for l in open(E/'P6_town05_hard/03_trainable/test.txt',encoding='utf-8') if l.strip()]:
        g=load_g(p);t=npz_target(p);sid='town05_hard:test:'+Path(p).stem.split('_')[-1];st=strata(g,t)
        I.append(dict(id=sid,cluster=sid,cond='present',label=t,g=g,st=st));I.append(dict(id=sid,cluster=sid,cond='deleted',label=None,g=delete(g,t),st=st))
    return I
def items_nus():
    P11=E/'P11_nuscenes_zeroshot';meta={m['id']:m for m in json.load(open(P11/'scenes_meta.json'))};lab={json.loads(l)['id']:json.loads(l)['target_index'] for l in open(P11/'labels.jsonl')};I=[]
    for f in sorted((P11/'03_trainable').glob('scen_*.npz')):
        i=int(f.stem.split('_')[1]);g=load_g(f);t=int(lab[i]);st=strata(g,t)
        I.append(dict(id=f'nuscenes:{f.stem}',cluster=meta[i]['scene'],cond='present',label=t,g=g,st=st));I.append(dict(id=f'nuscenes:{f.stem}',cluster=meta[i]['scene'],cond='deleted',label=None,g=delete(g,t),st=st))
    return I
def items_t2c():
    sys.path.insert(0,str(E/'P13_talk2car_selectors'));import p13_common as C
    from p12e_parse_kw import parse
    assert sha(C.P12E/'preds_t2c_parse.jsonl')==(C.P12E/'PREDICTION_HASH_p12e.txt').read_text().strip()
    I=[]
    for l in open(C.P12E/'t2c_sets.jsonl',encoding='utf-8'):
        r=json.loads(l);tr,g=C.graph(r['cands'],parse(r['command']))
        g['node_spawn_indices']=np.arange(len(g['node_feats']));g['edge_index']=g['edge_index'].astype(np.int64);g['node_feats']=g['node_feats'].astype(np.float32);g['edge_attr']=g['edge_attr'].astype(np.float32)
        d,_=T.delete_node(g,r['target']+1);st=strata(g,r['target']+1)
        I.append(dict(id=r['id'],cluster=r['scene'],cond='present',label=r['target']+1,g=g,route=tr,st=st));I.append(dict(id=r['id'],cluster=r['scene'],cond='deleted',label=None,g=d,route=tr,st=st))
    return I
def run(name,I,track_of):
    fn=OUT/f'preds_p15_{name}.jsonl'
    if fn.exists() and (OUT/f'{name}.done').exists():print('skip',name,flush=True);return
    recs=[dict(id=x['id'],cluster=x['cluster'],cond=x['cond'],label=x['label'],route=x.get('route'),ncand=int(x['g']['candidate_mask'].sum()),st=x['st'],out={}) for x in I]
    for track in sorted({track_of(x) for x in I}):
        idx=[i for i,x in enumerate(I) if track_of(x)==track and I[i]['g']['candidate_mask'].any()];G=[I[i]['g'] for i in idx]
        for a,m,s,p in models(track):
            P=predict_many(load_p15(p),G)
            for i,o in zip(idx,P):recs[i]['out'][f'{a}|{m}@{s}']=o
            torch.cuda.empty_cache()
        print('pred',name,track,len(idx),flush=True)
    with open(fn,'w') as f:
        for r in recs:f.write(json.dumps(r)+'\n')
    (OUT/f'{name}.done').write_text('ok')
def main():
    for track in TRACKS:
        for split in('train','val','test'):run(f't10_{track}_{split}',items_split(track,split),lambda x,t=track:t)
    H_=lambda x:'hard_confusable'
    run('t10_hard_rot',items_rot(),H_);run('town05',items_town05(),H_);run('nus',items_nus(),H_)
    run('t2c_kw',items_t2c(),lambda x:x['route'])
    h={f.name:sha(f) for f in sorted(OUT.glob('preds_p15_*.jsonl'))}
    (H/'PREDICTION_HASHES_P15.json').write_text(json.dumps(h,indent=1));print('HASHED',len(h))
if __name__=='__main__':main()