"""P14 prediction (label-blind models; labels stored only for scoring). Runs only after all training finished (P14_FINAL=1).
Models: baseline frozen Town10HD checkpoints (B) and P14 variants a/b/c (whichever are complete), 4 selectors x seeds 7/21/42.
Sets: train/val present+deleted (gate fitting / threshold calibration only), Town10 test present+deleted (both tracks),
Town10 Hard test rotated 90/180/270, Town05 Hard test present+deleted, nuScenes P11 layouts present+deleted (Hard checkpoints),
Talk2Car P13 keyword parses present+deleted (route as P13). Output preds_p14_<set>.jsonl, then PREDICTION_HASHES.json."""
import sys,json,re
from p14_common import *
assert os.environ.get('P14_FINAL')=='1','P14 TEST GUARD: run via run_p14.ps1 after training'
OUT=H/'preds';OUT.mkdir(exist_ok=True)
def models(track):
    L=[('B',m,s,base_ckpt(track,m,s)) for m in M for s in S3]
    for v in VARIANTS:
        if all((run_dir(v,track,m,s)/'summary.json').exists() for m in M for s in S3):L+=[(v,m,s,run_dir(v,track,m,s)/'checkpoint.pt') for m in M for s in S3]
    return L
def bundle(v,path):
    if v=='B':
        from r3_selectors.pipeline import load_checkpoint
        b=load_checkpoint(path,'cuda');return {'model':b['model'],'norm':b['normalization'],'null':False}
    return load_any(path)
def items_split(track,split):
    I=[]
    for r in manifest(track,split):
        npz=r.get('npz_path',r.get('npz_path_absolute'));g=load_g(npz);t=int(r['target_index'])
        I.append(dict(id=r['sample_id'],cluster=r['group_id'],cond='present',label=t,g=g));I.append(dict(id=r['sample_id'],cluster=r['group_id'],cond='deleted',label=None,g=delete(g,t)))
    return I
def items_rot(track='hard_confusable'):
    I=[]
    for r in manifest(track,'test'):
        npz=r.get('npz_path',r.get('npz_path_absolute'));g=load_g(npz)
        for a in(90,180,270):I.append(dict(id=r['sample_id'],cluster=r['group_id'],cond=f'rot{a}',label=int(r['target_index']),g=rot(g,math.radians(a))))
    return I
def items_town05():
    I=[]
    for p in [l.strip() for l in open(E/'P6_town05_hard/03_trainable/test.txt',encoding='utf-8') if l.strip()]:
        g=load_g(p);t=npz_target(p);sid='town05_hard:test:'+Path(p).stem.split('_')[-1]
        I.append(dict(id=sid,cluster=sid,cond='present',label=t,g=g));I.append(dict(id=sid,cluster=sid,cond='deleted',label=None,g=delete(g,t)))
    return I
def items_nus():
    P11=E/'P11_nuscenes_zeroshot';meta={m['id']:m for m in json.load(open(P11/'scenes_meta.json'))};lab={json.loads(l)['id']:json.loads(l)['target_index'] for l in open(P11/'labels.jsonl')};I=[]
    for f in sorted((P11/'03_trainable').glob('scen_*.npz')):
        i=int(f.stem.split('_')[1]);g=load_g(f);t=int(lab[i])
        I.append(dict(id=f'nuscenes:{f.stem}',cluster=meta[i]['scene'],cond='present',label=t,g=g));I.append(dict(id=f'nuscenes:{f.stem}',cluster=meta[i]['scene'],cond='deleted',label=None,g=delete(g,t)))
    return I
def items_t2c():
    sys.path.insert(0,str(E/'P13_talk2car_selectors'));import p13_common as C
    from p12e_parse_kw import parse
    assert sha(C.P12E/'preds_t2c_parse.jsonl')==(C.P12E/'PREDICTION_HASH_p12e.txt').read_text().strip()
    I=[]
    for l in open(C.P12E/'t2c_sets.jsonl',encoding='utf-8'):
        r=json.loads(l);tr,g=C.graph(r['cands'],parse(r['command']))
        g['node_spawn_indices']=np.arange(len(g['node_feats']));g['edge_index']=g['edge_index'].astype(np.int64);g['node_feats']=g['node_feats'].astype(np.float32);g['edge_attr']=g['edge_attr'].astype(np.float32)
        d,_=T.delete_node(g,r['target']+1)
        I.append(dict(id=r['id'],cluster=r['scene'],cond='present',label=r['target']+1,g=g,route=tr));I.append(dict(id=r['id'],cluster=r['scene'],cond='deleted',label=None,g=d,route=tr))
    return I
def run(name,I,track_of):
    fn=OUT/f'preds_p14_{name}.jsonl'
    if fn.exists() and (OUT/f'{name}.done').exists():print('skip',name,flush=True);return
    recs=[dict(id=x['id'],cluster=x['cluster'],cond=x['cond'],label=x['label'],route=x.get('route'),ncand=int(x['g']['candidate_mask'].sum()),out={}) for x in I]
    for track in sorted({track_of(x) for x in I}):
        idx=[i for i,x in enumerate(I) if track_of(x)==track and I[i]['g']['candidate_mask'].any()];G=[I[i]['g'] for i in idx]
        for v,m,s,p in models(track):
            P=predict_many(bundle(v,p),G)
            for i,o in zip(idx,P):recs[i]['out'][f'{v}|{m}@{s}']=o
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
    h={f.name:sha(f) for f in sorted(OUT.glob('preds_p14_*.jsonl'))}
    (H/'PREDICTION_HASHES.json').write_text(json.dumps(h,indent=1));print('HASHED',len(h))
if __name__=='__main__':main()
