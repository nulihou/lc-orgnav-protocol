"""P15 smoke checks (CPU, TRAIN data only; never reads VAL or TEST). Writes dev/smoke_results.json.
1 encoding invariance under global rotation; 2 model-level prediction invariance (smoke checkpoints);
3 K-augmentation / negative statistics on full TRAIN; 4 LTT/CRC Monte-Carlo guarantee check;
5 fake end-to-end score run on files built from TRAIN graphs (code-path test only)."""
import os,sys,json,subprocess,random,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
from p15_common import *
R={}
rows={t:manifest(t,'train') for t in TRACKS}
mx=0.0;n=0
for t in TRACKS:
    for r in rows[t][:300]:
        g=load_g(r.get('npz_path',r.get('npz_path_absolute')));e=ego_encode(g)
        for a in(90,180,270,37):
            x=ego_encode(rot(g,math.radians(a)));mx=max(mx,float(np.abs(x['node_feats']-e['node_feats']).max()),float(np.abs(x['edge_attr']-e['edge_attr']).max()));n+=1
R['1_encoding_invariance']=dict(pairs=n,max_abs_diff=mx,passed=mx<1e-3)
same=tot=0;pd=0.0
for t in TRACKS:
    for arm in ARMS:
        for m in M:
            b=load_p15(H/'smoke/base2'/arm/t/m/'seed_7'/'checkpoint.pt','cpu')
            G=[load_g(r.get('npz_path',r.get('npz_path_absolute'))) for r in rows[t][:40]];Gr=[rot(g,math.radians(123)) for g in G]
            A=predict_many(b,G,'cpu');B=predict_many(b,Gr,'cpu')
            for x,y in zip(A,B):
                tot+=1;same+=int(x['pred']==y['pred']);pd=max(pd,abs(x['maxp']-y['maxp']),abs((x['pexist'] or 0)-(y['pexist'] or 0)))
R['2_model_invariance']=dict(pairs=tot,identical_pred=same,max_prob_diff=pd,passed=(same==tot and pd<1e-4))
st={}
for t in TRACKS:
    c=dict(rows=0,up=0,down=0,discarded=0,label_changed=0,deleted=0,cf_query=0,near_miss=0,K_aug={})
    for r in rows[t]:
        g=load_g(r.get('npz_path',r.get('npz_path_absolute')));tt=int(r['target_index']);c['rows']+=1
        a=k_augment(g,tt,sseed('20261014','P15k',t,'train',r['sample_id']),fine_of(t))
        if a is None:c['discarded']+=1
        else:
            c[a[2]]+=1;c['label_changed']+=int(a[3]);K=str(int(a[0]['candidate_mask'].sum()));c['K_aug'][K]=c['K_aug'].get(K,0)+1
        for y,kind in negatives(g,tt,sseed('20261014','P15n',t,'train',r['sample_id'],0)):c[kind]+=1
    st[t]=c
R['3_train_augmentation_stats']=st
rng=np.random.default_rng(1);viol_ltt=0;frr_crc=[];trials=400
for _ in range(trials):
    cal=rng.normal(size=400);lt=ltt_threshold(cal);cr=crc_threshold(cal);te=rng.normal(size=20000)
    viol_ltt+=(te<lt).mean()>0.02;frr_crc.append((te<cr).mean())
R['4_calibration_mc']=dict(trials=trials,n_cal=400,ltt_violation_rate=viol_ltt/trials,ltt_bound=0.1,crc_mean_FRR=float(np.mean(frr_crc)),crc_bound=0.02,
                          passed=bool(viol_ltt/trials<=0.13 and np.mean(frr_crc)<=0.022))
root=H/'dev'/'fake_e2e';(root/'preds15').mkdir(parents=True,exist_ok=True);(root/'preds14').mkdir(parents=True,exist_ok=True)
rs=random.Random(0)
def fake_out(o,v):
    o=dict(o)
    if v in('a','c'):o['null_p']=rs.random();o['argnull']=o['null_p']>0.5
    return o
names=[f't10_{t}_{s}' for t in TRACKS for s in('train','val','test')]+['t10_hard_rot','town05','nus','t2c_kw']
bund={(arm,t,m):load_p15(H/'smoke/base2'/arm/t/m/'seed_7'/'checkpoint.pt','cpu') for arm in ARMS for t in TRACKS for m in M}
h15={};h14={}
for nm in names:
    t='scale300' if 'scale300' in nm else 'hard_confusable';src=rows[t][100:140] if 'train' in nm else rows[t][140:170];recs15=[];recs14=[]
    for r in src:
        g=load_g(r.get('npz_path',r.get('npz_path_absolute')));tt=int(r['target_index']);q=g['lang_feat']
        stt=dict(K=int(g['candidate_mask'].sum()),rel=int(np.argmax(q[:5])),policy=int(np.argmax(q[5:9])) if q[5:9].any() else -1,adist=float(g['node_feats'][tt,7]))
        conds=[('rot90',g,tt),('rot180',g,tt),('rot270',g,tt)] if nm=='t10_hard_rot' else [('present',g,tt),('deleted',delete(g,tt),None)]
        for cond,x,lab in conds:
            base=dict(id=r['sample_id'],cluster=r['group_id'],cond=cond,label=lab,route=t if nm=='t2c_kw' else None,ncand=int(x['candidate_mask'].sum()),st=stt)
            o15={};o14={}
            for arm in ARMS:
                for m in M:
                    p=predict_many(bund[(arm,t,m)],[x],'cpu')[0]
                    for s in S3:o15[f'{arm}|{m}@{s}']=p
                    if arm=='M1':
                        for v in('B','a','b','c'):
                            for s in S3:o14[f'{v}|{m}@{s}']=fake_out(p,v)
            recs15.append(dict(base,out=o15));recs14.append(dict(base,out=o14))
    for recs,sub,hd,tag in((recs15,'preds15',h15,'15'),(recs14,'preds14',h14,'14')):
        fn=root/sub/f'preds_p{tag}_{nm}.jsonl'
        with open(fn,'w') as f:
            for r in recs:f.write(json.dumps(r)+'\n')
        hd[fn.name]=sha(fn)
(root/'PREDICTION_HASHES_P15.json').write_text(json.dumps(h15));(root/'PREDICTION_HASHES.json').write_text(json.dumps(h14))
env=dict(os.environ,P15_SCORE_SMOKE_ROOT=str(root))
p=subprocess.run([sys.executable,'p15_score.py'],cwd=str(H),env=env,capture_output=True,text=True)
ok=p.returncode==0 and (root/'p15_results.json').exists()
fam=list(json.load(open(root/'p15_results.json'))['families'].keys()) if ok else None
R['5_fake_end_to_end_score']=dict(returncode=p.returncode,families=fam,stderr_tail=p.stderr[-1500:],passed=bool(ok and fam and len(fam)==5),note='fake files from TRAIN graphs; values meaningless')
json.dump(R,open(H/'dev'/'smoke_results.json','w'),indent=1);print(json.dumps(R,indent=1))