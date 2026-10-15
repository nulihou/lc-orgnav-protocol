"""P14 training. Variants: a = null output + deleted-target examples; b = rotation + distractor-subsampling augmentation;
c = a+b. Same architectures, hidden sizes, epochs (80), batch 64, AdamW lr 3e-4 wd 1e-5, cosine to 0.1x, clip 5, rank weight 0.2,
margin 1.0, seeds 7/21/42 as the frozen Town10HD models. Normalization = frozen baseline train statistics (train manifest only).
Uses TRAIN for fitting and VAL for checkpoint selection only; never reads TEST (guard).
Usage: p14_train.py VARIANT TRACK [--smoke]"""
import sys,time,random,json
from p14_common import *
import torch
from torch.nn import functional as F
from r3_selectors.pipeline import ManifestDataset,train_normalization
v,track=sys.argv[1],sys.argv[2];SMOKE='--smoke' in sys.argv
assert v in VARIANTS and track in TRACKS
NULL=v in('a','c');AUG=v in('b','c')
def guard(split):
    if split=='test':raise PermissionError('P14 TRAIN GUARD: test split refused')
def build(split):
    guard(split);rows=manifest(track,split)
    if SMOKE:rows=rows[:48]
    S=[];isnull=[];stats=dict(present=0,deleted=0,deleted_empty_skipped=0,aug=0)
    for r in rows:
        npz=r.get('npz_path',r.get('npz_path_absolute'));g=load_g(npz);t=int(r['target_index']);assert t==npz_target(npz)
        base=[(g,t)]
        if AUG:base+=[augment(g,t,sseed('20261014','P14b',track,split,r['sample_id'],c)) for c in range(AUG_COPIES)];stats['aug']+=AUG_COPIES
        for x,tt in base:
            S.append(to_sample(x,tt));isnull.append(False);stats['present']+=1
            if NULL:
                d=delete(x,tt)
                if not d['candidate_mask'].any():stats['deleted_empty_skipped']+=1;continue
                S.append(to_sample(d,int(np.flatnonzero(d['candidate_mask'])[0])));isnull.append(True);stats['deleted']+=1
    return S,np.array(isnull),stats
def tensors(S,isnull,norm):
    from r3_selectors.pipeline import assemble_batch
    t=assemble_batch(S,norm,'cuda');n=t['candidates'].shape[1]
    tg=t['targets'].clone();tg[torch.as_tensor(isnull,device='cuda')]=n;t['targets']=tg;t['isnull']=torch.as_tensor(isnull,device='cuda');return t
def loss_fn(lg,b):
    n=b['candidates'].shape[1]
    if NULL:
        full=torch.cat([lg[:,:n].masked_fill(~b['candidates'],-1e9),lg[:,n:n+1]],1);ce=F.cross_entropy(full,b['targets'])
        pr=~b['isnull']
        if pr.any():
            node=lg[pr,:n];cand=b['candidates'][pr];tg=b['targets'][pr]
            tl=node.gather(1,tg[:,None]);neg=cand.clone();neg.scatter_(1,tg[:,None],False)
            rank=((F.relu(1.0+node-tl)*neg).sum(1)/neg.sum(1).clamp_min(1)).mean()
        else:rank=lg.sum()*0
    else:
        node=lg.masked_fill(~b['candidates'],-1e9);ce=F.cross_entropy(node,b['targets'])
        tl=lg.gather(1,b['targets'][:,None]);neg=b['candidates'].clone();neg.scatter_(1,b['targets'][:,None],False)
        rank=((F.relu(1.0+lg-tl)*neg).sum(1)/neg.sum(1).clamp_min(1)).mean()
    return ce+0.2*rank,ce
def evaluate(model,t,bs=64):
    model.eval();cor=0;ce=0.0;N=len(t['targets'])
    with torch.no_grad():
        for st in range(0,N,bs):
            b={k:x[st:st+bs] for k,x in t.items()};lg=model(**{k:b[k] for k in['nodes','edges','query','candidates','valid','edge_mask']})
            n=b['candidates'].shape[1]
            full=torch.cat([lg[:,:n].masked_fill(~b['candidates'],-1e9),lg[:,n:n+1]],1) if NULL else lg.masked_fill(~b['candidates'],-1e9)
            cor+=int((full.argmax(-1)==b['targets']).sum());ce+=float(F.cross_entropy(full,b['targets'],reduction='sum'))
    return cor/N,ce/N
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    norm=train_normalization(ManifestDataset(R3/'02_protocol'/track/'train.jsonl'))
    base_norm=torch.load(base_ckpt(track,M[0],7),map_location='cpu',weights_only=False)['normalization']
    assert np.allclose(norm['node_mean'],base_norm['node_mean']) and np.allclose(norm['edge_std'],base_norm['edge_std']),'normalization differs from frozen baseline'
    tr,trn,trs=build('train');va,van,vas=build('val');T_tr=tensors(tr,trn,norm);T_va=tensors(va,van,norm)
    epochs=1 if SMOKE else 80;seeds=[7] if SMOKE else S3
    print(f'P14{v} {track} train={trs} val={vas}',flush=True)
    for m in M:
        for seed in seeds:
            out=(H/'smoke'/v/track/m/f'seed_{seed}') if SMOKE else run_dir(v,track,m,seed);out.mkdir(parents=True,exist_ok=True)
            if (out/'summary.json').exists():print('SKIP COMPLETE',out,flush=True);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=make_model(m,NULL).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=3e-5);gen=torch.Generator().manual_seed(seed+1000)
            best=(-1.0,float('-inf'));be=0;t0=time.perf_counter();hist=open(out/'history.jsonl','w')
            for ep in range(1,epochs+1):
                model.train();order=torch.randperm(len(tr),generator=gen).cuda();ls=[]
                for st in range(0,len(order),64):
                    ids=order[st:st+64];b={k:x[ids] for k,x in T_tr.items()};opt.zero_grad(set_to_none=True)
                    lg=model(**{k:b[k] for k in['nodes','edges','query','candidates','valid','edge_mask']});loss,ce=loss_fn(lg,b)
                    if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();ls.append(float(loss))
                acc,vce=evaluate(model,T_va);sel=(acc,-vce)
                if sel>best:
                    best=sel;be=ep
                    torch.save({'model_name':m,'null':NULL,'variant':v,'state_dict':model.state_dict(),'normalization':norm,'best_epoch':ep,'val_acc':acc,'val_ce':vce,'track':track,'seed':seed},out/'checkpoint.pt')
                hist.write(json.dumps(dict(epoch=ep,loss=float(np.mean(ls)),val_acc=acc,val_ce=vce,best=be))+'\n');hist.flush();sch.step()
                if ep==1 or ep%10==0:print(f'P14{v} {track} {m} s{seed} ep{ep} val={100*acc:.2f} best={be} t={time.perf_counter()-t0:.0f}s',flush=True)
            hist.close()
            json.dump(dict(variant=v,track=track,model=m,seed=seed,best_epoch=be,val_selection=list(best),train_counts=trs,val_counts=vas,
                           seconds=time.perf_counter()-t0,checkpoint_sha256=sha(out/'checkpoint.pt'),test_read=False,status='complete'),open(out/'summary.json','w'),indent=1)
            print(f'COMPLETE P14{v} {track} {m} s{seed} {time.perf_counter()-t0:.0f}s',flush=True)
if __name__=='__main__':main()
