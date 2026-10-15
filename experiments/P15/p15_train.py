"""P15 training. Usage: p15_train.py ARM TRACK [--smoke] [--device cpu|cuda]
ARM in M1 (ego encoding), M12 (+existence head, 3 negative types), FULL (+K-range augmentation, mean aggregation in Graph,
VAL split by scene group into selection/calibration halves; calibration half never used in training or selection).
Hyperparameters identical to the frozen Town10HD selectors / P14: 80 epochs, batch 64, AdamW lr 3e-4 wd 1e-5, cosine to 3e-5,
clip 5, rank weight 0.2 margin 1.0, seeds 7/21/42, same hidden sizes. Existence BCE weight 1.0.
Normalization: recomputed on ego-encoded TRAIN originals (the encoding changes 8 feature columns), stored in the checkpoint.
TEST is refused. --smoke: CPU, 1 epoch, seed 7, 48 TRAIN rows for fitting and 32 other TRAIN rows as pseudo-val (VAL not read).
Per-epoch resume state in last.pt; a run is complete only when summary.json exists."""
import sys,time,random,json
from p15_common import *
from torch.nn import functional as F
arm,track=sys.argv[1],sys.argv[2];SMOKE='--smoke' in sys.argv
DEV=sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else ('cpu' if SMOKE else 'cuda')
assert arm in ARMS and track in TRACKS;C=CFG[arm];FINE=fine_of(track)
def guard(split):
    if split=='test':raise PermissionError('P15 TRAIN GUARD: test split refused')
def rows_for(split):
    if SMOKE:
        rows=manifest(track,'train');return rows[:48] if split=='train' else rows[48:80]
    guard(split);rows=manifest(track,split)
    if split=='val' and C['valsplit']:rows=[r for r in rows if not in_cal(r['group_id'])]
    return rows
def build(split):
    rows=rows_for(split);S=[];neg=[];st=dict(rows=len(rows),present=0,kaug_up=0,kaug_down=0,kaug_discarded=0,kaug_label_changed=0,deleted=0,cf_query=0,near_miss=0)
    for r in rows:
        npz=r.get('npz_path',r.get('npz_path_absolute'));g=load_g(npz);t=int(r['target_index']);assert t==npz_target(npz)
        items=[(g,t)]
        if C['kaug'] and split=='train':
            a=k_augment(g,t,sseed('20261014','P15k',track,split,r['sample_id']),FINE)
            if a is None:st['kaug_discarded']+=1
            else:
                items.append((a[0],a[1]));st['kaug_'+a[2]]+=1;st['kaug_label_changed']+=int(a[3])
        for k,(x,tt) in enumerate(items):
            S.append(to_sample(ego_encode(x),tt));neg.append(False);st['present']+=1
            if C['exist']:
                for y,kind in negatives(x,tt,sseed('20261014','P15n',track,split,r['sample_id'],k)):
                    S.append(to_sample(ego_encode(y),int(np.flatnonzero(y['candidate_mask'])[0])));neg.append(True);st[kind]+=1
    return S,np.array(neg),st
def tensors(S,neg,norm):
    t=assemble_batch(S,norm,DEV);t['isneg']=torch.as_tensor(neg,device=DEV);return t
KEYS=['nodes','edges','query','candidates','valid','edge_mask']
def loss_fn(lg,b):
    n=b['candidates'].shape[1];pr=~b['isneg']
    if pr.any():
        node=lg[pr,:n];cand=b['candidates'][pr];tg=b['targets'][pr];ce=F.cross_entropy(node.masked_fill(~cand,-1e9),tg)
        tl=node.gather(1,tg[:,None]);ng=cand.clone();ng.scatter_(1,tg[:,None],False);rank=((F.relu(1.0+node-tl)*ng).sum(1)/ng.sum(1).clamp_min(1)).mean()
    else:ce=rank=lg.sum()*0
    loss=ce+0.2*rank
    if C['exist']:loss=loss+1.0*F.binary_cross_entropy_with_logits(lg[:,n],pr.float())
    return loss
def evaluate(model,t,bs=64):
    model.eval();N=len(t['targets']);cor=0;npres=0;tp=0;tn=0;nneg=0;tot=0.0
    with torch.no_grad():
        for st in range(0,N,bs):
            b={k:x[st:st+bs] for k,x in t.items()};lg=model(**{k:b[k] for k in KEYS});n=b['candidates'].shape[1];pr=~b['isneg']
            tot+=float(loss_fn(lg,b))*len(b['targets'])
            pred=lg[:,:n].masked_fill(~b['candidates'],-1e9).argmax(-1);cor+=int(((pred==b['targets'])&pr).sum());npres+=int(pr.sum())
            if C['exist']:
                e=lg[:,n]>0;tp+=int((e&pr).sum());tn+=int((~e&~pr).sum());nneg+=int((~pr).sum())
    acc=cor/max(npres,1)
    if C['exist']:
        bacc=0.5*(tp/max(npres,1)+tn/max(nneg,1));return 0.5*(acc+bacc),acc,bacc,tot/N
    return acc,acc,None,tot/N
def main():
    torch.set_num_threads(4)
    if DEV=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    norm_rows=manifest(track,'train')[:48] if SMOKE else manifest(track,'train')
    norm=norm_stats([ego_encode(load_g(r.get('npz_path',r.get('npz_path_absolute')))) for r in norm_rows])
    tr,trn,trs=build('train');va,van,vas=build('val');T_tr=tensors(tr,trn,norm);T_va=tensors(va,van,norm)
    epochs=int(os.environ.get('P15_SMOKE_EPOCHS','1')) if SMOKE else 80;seeds=[7] if SMOKE else S3;SM=H/'smoke'/os.environ.get('P15_SMOKE_TAG','base')
    print(f'P15 {arm} {track} dev={DEV} train={trs} val={vas}',flush=True)
    for m in M:
        for seed in seeds:
            out=(SM/arm/track/m/f'seed_{seed}') if SMOKE else run_dir(arm,track,m,seed);out.mkdir(parents=True,exist_ok=True)
            if (out/'summary.json').exists():print('SKIP COMPLETE',out,flush=True);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if DEV=='cuda':torch.cuda.manual_seed_all(seed)
            model=make_model(m,arm).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=3e-5);gen=torch.Generator().manual_seed(seed+1000)
            best=(-1.0,float('-inf'));be=0;start=1;t0=time.perf_counter();resumed=False
            if (out/'last.pt').exists():
                s=torch.load(out/'last.pt',map_location=DEV,weights_only=False)
                model.load_state_dict(s['model']);opt.load_state_dict(s['opt']);sch.load_state_dict(s['sch']);gen.set_state(s['gen'])
                torch.set_rng_state(s['cpu_rng'])
                if DEV=='cuda' and s.get('cuda_rng') is not None:torch.cuda.set_rng_state(s['cuda_rng'])
                best=tuple(s['best']);be=s['be'];start=s['epoch']+1;resumed=True;print(f'RESUME {arm} {track} {m} s{seed} from epoch {start}',flush=True)
            hist=open(out/'history.jsonl','a')
            for ep in range(start,epochs+1):
                model.train();order=torch.randperm(len(tr),generator=gen).to(DEV);ls=[]
                for st in range(0,len(order),64):
                    ids=order[st:st+64];b={k:x[ids] for k,x in T_tr.items()};opt.zero_grad(set_to_none=True)
                    loss=loss_fn(model(**{k:b[k] for k in KEYS}),b)
                    if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();ls.append(float(loss.detach()))
                selv,acc,bacc,vl=evaluate(model,T_va);sel=(selv,-vl)
                if sel>best:
                    best=sel;be=ep
                    torch.save({'model_name':m,'arm':arm,'state_dict':model.state_dict(),'normalization':norm,'best_epoch':ep,'val_sel':selv,'val_acc':acc,'val_exist_bacc':bacc,'val_loss':vl,'track':track,'seed':seed},out/'checkpoint.pt')
                hist.write(json.dumps(dict(epoch=ep,loss=float(np.mean(ls)),val_sel=selv,val_acc=acc,val_exist_bacc=bacc,val_loss=vl,best=be))+'\n');hist.flush();sch.step()
                torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'sch':sch.state_dict(),'gen':gen.get_state(),'cpu_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state() if DEV=='cuda' else None,'best':list(best),'be':be,'epoch':ep},out/'last.pt.tmp')
                os.replace(out/'last.pt.tmp',out/'last.pt')
                if SMOKE and os.environ.get('P15_SMOKE_CRASH_AFTER')==str(ep):raise SystemExit('SIMULATED CRASH (smoke only)')
                if ep==1 or ep%10==0:print(f'P15 {arm} {track} {m} s{seed} ep{ep} sel={100*selv:.2f} acc={100*acc:.2f} bacc={"-" if bacc is None else f"{100*bacc:.2f}"} best={be} t={time.perf_counter()-t0:.0f}s',flush=True)
            hist.close()
            json.dump(dict(arm=arm,track=track,model=m,seed=seed,best_epoch=be,val_selection=list(best),train_counts=trs,val_counts=vas,resumed=resumed,
                           seconds=time.perf_counter()-t0,checkpoint_sha256=sha(out/'checkpoint.pt'),test_read=False,status='complete'),open(out/'summary.json','w'),indent=1)
            print(f'COMPLETE P15 {arm} {track} {m} s{seed} {time.perf_counter()-t0:.0f}s',flush=True)
if __name__=='__main__':main()