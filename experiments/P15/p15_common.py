"""P15 shared helpers (structured remedy). New file; reuses P14/R3 code read-only (imports, no in-place edits).
M1 = ego-relative SE(2)-invariant encoding; M2 = separate existence head + 3 negative types; M3 = LTT/CRC calibration (score side);
M4 = K-range augmentation (up/down sampling of distractors, relabelled by the frozen label generator) + K-invariant (mean) aggregation in Graph.
Frozen before any P15 training (hash in P15_FREEZE_MANIFEST.json)."""
import os;os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
R3=Path('R3_revision_20261002');E=R3/'07_experiments_r3_20261013';H=E/'P15_structured_remedy';P14=E/'P14_remedies'
sys.path.insert(0,str(R3/'03_code'));sys.path.insert(0,str(P14))
from p14_common import M,S3,DIM,TRACKS,sha,sseed,manifest,load_g,npz_target,delete,rot,to_sample,base_ckpt
from r3_deployment import transforms as T
import torch
from torch import nn
from r3_selectors.models import ModelConfig,build_model,BatchedOriginalGraph
from r3_selectors.pipeline import assemble_batch,forward_logits
ARMS=('M1','M12','FULL')
CFG={'M1':dict(ego=True,exist=False,kaug=False,mean_agg=False,valsplit=False),
     'M12':dict(ego=True,exist=True,kaug=False,mean_agg=False,valsplit=False),
     'FULL':dict(ego=True,exist=True,kaug=True,mean_agg=True,valsplit=True)}
def run_dir(arm,track,m,s):return H/'runs'/arm/track/m/f'seed_{s}'
def in_cal(group_id):
    """FULL only: VAL is split by scene group into a selection half (False) and a calibration half (True)."""
    return int(hashlib.sha256(f'P15|valsplit|{group_id}'.encode()).hexdigest()[:8],16)%2==1
def fine_of(track):return track=='hard_confusable'
# ---------------- M1: ego-relative encoding ----------------
def _R(v,yaw):
    """Express world-axis vectors v (n,2) in a frame rotated by yaw (i.e. apply R(-yaw))."""
    c,s=math.cos(yaw),math.sin(yaw);v=np.asarray(v,np.float64);return np.column_stack([c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1]])
def ego_encode(g):
    """Replace the world-axis offsets (node cols 0:2 ego offset, 5:7 anchor offset; edge cols 0:2 delta, 3:5 direction)
    by the same quantities expressed in the ego heading frame; canonicalize zero-distance self-bearings. Other columns unchanged.
    Invariant to any global rotation/translation of the scene (the P12d rotation adds theta to all yaws)."""
    fr=T.recover_frame(g);yaw=fr['ego_yaw'];pos=g['node_feats'][:,0:2].astype(np.float64)
    nf=g['node_feats'].copy();nf[:,0:2]=_R(pos,yaw);nf[:,5:7]=_R(pos-pos[fr['anchor_index']],yaw)
    # Zero-distance rows (ego in cols 2:5, anchor in cols 7:10): the source schema stores bearing=atan2(0,0)-yaw=-yaw there,
    # i.e. the ABSOLUTE world heading leaks into the features. Replace by the fixed canonical direction (1,0).
    for dcol,bc in((2,3),(7,8)):
        z=nf[:,dcol]<1e-6;nf[z,bc]=1.0;nf[z,bc+1]=0.0
    src,dst=g['edge_index'];d=_R(pos[dst]-pos[src],yaw);ang=np.arctan2(d[:,1],d[:,0]);ea=g['edge_attr'].copy()
    ea[:,0:2]=d;ea[:,3]=np.cos(ang);ea[:,4]=np.sin(ang)
    out=dict(g);out['node_feats']=nf.astype(np.float32);out['edge_attr']=ea.astype(np.float32);return out
def norm_stats(graphs):
    nodes=np.concatenate([g['node_feats'] for g in graphs],0).astype(np.float64);edges=np.concatenate([g['edge_attr'] for g in graphs],0).astype(np.float64)
    def st(x):
        m,s=x.mean(0),x.std(0);s[s<1e-6]=1.0;return m.tolist(),s.tolist()
    nm,ns=st(nodes);em,es=st(edges);return {'node_mean':nm,'node_std':ns,'edge_mean':em,'edge_std':es,'source':'P15 ego-encoded TRAIN originals only'}
# ---------------- M4: K-range augmentation ----------------
def complete_edges(N):return np.array([(i,j) for i in range(N) for j in range(N) if i!=j],np.int64).T.reshape(2,-1)
def k_augment(g,t,seed,fine):
    """One copy per graph. p=0.5 'up': add u~U{1..4} synthetic distractors (attribute row copied from a random non-target candidate,
    anchor flag cleared) at ego-frame range U(5,60) m, bearing U(-pi,pi); else 'down': drop each non-target candidate (not ego/anchor)
    with prob q~U(0.2,0.6) (at least one if possible). Label = frozen label generator (rule_prediction, fine for Hard) on the new
    graph; copies whose generator margin <=1e-3 are discarded. Returns (graph,label,kind,label_changed) or None."""
    rng=np.random.default_rng(seed);fr=T.recover_frame(g);pos=g['node_feats'][:,:2].astype(np.float64)
    others=[int(i) for i in np.flatnonzero(g['candidate_mask']) if i not in(t,fr['ego_index'],fr['anchor_index'])]
    if rng.random()<0.5 or not others:
        u=int(rng.integers(1,5));rows=[];newp=[]
        for _ in range(u):
            dd=rng.uniform(5,60);b=rng.uniform(-math.pi,math.pi);w=_R(np.array([[dd*math.cos(b),dd*math.sin(b)]]),-fr['ego_yaw'])[0]
            src=int(rng.choice(others)) if others else int(t);row=g['node_feats'][src].copy();row[11]=0;rows.append(row);newp.append(w)
        x={k:v.copy() for k,v in g.items()};N=len(pos)+u
        x['node_feats']=np.vstack([g['node_feats'],np.array(rows,np.float32)]);x['candidate_mask']=np.concatenate([g['candidate_mask'],np.ones(u,bool)])
        x['node_spawn_indices']=np.concatenate([g['node_spawn_indices'],g['node_spawn_indices'].max()+1+np.arange(u)])
        x['edge_index']=complete_edges(N);x['edge_attr']=np.zeros((x['edge_index'].shape[1],8),np.float32)
        x=T.rebuild_geometry(x,np.vstack([pos,np.array(newp)]),fr);tt=t;kind='up'
    else:
        q=float(rng.uniform(0.2,0.6));drop=[i for i in others if rng.random()<q] or [int(rng.choice(others))];x=g;tt=t
        for i in sorted(drop,reverse=True):
            x,mp=T.delete_node(x,i);tt=int(mp[tt])
        kind='down'
    r=T.rule_prediction(x,fine=fine)
    if r['pred_index'] is None or (r['margin'] is not None and r['margin']<=1e-3):return None
    return x,int(r['pred_index']),kind,int(r['pred_index'])!=tt
# ---------------- M2: negatives ----------------
def negatives(g,t,seed):
    """(i) target deleted; (ii) counterfactual query: relation bit replaced by another relation (1..4) that NO candidate satisfies
    under the frozen relation definition (policy bits kept); (iii) near-miss: target deleted after moving the most target-like
    distractor (nearest to the target) half-way to the target position, kept only if that distractor still fails the query relation.
    All are target-absent graphs. Returns list of (graph,kind)."""
    out=[];rng=np.random.default_rng(seed);d=delete(g,t)
    if d['candidate_mask'].any():out.append((d,'deleted'))
    cur=int(np.argmax(g['lang_feat'][:5]))
    for r in rng.permutation([1,2,3,4]):
        r=int(r)
        if r!=cur and len(T.relation_matches(g,r))==0 and g['candidate_mask'].any():
            c=T.clone(g);c['lang_feat'][:5]=0;c['lang_feat'][r]=1;out.append((c,'cf_query'));break
    fr=T.recover_frame(g);pos=g['node_feats'][:,:2].astype(np.float64)
    others=[int(i) for i in np.flatnonzero(g['candidate_mask']) if i!=t]
    if others:
        j=min(others,key=lambda i:float(np.hypot(*(pos[i]-pos[t]))));P=pos.copy();P[j]=pos[j]+0.5*(pos[t]-pos[j])
        h=T.rebuild_geometry(g,P,fr);h,mp=T.delete_node(h,t);jj=int(mp[j])
        if jj not in set(T.relation_matches(h).tolist()):out.append((h,'near_miss'))
    return out
# ---------------- models ----------------
class MeanAggGraph(BatchedOriginalGraph):
    """M4: identical to BatchedOriginalGraph except incoming messages are averaged (not summed) over valid edges."""
    def forward(self,nodes,edges,query,candidates,valid,edge_mask):
        from torch.nn import functional as F
        m=self.backbone;h=m.node_encoder(nodes);e=m.edge_encoder(edges);q=m.lang_proj(query)
        cnt=edge_mask.sum(1).clamp_min(1)[...,None].to(h.dtype)
        for layer in range(m.num_layers):
            gn,bn,ge,be=m.film_layers[layer](q).chunk(4,-1);hn=h*gn[:,None]+bn[:,None];he=e*ge[:,None,None]+be[:,None,None]
            src=hn[:,:,None].expand(-1,-1,h.shape[1],-1);msg=m.msg_mlps[layer](torch.cat([src,he],-1))
            agg=(msg*edge_mask[...,None]).sum(1)/cnt;h=F.relu(m.self_mlps[layer](hn)+agg)*valid[...,None]
        r=m.rel_proj(query)[:,None].expand(-1,h.shape[1],-1);gate=m.gate_mlp(torch.cat([h,r],-1));h=gate*h+(1-gate)*r
        return m.goal_head(torch.cat([h,q[:,None].expand_as(h)],-1)).squeeze(-1)
class ExistWrapper(nn.Module):
    """M2: separate existence logit (sigmoid, trained with BCE) from (mean,max) pooling over candidates of the input to the
    selector's own final scoring head (which already contains the query). No candidate count is given as a feature.
    Output = [node logits (B,N), existence logit (B,1)]."""
    def __init__(self,base):
        super().__init__();self.base=base;head=base.backbone.goal_head if hasattr(base,'backbone') else base.score
        lin=[mm for mm in head.modules() if isinstance(mm,nn.Linear)][0];f=lin.in_features
        self.exist=nn.Sequential(nn.Linear(2*f,f),nn.ReLU(),nn.Linear(f,1));self._x=None
        head.register_forward_hook(lambda mod,inp,out:setattr(self,'_x',inp[0]))
    def forward(self,nodes,edges,query,candidates,valid,edge_mask):
        logits=self.base(nodes,edges,query,candidates,valid,edge_mask);x=self._x;c=candidates.unsqueeze(-1);cnt=c.sum(1).clamp_min(1)
        mean=(x*c).sum(1)/cnt;mx=x.masked_fill(~c,-1e9).max(1).values;mx=torch.where(candidates.any(1,keepdim=True),mx,torch.zeros_like(mx))
        return torch.cat([logits,self.exist(torch.cat([mean,mx],-1))],1)
def make_model(m,arm):
    c=CFG[arm];cfg=ModelConfig(name=m,hidden_dim=DIM[m])
    b=MeanAggGraph(cfg) if (c['mean_agg'] and m=='graph_relation_gate') else build_model(cfg)
    return ExistWrapper(b) if c['exist'] else b
def load_p15(path,device='cuda'):
    s=torch.load(path,map_location=device,weights_only=False);mdl=make_model(s['model_name'],s['arm']).to(device);mdl.load_state_dict(s['state_dict']);mdl.eval()
    return {'model':mdl,'norm':s['normalization'],'arm':s['arm'],'exist':CFG[s['arm']]['exist']}
def predict_many(bundle,graphs,device='cuda',bs=256):
    out=[]
    for st in range(0,len(graphs),bs):
        chunk=[ego_encode(g) for g in graphs[st:st+bs]];samples=[to_sample(g,int(np.flatnonzero(g['candidate_mask'])[0])) for g in chunk]
        b=assemble_batch(samples,bundle['norm'],device)
        with torch.no_grad():lg=forward_logits(bundle['model'],b)
        n=b['candidates'].shape[1];node=lg[:,:n].masked_fill(~b['candidates'],-1e9);pn=node.softmax(-1)
        pex=torch.sigmoid(lg[:,n]) if bundle['exist'] else None
        for i,g in enumerate(chunk):
            k=len(g['node_feats']);p=pn[i,:k].cpu().numpy();l=node[i,:k].cpu().numpy();m=g['candidate_mask'];ps=np.sort(p[m])[::-1];ls=np.sort(l[m])[::-1]
            out.append(dict(pred=int(np.argmax(np.where(m,l,-1e30))),maxp=float(ps[0]),margin=float(ps[0]-ps[1]) if len(ps)>1 else 1.0,
                entropy=float(-(p[m]*np.log(np.clip(p[m],1e-12,1))).sum()),toplogit=float(ls[0]),logit_margin=float(ls[0]-ls[1]) if len(ls)>1 else 10.0,
                ncand=int(m.sum()),null_p=None,argnull=False,pexist=float(pex[i]) if pex is not None else None))
    return out
# ---------------- M3: calibration ----------------
def ltt_threshold(present_scores,alpha=0.02,delta=0.1):
    """Learn-then-Test (fixed-sequence over increasing thresholds; exact binomial p-values). Abstain iff score < lam.
    Returns largest lam such that FRR(lam)<=alpha is certified w.p. >=1-delta (in-distribution, exchangeable calibration data).
    -inf if nothing is certified (never abstain)."""
    from scipy.stats import binom
    s=np.sort(np.asarray(present_scores,float));n=len(s);lam=-np.inf
    for l in np.unique(s):
        k=int((s<l).sum())
        if binom.cdf(k,n,alpha)<=delta:lam=float(l)
        else:break
    return lam
def crc_threshold(present_scores,alpha=0.02):
    """Conformal risk control for the (monotone, bounded by 1) false-reject loss: largest lam with (k(lam)+1)/(n+1)<=alpha.
    Guarantees E[FRR]<=alpha (marginal, in-distribution)."""
    s=np.sort(np.asarray(present_scores,float));n=len(s);lam=-np.inf
    for l in np.unique(s):
        if ((s<l).sum()+1)/(n+1)<=alpha:lam=float(l)
        else:break
    return lam
def p9_threshold(sp,tg=0.02):
    sp=np.sort(np.asarray(sp,float));k=int(np.floor(tg*len(sp)));return float(sp[k]) if k<len(sp) else float('inf')