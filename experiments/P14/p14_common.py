"""P14 shared helpers: paths, graph I/O, deletion, augmentation, null-output wrapper, batched prediction.
Frozen before any P14 training or evaluation (hash in P14_FREEZE_MANIFEST.json)."""
import os;os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
R3=Path('R3_revision_20261002');E=R3/'07_experiments_r3_20261013';H=E/'P14_remedies'
sys.path.insert(0,str(R3/'03_code'))
from r3_deployment import transforms as T
M=['graph_relation_gate','matched_geometry_mlp','query_geometry_attention','drivetransformer_task_adapted'];S3=[7,21,42]
DIM={'graph_relation_gate':128,'matched_geometry_mlp':160,'query_geometry_attention':96,'drivetransformer_task_adapted':112}
TRACKS=('scale300','hard_confusable');VARIANTS=('a','b','c');AUG_COPIES=3
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sseed(*p):return int(hashlib.sha256('|'.join(map(str,p)).encode()).hexdigest()[:8],16)
def base_ckpt(track,m,s):return R3/'04_results/offline'/track/m/f'seed_{s}'/'checkpoint.pt'
def run_dir(v,track,m,s):return H/'runs'/v/track/m/f'seed_{s}'
def manifest(track,split):
    f=R3/'02_protocol'/track/f'{split}.jsonl'
    return [json.loads(l) for l in open(f,encoding='utf-8-sig') if l.strip()]
def load_g(npz):
    g=T.load_graph(npz);sq=lambda a:a[0] if a.ndim==3 else a
    for k in('node_feats','edge_index','edge_attr'):g[k]=sq(g[k])
    g['lang_feat']=g['lang_feat'].reshape(-1).astype(np.float32);g['candidate_mask']=g['candidate_mask'].reshape(-1).astype(bool)
    g['node_feats']=g['node_feats'].astype(np.float32);g['edge_attr']=g['edge_attr'].astype(np.float32);g['edge_index']=g['edge_index'].astype(np.int64)
    return g
def npz_target(npz):
    z=np.load(npz);return int(z['target_index'] if 'target_index' in z else z['target_idx_gt'])
def delete(g,ti):
    out,_=T.delete_node(g,ti);return out
def rot(g,theta):
    """Global rotation, identical to P12d.rot (relative geometry preserved)."""
    fr=T.recover_frame(g);pos=g['node_feats'][:,:2].astype(np.float64);c,s=math.cos(theta),math.sin(theta)
    pr=pos@np.array([[c,s],[-s,c]]);f=dict(fr);f['ego_yaw']=fr['ego_yaw']+theta;f['anchor_yaw']=fr['anchor_yaw']+theta
    return T.rebuild_geometry(g,pr,f)
def augment(g,ti,seed):
    """P14b: random global rotation U(0,2pi) always; with prob 0.5 drop each non-target candidate (not ego/anchor) with prob q~U(0.2,0.6).
    Label preserved: target kept; relation and extreme-order policies are monotone under removal of other candidates."""
    rng=np.random.default_rng(seed);fr=T.recover_frame(g);x=rot(g,float(rng.uniform(0,2*math.pi)));t=ti
    if rng.random()<0.5:
        q=float(rng.uniform(0.2,0.6))
        cand=[int(i) for i in np.flatnonzero(x['candidate_mask']) if i not in(ti,fr['ego_index'],fr['anchor_index'])]
        drop=sorted([i for i in cand if rng.random()<q],reverse=True)
        for i in drop:
            x,mp=T.delete_node(x,i);t=int(mp[t])
    return x,t
def to_sample(g,target):
    return {'nodes':g['node_feats'].astype(np.float32),'edge_index':g['edge_index'].astype(np.int64),'edge_attr':g['edge_attr'].astype(np.float32),
            'query':g['lang_feat'].astype(np.float32),'candidates':g['candidate_mask'].astype(bool),'target':int(target)}

# ---------------- null-output wrapper (P14a/c) ----------------
import torch
from torch import nn
from r3_selectors.models import ModelConfig,build_model
def final_head(model):
    return model.backbone.backbone.goal_head if hasattr(model,'backbone') and hasattr(model.backbone,'backbone') else (model.backbone.goal_head if hasattr(model,'backbone') else model.score)
class NullWrapper(nn.Module):
    """Adds one 'no target' logit computed from the selector's own final candidate representation
    (input of its final scoring head), pooled (mean,max) over candidate nodes; trained end to end."""
    def __init__(self,base):
        super().__init__();self.base=base;head=base.backbone.goal_head if hasattr(base,'backbone') else base.score
        lin=[m for m in head.modules() if isinstance(m,nn.Linear)][0];f=lin.in_features
        self.null=nn.Sequential(nn.Linear(2*f,f),nn.ReLU(),nn.Linear(f,1));self._x=None
        head.register_forward_hook(lambda mod,inp,out:setattr(self,'_x',inp[0]))
    def forward(self,nodes,edges,query,candidates,valid,edge_mask):
        logits=self.base(nodes,edges,query,candidates,valid,edge_mask);x=self._x
        c=candidates.unsqueeze(-1);cnt=c.sum(1).clamp_min(1)
        mean=(x*c).sum(1)/cnt;mx=x.masked_fill(~c,-1e9).max(1).values;mx=torch.where(candidates.any(1,keepdim=True),mx,torch.zeros_like(mx))
        return torch.cat([logits,self.null(torch.cat([mean,mx],-1))],1)
def make_model(m,null):
    b=build_model(ModelConfig(name=m,hidden_dim=DIM[m]));return NullWrapper(b) if null else b
def load_any(path,device='cuda'):
    s=torch.load(path,map_location=device,weights_only=False)
    null=bool(s.get('null',False));
    if 'model_config' in s and not null:
        b=build_model(s['model_config']).to(device);b.load_state_dict(s['state_dict'])
    else:
        b=make_model(s['model_name'],null).to(device);b.load_state_dict(s['state_dict'])
    b.eval();return {'model':b,'norm':s['normalization'],'null':null}

# ---------------- batched prediction ----------------
from r3_selectors.pipeline import assemble_batch,forward_logits
def predict_many(bundle,graphs,device='cuda',bs=256):
    """graphs: list of graph dicts with non-empty candidate masks. Returns per graph dict:
    pred (node argmax over candidates), null_p (None if no null output), argnull (bool), and gate features over node candidates."""
    out=[]
    for st in range(0,len(graphs),bs):
        chunk=graphs[st:st+bs];samples=[to_sample(g,int(np.flatnonzero(g['candidate_mask'])[0])) for g in chunk]
        b=assemble_batch(samples,bundle['norm'],device)
        with torch.no_grad():lg=forward_logits(bundle['model'],b)
        n=b['candidates'].shape[1];node=lg[:,:n].masked_fill(~b['candidates'],-1e9)
        pn=node.softmax(-1);
        if bundle['null']:
            full=torch.cat([node,lg[:,n:n+1]],1).softmax(-1);nullp=full[:,-1];argnull=full.argmax(-1)==n
        for i,g in enumerate(chunk):
            k=len(g['node_feats']);p=pn[i,:k].cpu().numpy();l=node[i,:k].cpu().numpy();m=g['candidate_mask']
            ps=np.sort(p[m])[::-1];ls=np.sort(l[m])[::-1]
            rec=dict(pred=int(np.argmax(np.where(m,l,-1e30))),maxp=float(ps[0]),margin=float(ps[0]-ps[1]) if len(ps)>1 else 1.0,
                     entropy=float(-(p[m]*np.log(np.clip(p[m],1e-12,1))).sum()),toplogit=float(ls[0]),logit_margin=float(ls[0]-ls[1]) if len(ls)>1 else 10.0,
                     ncand=int(m.sum()),null_p=float(nullp[i]) if bundle['null'] else None,argnull=bool(argnull[i]) if bundle['null'] else False)
            out.append(rec)
    return out
