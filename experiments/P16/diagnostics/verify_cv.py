import json,numpy as np,torch,sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
F=Path(__file__).resolve().parent/"features"  # copy of the archived presence-head test features
d=np.load(F/"hard_confusable_test.npz",allow_pickle=True); X=d["features"]; names=[str(x) for x in d["feature_names"]]
meta=json.loads((F/"hard_confusable_test_meta.json").read_text(encoding="utf-8"))["meta"]
cond=np.array([m["condition"] for m in meta]); rm=X[:,names.index("relation_match_count")]
sel=((cond=="target_missing")|(cond=="distractor_missing_matching"))&(rm==1)
Xs=X[sel]; ys=(cond[sel]=="target_missing").astype(int)
g=np.array([m["group_id"] for m,s in zip(meta,sel) if s]); sc=np.array([m["scene_id"] for m,s in zip(meta,sel) if s])
print("groups",len(set(g)),"scenes",len(set(sc)))
def train_fold(tr,te,scores):
    mu,sd=Xs[tr].mean(0),Xs[tr].std(0); sd[sd<1e-9]=1
    Xt=torch.tensor((Xs[tr]-mu)/sd,dtype=torch.float32); yt=torch.tensor(ys[tr],dtype=torch.float32)
    m=torch.nn.Linear(Xs.shape[1],1); opt=torch.optim.Adam(m.parameters(),lr=0.05,weight_decay=1e-3); L=torch.nn.BCEWithLogitsLoss()
    for _ in range(400): opt.zero_grad(); L(m(Xt).squeeze(-1),yt).backward(); opt.step()
    with torch.no_grad(): scores[te]=m(torch.tensor((Xs[te]-mu)/sd,dtype=torch.float32)).squeeze(-1).numpy()
def run(splits):
    torch.manual_seed(20261002); s=np.zeros(len(ys))
    for tr,te in splits: train_fold(tr,te,s)
    return roc_auc_score(ys,s)
# original split
idx=np.arange(len(ys)); np.random.seed(20261002); torch.manual_seed(20261002); rng=np.random.default_rng(20261002); rng.shuffle(idx); ch=np.array_split(idx,5)
orig=[(np.concatenate([c for j,c in enumerate(ch) if j!=k]),ch[k]) for k in range(5)]
leak=sum(len(set(g[ch[k]])&set(g[np.concatenate([c for j,c in enumerate(ch) if j!=k])])) for k in range(5))
r={"orig_split_sklearn_auc":run(orig),"groups_shared_across_folds":int(leak),
   "groupkfold_group_id":run(list(GroupKFold(5).split(Xs,ys,g))),"groupkfold_scene_id":run(list(GroupKFold(5).split(Xs,ys,sc))),
   "n_groups":len(set(g)),"n_scenes":len(set(sc))}
print(r); json.dump(r,open(sys.argv[1]+"/cv_check.json","w"),indent=1)
