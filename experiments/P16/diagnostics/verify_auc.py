import json,numpy as np,sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
F=Path(__file__).resolve().parent/"features"  # copy of the archived presence-head test features
d=np.load(F/"hard_confusable_test.npz",allow_pickle=True); X=d["features"]; names=[str(x) for x in d["feature_names"]]
meta=json.loads((F/"hard_confusable_test_meta.json").read_text(encoding="utf-8"))["meta"]
print("meta keys",list(meta[0].keys()))
cond=np.array([m["condition"] for m in meta]); rm=X[:,names.index("relation_match_count")]
sel=((cond=="target_missing")|(cond=="distractor_missing_matching"))&(rm==1)
Xs=X[sel]; ys=(cond[sel]=="target_missing").astype(int); ms=[m for m,s in zip(meta,sel) if s]
print("n",len(ys),ys.sum())
def old(score,y):
    o=np.argsort(score); r=np.empty(len(score)); r[o]=np.arange(1,len(score)+1); p=y==1
    return (r[p].sum()-p.sum()*(p.sum()+1)/2)/(p.sum()*(~p).sum())
out={}
for j,n in enumerate(names):
    if len(np.unique(Xs[:,j]))<2: continue
    out[n]=(round(old(Xs[:,j],ys),6),round(roc_auc_score(ys,Xs[:,j]),6))
for k,v in out.items(): print(k,v)
json.dump({"n":int(len(ys)),"per_feature_old_vs_sklearn":out,"feature_names":names},open(sys.argv[1]+"/auc_check.json","w"),indent=1)
