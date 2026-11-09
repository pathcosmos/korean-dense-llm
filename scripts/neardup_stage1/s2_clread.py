import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, json, pyarrow.parquet as pq
import nd_run as N, nd_strat as S
import s2_pairs as P
OUT="/tmp/claude-0/scratch/s2"
C=json.load(open(f"{OUT}/clusters.json"))
fsz=[S.rg_sizes(p) for p in N.FW]; fst=[np.concatenate([[0],np.cumsum(s)]) for s in fsz]
hsz=[P.hp_sizes(p) for p in P.sel]; hst=[np.concatenate([[0],np.cumsum(s)]) for s in hsz]
out={}
for key,src,n in (("fw2_J0.5","fw2",14),("hplt_J0.5","hplt",14),("fw2_J0.8","fw2",8),("hplt_J0.8","hplt",8)):
    cl=C[key]; res=[]
    for ci in range(n):
        mem=cl["top_members"][ci][:2]; texts=[]
        for fi,row in mem:
            path=N.FW[fi] if src=="fw2" else P.sel[fi]; st=fst[fi] if src=="fw2" else hst[fi]
            g=int(np.searchsorted(st,row,"right")-1)
            _,_,col=P.readrg((path,g)); texts.append(col[row-int(st[g])])
        res.append(dict(size=cl["top30"][ci],texts=[t[:260].replace("\n"," | ") for t in texts],lens=[len(t) for t in texts]))
    out[key]=res
json.dump(out,open(f"{OUT}/cluster_texts.json","w"),ensure_ascii=False,indent=1)
for k,v in out.items():
    print("=====",k)
    for c in v: print(c["size"],c["lens"],"\n  ",c["texts"][0],"\n  ",c["texts"][1])
