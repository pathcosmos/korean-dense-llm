import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, json, random, time, pyarrow.parquet as pq
from multiprocessing import Pool
import nd_run as N, nd_strat as S
OUT="/tmp/claude-0/scratch/s2"; ND="/tmp/claude-0/scratch/nd"; rnd=random.Random(20261109)
def log(*a): print(time.strftime("%H:%M:%S"),*a,flush=True)
sel=[N.HP[i] for i in range(4,len(N.HP),8)]
def hp_sizes(p):
    with N.fs().open(f"datasets/{N.R}@{N.V}/{p}","rb") as fh:
        md=pq.ParquetFile(fh).metadata; return [md.row_group(g).num_rows for g in range(md.num_row_groups)]
def locate(starts,row): return int(np.searchsorted(starts,row,"right")-1)
def readrg(task):
    path,g=task
    for a in range(4):
        try:
            with N.fs().open(f"datasets/{N.R}@{N.V}/{path}","rb",block_size=8*1024**2) as fh:
                pf=pq.ParquetFile(fh); return (path,g,pf.read_row_group(g,columns=["text"]).column("text").to_pylist())
        except Exception as e: err=e; time.sleep(3*(a+1))
    raise err
if __name__=="__main__":
    z=np.load(f"{ND}/strat_101.npz"); ffi,frow=z["fidx"],z["row"]
    z=np.load(f"{ND}/hplt_wide.npz"); hfi,hrow=z["fidx"],z["row"]
    c=np.load(f"{OUT}/cross_hplt_fw2.npz"); cb,cbi=c["best"],c["bi"]
    w=np.load(f"{OUT}/within_hplt.npz"); wb,wbi,wq=w["best"],w["bi"],w["qidx"]
    fsz=[S.rg_sizes(p) for p in N.FW]; fst=[np.concatenate([[0],np.cumsum(s)]) for s in fsz]
    hsz=[hp_sizes(p) for p in sel]; hst=[np.concatenate([[0],np.cumsum(s)]) for s in hsz]
    pairs=[]
    def add(kind,stratum,q,i,j,qside,iside):
        pairs.append(dict(id=f"P{len(pairs):04d}",kind=kind,stratum=stratum,J=float(j),q=qside,i=iside))
    strata=[("0.50-0.65",.5,.65),("0.65-0.80",.65,.8),("0.80-0.90",.8,.9),("0.90-1.00",.9,1.01)]
    for name,lo,hi in strata:
        idx=np.nonzero((cb>=lo)&(cb<hi))[0]; log("cross",name,len(idx))
        for q in rnd.sample(list(idx),100):
            i=cbi[q]; add("cross_hplt_to_fw2",name,q,i,cb[q],
                dict(src="hplt",path=sel[hfi[q]],rg=locate(hst[hfi[q]],hrow[q]),row=int(hrow[q]),base=int(hst[hfi[q]][locate(hst[hfi[q]],hrow[q])])),
                dict(src="fw2",path=N.FW[ffi[i]],rg=locate(fst[ffi[i]],frow[i]),row=int(frow[i]),base=int(fst[ffi[i]][locate(fst[ffi[i]],frow[i])])))
    for name,lo,hi in (("0.80-1.00",.8,1.01),):
        idx=np.nonzero((wb>=lo)&(wb<hi))[0]; log("within_hplt",len(idx))
        for k in rnd.sample(list(idx),100):
            q=wq[k]; i=wbi[k]
            add("within_hplt",name,q,i,wb[k],
                dict(src="hplt",path=sel[hfi[q]],rg=locate(hst[hfi[q]],hrow[q]),row=int(hrow[q]),base=int(hst[hfi[q]][locate(hst[hfi[q]],hrow[q])])),
                dict(src="hplt",path=sel[hfi[i]],rg=locate(hst[hfi[i]],hrow[i]),row=int(hrow[i]),base=int(hst[hfi[i]][locate(hst[hfi[i]],hrow[i])])))
    need=sorted({(p[s]["path"],p[s]["rg"]) for p in pairs for s in("q","i")}); log("pairs",len(pairs),"rowgroups to read",len(need))
    cache={}
    with Pool(3) as pool:
        for n,(path,g,col) in enumerate(pool.imap_unordered(readrg,need,chunksize=1)):
            cache[(path,g)]=col
            if n%100==0: log("read",n,"/",len(need))
    for p in pairs:
        for s in("q","i"):
            d=p[s]; d["text"]=cache[(d["path"],d["rg"])][d["row"]-d["base"]]
    json.dump(pairs,open(f"{OUT}/pairs.json","w"),ensure_ascii=False)
    log("saved",len(pairs))
