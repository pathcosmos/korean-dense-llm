import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, pyarrow.parquet as pq, time, json, os
from multiprocessing import Pool
from huggingface_hub import HfFileSystem, HfApi
from nd_lib import signature, NP
R="pathcosmos/korean-pretrain-corpus"; V="v1.1.1"; OUT="/tmp/claude-0/scratch/nd"; os.makedirs(OUT,exist_ok=True)
info=HfApi().dataset_info(R,revision=V)
def tfiles(src): return sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(src+"/text/"))
FW=tfiles("fineweb2_ko"); HP=tfiles("hplt_ko")
_fs=None
def fs():
    global _fs
    if _fs is None: _fs=HfFileSystem(token=False)
    return _fs
def work(task):
    path,g=task
    for attempt in range(4):
        try:
            with fs().open(f"datasets/{R}@{V}/{path}","rb",block_size=8*1024**2) as fh:
                pf=pq.ParquetFile(fh)
                base=sum(pf.metadata.row_group(k).num_rows for k in range(g))
                col=pf.read_row_group(g,columns=["text"]).column("text").to_pylist()
            break
        except Exception as e:
            err=e; time.sleep(3*(attempt+1))
    else: raise err
    sigs=[];rows=[]
    for j,t in enumerate(col):
        if len(t)<300: continue
        s=signature(t)
        if s is None: continue
        sigs.append(s); rows.append(base+j)
    return (np.array(sigs,dtype=np.uint32).reshape(-1,NP), np.array(rows,dtype=np.int64), g)
def nrg(path):
    with fs().open(f"datasets/{R}@{V}/{path}","rb") as fh: return pq.ParquetFile(fh).num_row_groups, pq.ParquetFile(fh).metadata.num_rows
def build(path_list,tag):
    cache=f"{OUT}/{tag}.npz"
    if os.path.exists(cache):
        z=np.load(cache); return z["sig"],z["fidx"],z["row"]
    tasks=[];
    for fi,p in enumerate(path_list):
        n,_=nrg(p); tasks+= [(p,g,fi) for g in range(n)]
    sigs=[];fidx=[];rows=[];t0=time.time();done=0
    with Pool(3) as pool:
        for (s,r,g),(p,_,fi) in zip(pool.imap(work,[(p,g) for p,g,_ in tasks],chunksize=1),tasks):
            sigs.append(s);rows.append(r);fidx.append(np.full(len(r),fi,dtype=np.int16));done+=1
            if done%20==0: print(tag,done,"/",len(tasks),"rowgroups",sum(len(x) for x in rows),"docs",int(time.time()-t0),"s",flush=True)
    sig=np.concatenate(sigs);fidx=np.concatenate(fidx);row=np.concatenate(rows)
    np.savez(cache,sig=sig,fidx=fidx,row=row); return sig,fidx,row
MULT=np.array([0x9E3779B97F4A7C15,0xC2B2AE3D27D4EB4F,0x165667B19E3779F9,0x27D4EB2F165667C5],dtype=np.uint64)
def bandkeys(sig,b):
    x=sig[:,b*4:(b+1)*4].astype(np.uint64)
    return x[:,0]*MULT[0]^x[:,1]*MULT[1]^x[:,2]*MULT[2]^x[:,3]*MULT[3]
def match(qs,isig):
    best=np.zeros(len(qs),dtype=np.float32); bi=np.full(len(qs),-1,dtype=np.int64)
    for b in range(NP//4):
        ik=bandkeys(isig,b); order=np.argsort(ik,kind="stable"); iks=ik[order]
        qk=bandkeys(qs,b)
        lo=np.searchsorted(iks,qk,"left"); hi=np.searchsorted(iks,qk,"right")
        cand=np.nonzero(hi>lo)[0]
        for q in cand:
            for pos in range(lo[q],min(hi[q],lo[q]+50)):
                i=order[pos]; j=float((qs[q]==isig[i]).mean())
                if j>best[q]: best[q]=j; bi[q]=i
    return best,bi
if __name__=="__main__":
    fw_idx=[int(x) for x in sys.argv[1:]] or [3,17]
    hp_sel=[HP[i] for i in (0,116,232,348,463)]
    qsig,qf,qrow=build(hp_sel,"hplt_query"); print("query docs",len(qsig),flush=True)
    tot_fw=58664414
    for k in fw_idx:
        path=FW[k]; isig,ifi,irow=build([path],f"fw2_index_{k}")
        _,nrows=nrg(path); f=nrows/tot_fw
        t0=time.time(); best,bi=match(qsig,isig); print("match secs",int(time.time()-t0))
        res=dict(fw2_file=path,indexed_docs=int(len(isig)),file_rows=int(nrows),indexed_fraction_of_fw2=f,query_docs=int(len(qsig)))
        for th in (0.8,0.5,0.3): 
            c=int((best>=th).sum()); res[f"q_with_match_J>={th}"]=c; res[f"rate_J>={th}"]=c/len(qsig); res[f"extrapolated_to_full_fw2_J>={th}"]=min(1.0,c/len(qsig)/f)
        pairs=[dict(q=[int(qf[q]),int(qrow[q])],i=int(irow[bi[q]]),J=float(best[q])) for q in np.argsort(-best)[:400] if best[q]>=0.3]
        json.dump(dict(summary=res,pairs=pairs),open(f"{OUT}/result_fw{k}.json","w"),indent=1)
        print(json.dumps(res,indent=1),flush=True)
