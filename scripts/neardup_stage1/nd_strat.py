import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, pyarrow.parquet as pq, random, json, time, os
from multiprocessing import Pool
import nd_run as N
OUT=N.OUT; TOT_FW=58664414; TARGET=0.0455; PER_FILE=None
def rg_sizes(path):
    for a in range(6):
        try:
            with N.fs().open(f"datasets/{N.R}@{N.V}/{path}","rb",block_size=8*1024**2) as fh:
                md=pq.ParquetFile(fh).metadata
                return [md.row_group(g).num_rows for g in range(md.num_row_groups)]
        except Exception as e: err=e; time.sleep(5*(a+1))
    raise err
def run_seed(seed,qs,qf,qrow,sizes):
    cache=f"{OUT}/strat_{seed}.npz"
    rnd=random.Random(seed); tasks=[]
    for fi,p in enumerate(N.FW):
        n=len(sizes[fi]); k=max(1,round(n*TARGET)); tasks+=[(p,g,fi) for g in sorted(rnd.sample(range(n),k))]
    sampled_rows=sum(sizes[fi][g] for _,g,fi in tasks); f=sampled_rows/TOT_FW
    if os.path.exists(cache):
        z=np.load(cache); isig,ifi,irow=z["sig"],z["fidx"],z["row"]
    else:
        sigs=[];fidx=[];rows=[];t0=time.time();done=0
        with Pool(3) as pool:
            for (s,r,g),(p,_,fi) in zip(pool.imap(N.work,[(p,g) for p,g,_ in tasks],chunksize=1),tasks):
                sigs.append(s);rows.append(r);fidx.append(np.full(len(r),fi,dtype=np.int16));done+=1
                if done%50==0: print(f"seed{seed} index {done}/{len(tasks)} rowgroups {sum(len(x) for x in rows)} docs {int(time.time()-t0)}s",flush=True)
        isig=np.concatenate(sigs);ifi=np.concatenate(fidx);irow=np.concatenate(rows)
        np.savez(cache,sig=isig,fidx=ifi,row=irow)
    best,bi=N.match(qs,isig)
    res=dict(seed=seed,sampled_rowgroups=len(tasks),sampled_rows=int(sampled_rows),indexed_docs=int(len(isig)),indexed_fraction=f,query_docs=int(len(qs)))
    for th in (0.8,0.5,0.3):
        c=int((best>=th).sum()); res[f"hits_J>={th}"]=c; res[f"rate_J>={th}"]=c/len(qs); res[f"naive_full_J>={th}"]=c/len(qs)/f
    m=best>=0.8
    res["hits_by_hplt_file_J>=0.8"]={int(x):[int((qf[m]==x).sum()),int((qf==x).sum())] for x in range(5)}
    res["hits_by_fw2_file_J>=0.8"]={int(x):int((ifi[bi[m]]==x).sum()) for x in range(len(N.FW))}
    res["exactish_J>=0.999"]=int((best>=0.999).sum())
    json.dump(res,open(f"{OUT}/strat_result_{seed}.json","w"),indent=1)
    np.save(f"{OUT}/strat_best_{seed}.npy",best)
    print(json.dumps(res),flush=True)
if __name__=="__main__":
    z=np.load(f"{OUT}/hplt_query.npz"); qs,qf,qrow=z["sig"],z["fidx"],z["row"]
    sizes=[rg_sizes(p) for p in N.FW]; print("fw2 files",len(sizes),"rowgroups",sum(len(s) for s in sizes),"rows",sum(sum(s) for s in sizes),flush=True)
    for seed in [int(x) for x in sys.argv[1:]]: run_seed(seed,qs,qf,qrow,sizes)
