import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, json, os, time
import nd_run as N
OUT="/tmp/claude-0/scratch/s1"; ND="/tmp/claude-0/scratch/nd"
def log(*a): print(time.strftime("%H:%M:%S"),*a,flush=True)
if __name__=="__main__":
    files=sorted(s.rfilename for s in N.info.siblings if s.rfilename.startswith("wiki_ko/text/"))
    cache=f"{OUT}/wiki_sig.npz"
    if os.path.exists(cache): z=np.load(cache); wsig=z["sig"]
    else:
        tasks=[]
        for p in files:
            n,_=N.nrg(p); tasks+=[(p,g) for g in range(n)]
        log("wiki rowgroups",len(tasks))
        from multiprocessing import Pool
        sigs=[]
        with Pool(2) as pool:
            for s,r,g in pool.imap(N.work,tasks,chunksize=1): sigs.append(s)
        wsig=np.concatenate(sigs); np.savez(cache,sig=wsig)
    log("wiki docs with sig",len(wsig))
    res={"wiki_docs_with_sig":int(len(wsig))}
    z=np.load(f"{ND}/strat_101.npz"); f1=json.load(open(f"{ND}/strat_result_101.json"))["indexed_fraction"]
    best,_=N.match(wsig,z["sig"]); del z
    for th in (0.8,0.5): res[f"wiki_vs_fw2_J>={th}"]=dict(hit_rate=float((best>=th).mean()),est_full=float((best>=th).mean()/f1))
    z=np.load(f"{ND}/hplt_wide.npz"); f2=3455188/27434232
    best2,_=N.match(wsig,z["sig"]); del z
    for th in (0.8,0.5): res[f"wiki_vs_hplt_J>={th}"]=dict(hit_rate=float((best2>=th).mean()),est_full=float((best2>=th).mean()/f2))
    res["f_fw2"]=f1;res["f_hplt_approx"]=f2
    json.dump(res,open(f"{OUT}/wiki_result.json","w"),indent=1); log(json.dumps(res))
