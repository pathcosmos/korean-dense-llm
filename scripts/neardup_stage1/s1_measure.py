"""Stage-1 near-dup measurements (docs/NEAR_DUP_DESIGN.md sec.6). Reuses existing signature indexes (nd/)."""
import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, pyarrow.parquet as pq, json, time, random, os
import nd_run as N, nd_strat as S
OUT="/tmp/claude-0/scratch/s1"; ND="/tmp/claude-0/scratch/nd"
TOT_FW=58664414; TOT_HP=27434232; SEED=20261109
def log(*a): print(time.strftime("%H:%M:%S"),*a,flush=True)
def split_flags(text_path):
    sp=text_path.replace("/text/","/splits/")
    for a in range(5):
        try:
            with N.fs().open(f"datasets/{N.R}@{N.V}/{sp}","rb",block_size=8*1024**2) as fh:
                t=pq.read_table(fh,columns=["row","split"])
            row=t.column("row").to_numpy(); ev=np.array([x=="eval" for x in t.column("split").to_pylist()])
            out=np.zeros(int(row.max())+1,dtype=bool); out[row]=ev; return out,len(row)
        except Exception as e: err=e; time.sleep(5*(a+1))
    raise err
def boot(hit,grp,f,nb=2000):
    rng=np.random.default_rng(SEED); ug,inv=np.unique(grp,return_inverse=True)
    k=np.bincount(inv,weights=hit.astype(float)); n=np.bincount(inv).astype(float)
    est=k.sum()/n.sum()/f; bs=[]
    for _ in range(nb):
        s=rng.integers(0,len(ug),len(ug)); bs.append(k[s].sum()/n[s].sum()/f)
    return dict(queries=int(n.sum()),groups=int(len(ug)),hit_rate=float(k.sum()/n.sum()),est_full=float(est),ci95=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))])
def run(name,qs,qgrp,isig,f,res):
    t0=time.time(); best,_=N.match(qs,isig); r={}
    for th in (0.8,0.5): r[f"J>={th}"]=boot(best>=th,qgrp,f)
    r["index_docs"]=int(len(isig)); r["index_fraction"]=float(f); res[name]=r
    log(name,json.dumps({k:(v if not isinstance(v,dict) else {kk:v[kk] for kk in("queries","hit_rate","est_full","ci95")}) for k,v in r.items()}),f"{int(time.time()-t0)}s")
    json.dump(res,open(f"{OUT}/result.json","w"),indent=1)
if __name__=="__main__":
    res=json.load(open(f"{OUT}/result.json")) if os.path.exists(f"{OUT}/result.json") else {}
    z=np.load(f"{ND}/strat_101.npz"); fsig,ffi,frow=z["sig"],z["fidx"].astype(np.int64),z["row"]
    z=np.load(f"{ND}/hplt_wide.npz"); hsig,hfi,hrow=z["sig"],z["fidx"].astype(np.int64),z["row"]
    log("loaded",len(fsig),len(hsig))
    # FW2: eval flags + row group
    sizes=[S.rg_sizes(p) for p in N.FW]; fev=np.zeros(len(fsig),bool); frg=np.zeros(len(fsig),np.int64); sizes_all={}
    for fi,p in enumerate(N.FW):
        flags,_=split_flags(p); m=ffi==fi; fev[m]=flags[frow[m]]
        starts=np.concatenate([[0],np.cumsum(sizes[fi])]); frg[m]=np.searchsorted(starts,frow[m],"right")-1
    # HPLT: eval flags
    sel=[N.HP[i] for i in range(4,len(N.HP),8)]; hev=np.zeros(len(hsig),bool); hrows=np.zeros(len(sel),np.int64)
    for k,p in enumerate(sel):
        flags,n=split_flags(p); m=hfi==k; hev[m]=flags[hrow[m]]; hrows[k]=n
    log("eval flags: fw2 eval",int(fev.sum()),"/",len(fev),"hplt eval",int(hev.sum()),"/",len(hev),"hplt rows in 58 files",int(hrows.sum()))
    res["meta"]=dict(fw2_indexed=int(len(fsig)),fw2_eval=int(fev.sum()),hplt_indexed=int(len(hsig)),hplt_eval=int(hev.sum()))
    f_fw_all=json.load(open(f"{ND}/strat_result_101.json"))["indexed_fraction"]; f_hp_all=float(hrows.sum())/TOT_HP
    # --- FW2 within-source: split row groups into halves
    rnd=random.Random(SEED); keys=sorted(set(zip(ffi.tolist(),frg.tolist()))); rnd.shuffle(keys); half=set(keys[:len(keys)//2])
    inA=np.array([(a,b) in half for a,b in zip(ffi.tolist(),frg.tolist())])
    rows_A=sum(sizes[a][b] for a,b in half); fA=rows_A/TOT_FW
    for tag,idxmask,qmask in (("fw2_within_all_vs_trainA",~inA&False,None),):
        pass
    A_train=inA&~fev; B=~inA
    run("fw2_within__queries_B_all__index_A_train",fsig[B],ffi[B]*100000+frg[B],fsig[A_train],fA,res)
    run("fw2_within__queries_B_eval__index_A_train",fsig[B&fev],ffi[B&fev]*100000+frg[B&fev],fsig[A_train],fA,res)
    run("fw2_within__queries_B_train__index_A_train",fsig[B&~fev],ffi[B&~fev]*100000+frg[B&~fev],fsig[A_train],fA,res)
    # --- HPLT within-source: even/odd files
    ev_f=(hfi%2==0); fHA=hrows[0::2].sum()/TOT_HP; fHB=hrows[1::2].sum()/TOT_HP
    run("hplt_within__queries_oddfiles_all__index_evenfiles_train",hsig[~ev_f],hfi[~ev_f],hsig[ev_f&~hev],fHA,res)
    run("hplt_within__queries_oddfiles_eval__index_evenfiles_train",hsig[~ev_f&hev],hfi[~ev_f&hev],hsig[ev_f&~hev],fHA,res)
    run("hplt_within__queries_oddfiles_train__index_evenfiles_train",hsig[~ev_f&~hev],hfi[~ev_f&~hev],hsig[ev_f&~hev],fHA,res)
    run("hplt_within__queries_evenfiles_all__index_oddfiles_train",hsig[ev_f],hfi[ev_f],hsig[~ev_f&~hev],fHB,res)
    # --- cross-source: HPLT queries vs FW2 train index; FW2 queries vs HPLT train index
    run("cross__hplt_eval__vs_fw2_train_index",hsig[hev],hfi[hev],fsig[~fev],f_fw_all,res)
    run("cross__hplt_train__vs_fw2_train_index",hsig[~hev],hfi[~hev],fsig[~fev],f_fw_all,res)
    run("cross__fw2_eval__vs_hplt_train_index",fsig[fev],ffi[fev]*100000+frg[fev],hsig[~hev],f_hp_all,res)
    run("cross__fw2_train__vs_hplt_train_index",fsig[~fev],ffi[~fev]*100000+frg[~fev],hsig[~hev],f_hp_all,res)
    log("done")
