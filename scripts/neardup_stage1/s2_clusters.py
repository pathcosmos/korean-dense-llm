import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, json, time, collections
import nd_run as N
ND="/tmp/claude-0/scratch/nd"; OUT="/tmp/claude-0/scratch/s2"
def log(*a): print(time.strftime("%H:%M:%S"),*a,flush=True)
def components(sig,th):
    n=len(sig); parent=np.arange(n)
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for b in range(32):
        k=N.bandkeys(sig,b); o=np.argsort(k,kind="stable"); ks=k[o]
        starts=np.nonzero(np.concatenate([[True],ks[1:]!=ks[:-1]]))[0]; ends=np.concatenate([starts[1:],[n]])
        for s,e in zip(starts,ends):
            if e-s<2: continue
            mem=o[s:e]
            if e-s>300: mem=mem[:300]
            r=mem[0]
            for m in mem[1:]:
                if (sig[r]==sig[m]).mean()>=th:
                    a,b2=find(r),find(m)
                    if a!=b2: parent[b2]=a
    roots=np.array([find(i) for i in range(n)]); return roots
res={}
for name,f in (("fw2","strat_101.npz"),("hplt","hplt_wide.npz")):
    z=np.load(f"{ND}/{f}"); sig=z["sig"]; fi=z["fidx"]; row=z["row"]
    for th in (0.8,0.5):
        t0=time.time(); roots=components(sig,th); cnt=collections.Counter(roots.tolist())
        sizes=np.array(list(cnt.values())); big=cnt.most_common(30)
        in_multi=int(sum(v for v in cnt.values() if v>1))
        res[f"{name}_J{th}"]=dict(docs=int(len(sig)),clusters=int(len(cnt)),docs_in_multi=in_multi,share_docs_in_multi=in_multi/len(sig),max=int(sizes.max()),
            top30=[int(v) for _,v in big],size_hist={k:int(((sizes>=lo)&(sizes<hi)).sum()) for k,(lo,hi) in {"1":(1,2),"2":(2,3),"3-4":(3,5),"5-9":(5,10),"10-49":(10,50),"50-199":(50,200),"200+":(200,10**9)}.items()},
            docs_in_size_ge10=int(sum(v for v in cnt.values() if v>=10)),
            top_members=[[ [int(fi[m]),int(row[m])] for m in np.nonzero(roots==r)[0][:3]] for r,_ in big[:30]])
        log(name,th,json.dumps({k:v for k,v in res[f"{name}_J{th}"].items() if k!="top_members"}),f"{int(time.time()-t0)}s")
        json.dump(res,open(f"{OUT}/clusters.json","w"))
