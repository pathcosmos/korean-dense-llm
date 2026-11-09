import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, json, random, unicodedata
import nd_lib as L
import nd_run as N
rnd=random.Random(20261109)
docs=[json.loads(l)["text"] for l in open("/tmp/claude-0/scratch/samples/fineweb2_ko.jsonl",encoding="utf-8")]
docs=[d for d in docs if 300<=len(d) and 60<=len(d.split())<=2000]
rnd.shuffle(docs); base=docs[:300]; pool=docs[300:900]
vocab=[w for d in pool for w in d.split()]
def words(t): return t.split()
def sig(ws): return L.signature(" ".join(ws))
def cand(a,b): return any((N.bandkeys(a[None,:],k)==N.bandkeys(b[None,:],k))[0] for k in range(32))
def foreign(n):
    d=words(rnd.choice(pool)); out=[]
    while len(out)<n: out+=words(rnd.choice(pool))
    return out[:n]
def variants(w):
    n=len(w); v={}
    for p in (0.02,0.05,0.10,0.20):
        x=list(w)
        for i in rnd.sample(range(n),max(1,int(p*n))): x[i]=rnd.choice(vocab)
        v[f"replace_{int(p*100)}%"]=x
    for r in (0.1,0.25,0.5,1.0,2.0):
        k=int(r*n); a=k//2
        v[f"add_foreign_{int(r*100)}%_of_len(split head/tail)"]=foreign(a)+w+foreign(k-a)
    v["add_fixed_50w_head+tail"]=foreign(25)+w+foreign(25)
    v["keep_first_70%"]=w[:int(.7*n)]; v["keep_first_50%"]=w[:int(.5*n)]
    v["keep_middle_50%"]=w[int(.25*n):int(.75*n)]
    return v
res={}
for d in base:
    w=words(unicodedata.normalize("NFKC",d)); 
    if len(w)<10: continue
    s0=sig(w)
    for name,x in variants(w).items():
        if len(x)<5: continue
        s1=sig(x); j=float((s0==s1).mean()); c=cand(s0,s1)
        r=res.setdefault(name,{"n":0,"J_sum":0,"ge08":0,"ge05":0,"cand":0,"cand_and_ge08":0})
        r["n"]+=1;r["J_sum"]+=j;r["ge08"]+=j>=.8;r["ge05"]+=j>=.5;r["cand"]+=c;r["cand_and_ge08"]+=(c and j>=.8)
out={k:dict(n=v["n"],mean_J=round(v["J_sum"]/v["n"],3),share_J_ge_0_8=round(v["ge08"]/v["n"],3),share_J_ge_0_5=round(v["ge05"]/v["n"],3),share_lsh_candidate=round(v["cand"]/v["n"],3),share_detected_at_0_8=round(v["cand_and_ge08"]/v["n"],3)) for k,v in res.items()}
json.dump(out,open("/tmp/claude-0/scratch/s2/recall.json","w"),ensure_ascii=False,indent=1)
for k,v in out.items(): print(k,v)
