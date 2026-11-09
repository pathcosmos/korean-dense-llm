import sys; sys.path.insert(0,"/tmp/claude-0/scratch")
import numpy as np, json, time
import nd_run as N
OUT="/tmp/claude-0/scratch/s2"; ND="/tmp/claude-0/scratch/nd"
import os; os.makedirs(OUT,exist_ok=True)
def log(*a): print(time.strftime("%H:%M:%S"),*a,flush=True)
z=np.load(f"{ND}/strat_101.npz"); fsig,ffi,frow=z["sig"],z["fidx"],z["row"]
z=np.load(f"{ND}/hplt_wide.npz"); hsig,hfi,hrow=z["sig"],z["fidx"],z["row"]
log("loaded")
b,bi=N.match(hsig,fsig); np.savez(f"{OUT}/cross_hplt_fw2.npz",best=b,bi=bi); log("cross done",int((b>=0.8).sum()))
ev=(hfi%2==0)
qi=np.nonzero(~ev)[0]; ii=np.nonzero(ev)[0]
b2,bi2=N.match(hsig[qi],hsig[ii]); np.savez(f"{OUT}/within_hplt.npz",best=b2,bi=ii[bi2],qidx=qi); log("within done",int((b2>=0.8).sum()))
