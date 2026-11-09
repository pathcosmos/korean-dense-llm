import numpy as np, unicodedata, xxhash, re
NP=128
rs=np.random.RandomState(12345)
A=(rs.randint(1,2**62,size=NP,dtype=np.int64).astype(np.uint64)*2+1)
B=rs.randint(0,2**62,size=NP,dtype=np.int64).astype(np.uint64)
P=np.array([0x9E3779B97F4A7C15,0xC2B2AE3D27D4EB4F,0x165667B19E3779F9,0x27D4EB2F165667C5,0x85EBCA77C2B2AE63],dtype=np.uint64)
MAXW=5000
def signature(text):
    w=unicodedata.normalize("NFKC",text).lower().split()[:MAXW]
    if len(w)<5: return None
    ids=np.fromiter((xxhash.xxh64_intdigest(x.encode()) for x in w),dtype=np.uint64,count=len(w))
    n=len(ids)-4
    h=ids[0:n]*P[0]
    for k in range(1,5): h=h^(ids[k:k+n]*P[k])
    # permuted hashes -> min
    sig=np.empty(NP,dtype=np.uint32)
    for s in range(0,NP,32):
        m=((h[None,:]*A[s:s+32,None]+B[s:s+32,None])>>np.uint64(32)).min(axis=1)
        sig[s:s+32]=m.astype(np.uint32)
    return sig
