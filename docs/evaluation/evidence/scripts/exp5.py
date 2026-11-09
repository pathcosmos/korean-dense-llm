import sys,os,tempfile,random,collections,json; sys.path.insert(0,'.'); sys.dont_write_bytecode=True
import pyarrow as pa, pyarrow.parquet as pq
from unittest.mock import patch
from data import clean
from data.filters import *
R=random.Random(3); SY=[chr(0xAC00+i*11) for i in range(200)]
def kw(): return "".join(R.choice(SY) for _ in range(R.randint(2,4)))
body=" ".join(kw() for _ in range(80))
docA=body+" Hello World 연락처 010-1234-5678 입니다"
docB=docA.replace("Hello World","hello  world")      # case+space variant
docC=body+" Hello World 연락처 010-9999-0000 입니다"  # differs only in phone
with tempfile.TemporaryDirectory() as tmp:
    src=os.path.join(tmp,"s.parquet")
    pq.write_table(pa.table({"text":[docA,docB,docC,None,"  "+docA+"\r\n\r\n\r\n\r\n"]}),src)
    with patch.object(clean,"ROOT",tmp):
        name,(cnt,chars)=clean.work(("x","ko",src)); print(dict(cnt))
        out=pq.read_table(f"{tmp}/clean/x/s.parquet").to_pylist()
        for r in out: print(r["h"], repr(r["text"][-40:]))
        print("h of kept == exact_hash(premask)+offset:", out[0]["h"]==exact_hash(norm_ws(docA))-(1<<63))
        # silent skip after input change / policy change
        pq.write_table(pa.table({"text":[docC]*1+[body+" 완전히 다른 문서 "+" ".join(kw() for _ in range(40))]}),src)
        name,(cnt2,_)=clean.work(("x","ko",src)); print("rerun after input changed ->",dict(cnt2),"(stale stats served)")
        # lang switch on same output
        name,(cnt3,_)=clean.work(("x","en",src)); print("rerun with different lang ->",dict(cnt3))
        print(sorted(os.listdir(f"{tmp}/clean/x")))
        stats=json.load(open(f"{tmp}/clean/x/s.parquet.stats.json")); print(stats.keys())
    # crash mid-way: stale .tmp
    src2=os.path.join(tmp,"t.parquet"); pq.write_table(pa.table({"text":[docA]}),src2)
    os.makedirs(f"{tmp}/clean/y",exist_ok=True); open(f"{tmp}/clean/y/t.parquet.tmp","wb").write(b"junk")
    with patch.object(clean,"ROOT",tmp):
        print("stale .tmp recover:",dict(clean.work(("y","ko",src2))[1][0]))
    # empty-string/whitespace and non-str column
    src3=os.path.join(tmp,"u.parquet"); pq.write_table(pa.table({"text":["", "   ", docA]}),src3)
    with patch.object(clean,"ROOT",tmp): print("empty/ws:",dict(clean.work(("z","ko",src3))[1][0]))
    # missing text column
    src4=os.path.join(tmp,"v.parquet"); pq.write_table(pa.table({"content":[docA]}),src4)
    try:
        with patch.object(clean,"ROOT",tmp): clean.work(("w","ko",src4))
    except Exception as e: print("missing col ->",type(e).__name__, str(e)[:80], sorted(os.listdir(f"{tmp}/clean/w")))
    # re-run after failure
    try:
        with patch.object(clean,"ROOT",tmp): clean.work(("w","ko",src4))
    except Exception as e: print("rerun after failure ->",type(e).__name__)
