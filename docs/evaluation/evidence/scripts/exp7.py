import sys,time,random,string; sys.path.insert(0,'.'); sys.dont_write_bytecode=True
from data.filters import *
from data import clean_code as cc
R=random.Random(1); SY=[chr(0xAC00+i*11) for i in range(200)]
kw=lambda: "".join(R.choice(SY) for _ in range(R.randint(2,4)))
ko=" ".join(kw() for _ in range(150))
print("check_en(한국어 산문):",check_en(ko)," check_en(일본어):",check_en("これは日本語の文章です。"*40)," check_en(불어형 무작위):",check_en(" ".join("".join(R.choice('abcdéèàç') for _ in range(5)) for _ in range(80))))
print("repeated unspaced:",check_ko("가나다라마바사아자차카타파하"*30), "| 19 words repeated:", check_ko(" ".join(["안녕하세요반갑습니다오늘날씨가좋네요"]*19)))
print("20 words repeated:", check_ko(" ".join(["안녕하세요반갑습니다오늘날씨가좋네요"]*20)))
print(".C lang:",cc.select_code(dict(content="# hdr line long enough\nint x;\n",path="a/b.C",repo_name="o/p",license="mit"))[1])
docs=[" ".join(kw() for _ in range(700)) for _ in range(300)]
t0=time.perf_counter(); 
for d in docs: check_ko(norm_ws(d))
dt=time.perf_counter()-t0; ch=sum(len(d) for d in docs)
print(f"check_ko+norm_ws: {len(docs)/dt:.0f} docs/s, {ch/dt/1e6:.1f} Mchar/s (avg {ch//len(docs)} chars)")
t0=time.perf_counter()
for d in docs: mask_pii(d)
print(f"mask_pii {ch/(time.perf_counter()-t0)/1e6:.1f} Mchar/s")
t0=time.perf_counter()
for d in docs: exact_hash(d)
print(f"exact_hash {ch/(time.perf_counter()-t0)/1e6:.1f} Mchar/s")
# catastrophic regex? EMAIL on long string of a's with dots
s="a."*20000
t0=time.perf_counter(); EMAIL.sub("",s); print("EMAIL pathological 'a.'*20000: %.3fs"%(time.perf_counter()-t0))
s="a"*50000+"@"
t0=time.perf_counter(); EMAIL.sub("",s); print("EMAIL 'a'*50000+'@': %.3fs"%(time.perf_counter()-t0))
s="a"*20000+"@"+"b."*5000
t0=time.perf_counter(); EMAIL.sub("",s); print("EMAIL local+@+domain dots: %.3fs"%(time.perf_counter()-t0))
s=("a@"*20000)
t0=time.perf_counter(); EMAIL.sub("",s); print("EMAIL 'a@'*20000: %.3fs"%(time.perf_counter()-t0))
