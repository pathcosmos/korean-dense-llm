import sys, os, json, shutil, sqlite3, unicodedata, subprocess, time
from pathlib import Path
sys.path.insert(0,'.')
import pyarrow as pa, pyarrow.parquet as pq
from data.dedup import deduplicate
W=Path(sys.argv[1]) if len(sys.argv)>1 else Path('work'); 
def mk(root):
    shutil.rmtree(root,ignore_errors=True); root.mkdir(parents=True)
    a,b,c=root/'a.parquet',root/'b.parquet',root/'c.parquet'
    nfd='é'; nfc='é'
    pq.write_table(pa.table({'text':['Hello world','same body','caf'+nfc,'tab\tsep'],'h':[1,2,3,4]}),a)
    pq.write_table(pa.table({'text':['hello world','Hello world',' Hello world','Hello  world','Hello world\n','caf'+nfd,'same body','x'],'h':[5,6,7,8,9,10,11,12]}),b)
    pq.write_table(pa.table({'text':['x','y','Hello world'],'h':[13,14,15]}),c)
    return [('s1',a),('s2',b),('s3',c)]
# 1. normal
r=W/'t1'; ins=mk(r/'in'); print('NORMAL',deduplicate(ins,r/'out','p'))
for s,p in ins:
    f=r/'out'/'clean'/s/p.name; print(s,p.name,pq.read_table(f).column('text').to_pylist())
# 2. crash mid-shard-2 (after writing tmp, before db commit) via os._exit in subprocess; then resume
r=W/'t2'; ins=mk(r/'in')
code=f'''
import sys,os
sys.path.insert(0,".")
from pathlib import Path
import pyarrow.parquet as pq
import data.dedup as d
orig=Path.replace
n=[0]
def rep(self,t):
    if str(self).endswith("b.parquet.tmp"): os._exit(77)   # crash just before rename of shard 2
    return orig(self,t)
Path.replace=rep
ins=[("s1",Path("{ins[0][1]}")),("s2",Path("{ins[1][1]}")),("s3",Path("{ins[2][1]}"))]
d.deduplicate(ins,Path("{r/'out'}"),"p")
'''
p=subprocess.run([sys.executable,'-c',code]); print('crash rc',p.returncode)
db=sqlite3.connect(r/'out'/'index.sqlite'); print('after crash seen',db.execute('select count(*) from seen').fetchone(),'processed',db.execute('select name from processed').fetchall()); db.close()
print('resume',deduplicate(ins,r/'out','p'))
# compare to uninterrupted
r1=W/'t1'; 
for s,pth in ins:
    a=pq.read_table(r/'out'/'clean'/s/pth.name).column('text').to_pylist(); b=pq.read_table(W/'t1'/'out'/'clean'/s/pth.name).column('text').to_pylist(); print('resume==clean',s,a==b)
# 3. input change detection
print('rerun idempotent',deduplicate(ins,r/'out','p'))
os.utime(ins[2][1],ns=(time.time_ns(),time.time_ns()+10**9))
try: deduplicate(ins,r/'out','p'); print('mtime change NOT detected')
except RuntimeError as e: print('mtime change ->',e)
# 4. content change same size & restored mtime
r=W/'t4'; ins=mk(r/'in'); deduplicate(ins,r/'out','p')
st=ins[2][1].stat(); 
t=pq.read_table(ins[2][1]); pq.write_table(pa.table({'text':['x','z','Hello world'],'h':[13,14,15]}),ins[2][1])
print('size same?',st.st_size==ins[2][1].stat().st_size)
os.utime(ins[2][1],ns=(st.st_atime_ns,st.st_mtime_ns))
try: print('same size+mtime content change:',deduplicate(ins,r/'out','p'),'-> NOT detected')
except RuntimeError as e: print('detected',e)
# 5. h column / schema robustness: missing h
r=W/'t5'; shutil.rmtree(r,ignore_errors=True); r.mkdir(parents=True); a=r/'a.parquet'; pq.write_table(pa.table({'text':['a']}),a)
try: deduplicate([('s',a)],r/'out','p')
except Exception as e: print('no h column ->',type(e).__name__,e)
# 6. surrogate / lone surrogate text, empty strings
r=W/'t6'; shutil.rmtree(r,ignore_errors=True); r.mkdir(parents=True); a=r/'a.parquet'; pq.write_table(pa.table({'text':['','',' ',' ','　', 'a﻿','a'],'h':list(range(7))}),a)
print('edge', deduplicate([('s',a)],r/'out','p'), pq.read_table(r/'out/clean/s/a.parquet').column('text').to_pylist())
