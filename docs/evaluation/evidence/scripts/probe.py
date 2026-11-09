import os, sys, json, math, tempfile, threading, locale
sys.path.insert(0,'.')
from pathlib import Path
import common.data_contracts as dc
out={}
# identity
out['id_key_order_invariant']=dc.identity({'a':1,'b':[1,2]})==dc.identity({'b':[1,2],'a':1})
out['id_int_vs_float']=dc.identity({'a':1})==dc.identity({'a':1.0})
out['id_tuple_vs_list']=dc.identity((1,2))==dc.identity([1,2])
out['id_nan']=dc.identity(float('nan'))  # accepted silently?
try: dc.identity({1:'a','b':2}); out['id_mixed_keys']='ok'
except Exception as e: out['id_mixed_keys']=type(e).__name__
try: dc.identity({'p':Path('x')}); out['id_path']='ok'
except Exception as e: out['id_path']=type(e).__name__
out['id_key_int_vs_str']=dc.identity({1:'a'})==dc.identity({'1':'a'})
# atomic_json
d=Path(tempfile.mkdtemp(dir=os.environ.get('S','/tmp')))
dc.atomic_json(d/'x/y.json',{'k':'한글'}); out['atomic_ok']=(d/'x/y.json').read_text(encoding='utf-8')
# non-utf8 default encoding?
out['preferred_encoding']=locale.getpreferredencoding(False)
# concurrent writers share one .tmp name
errs=[]
def w(i):
    try:
        for _ in range(200): dc.atomic_json(d/'c.json',{'i':i,'pad':'x'*5000})
    except Exception as e: errs.append(type(e).__name__)
ts=[threading.Thread(target=w,args=(i,)) for i in range(4)]
[t.start() for t in ts];[t.join() for t in ts]
out['concurrent_errors']=sorted(set(errs)); 
# sha256_file
(d/'e').write_bytes(b''); out['sha_empty']=dc.sha256_file(d/'e')
# relative outside root
try: dc.relative('/etc/passwd')
except Exception as e: out['relative_outside']=type(e).__name__
out['artifact_ext']=dc.artifact_path('/etc/passwd')
# PROJECT_ROOT coupling
print(json.dumps(out,ensure_ascii=False,indent=1,default=str))
