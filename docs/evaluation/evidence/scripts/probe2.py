import sys,os,hashlib,json,tempfile
sys.path.insert(0,'.')
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from data.public_corpus_reader import iter_split, document_id
from data.release_offsets import scan_boundaries
d=Path(tempfile.mkdtemp(dir=os.environ['S']))
texts=['가나다','abc','라마바','def']
pq.write_table(pa.table({'text':texts,'h':[1,2,3,4]}),d/'t.parquet')
sp=['train','eval','train','train']
sha=[hashlib.sha256(t.encode()).digest() for t in texts]
schema=pa.schema([('row',pa.int64()),('content_sha256',pa.binary(32)),('repo',pa.string()),('group_id',pa.string()),('split',pa.string())],metadata={b'original_final_file':b'f.parquet'})
def wa(sha_,split):
    pq.write_table(pa.Table.from_arrays([pa.array(range(4),pa.int64()),pa.array(sha_,pa.binary(32)),pa.array([None]*4,pa.string()),pa.array(['g']*4),pa.array(split)],schema=schema),d/'a.parquet')
wa(sha,sp)
r=lambda s:[x['text'] for x in iter_split(d/'t.parquet',d/'a.parquet',split=s,source='s',original_file='f.parquet',inventory_id='i')]
print('train',r('train'),'val',r('validation'))
try: r('all')
except Exception as e: print('all ->',e)
bad=list(sha); bad[1]=sha[0]; wa(bad,sp)
try: r('train')
except Exception as e: print('tamper(eval row, train read) ->',repr(e))
# unknown split label in rows not selected
wa(sha,['train','bogus','train','train'])
try: print(r('train'))
except Exception as e: print('bogus split ->',e)
# offsets: literal EOS in text token stream
eos=3
arr=np.array([1,2,eos,5,eos],dtype='<u2'); p=d/'tok.bin'; p.write_bytes(arr.tobytes()); h=hashlib.sha256(p.read_bytes()).hexdigest()
s,e,_=scan_boundaries(p,eos,2,h); print('ok',s,e)
try: scan_boundaries(p,eos,1,h)
except Exception as ex: print('count mismatch ->',ex)
arr=np.array([1,eos,eos,eos],dtype='<u2'); p.write_bytes(arr.tobytes()); h=hashlib.sha256(p.read_bytes()).hexdigest()
# 3 docs: [1,eos],[eos],[eos] - empty docs undetectable
print('ambiguous passes as 3 docs:',scan_boundaries(p,eos,3,h)[:2])
