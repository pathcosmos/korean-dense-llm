"""Bounded read-only DB cache warming; original publisher remains authoritative."""
import hashlib,json,math,os,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,TimeoutError
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import atomic_json,sha256_file,resolve
from scripts.hardware_profile import inspect

root=Path(__file__).resolve().parent;hw=inspect();m=json.loads(resolve('data/datasets/final_v2.split/manifest.json').read_text())
target=resolve(m['database']['path']);stat=target.stat();block=8*2**20;window=8;limit=900
if hw['memory']['conservative_headroom_gib']<256:raise RuntimeError('Insufficient headroom for read cache warming')
contract=dict(hardware=hw,database=str(target),expected_sha256=m['database']['sha256'],bytes=stat.st_size,block_bytes=block,read_workers=4,
              pending_blocks=window,anonymous_memory_limit_gib=1,max_seconds=limit,
              budget_note='Original F4 final-reader observed131.35GiB with128GiB cache/108GiB closed DB; expected remaining peak160GiB is distinct from384GiB safety cap. F5 is idle awaiting F4 and cannot start BPE before F4 completes. Stop warming when source terminates.',
              code_sha256=sha256_file(Path(__file__)),scope='Read-only sequential prefetch and actual whole-file SHA. Not F4 SQL completion; no source/output modifications.')
atomic_json(root/'contract.json',contract);Path('logs/f4_prefetch_v1.pid').write_text(str(os.getpid()))
fd=os.open(target,os.O_RDONLY);pool=ThreadPoolExecutor(max_workers=4);pending={};digest=hashlib.sha256();read=0;started=time.monotonic();last=0;reason=None
total=math.ceil(stat.st_size/block)
def chunk(i):
 offset=i*block;wanted=min(block,stat.st_size-offset);parts=[]
 while wanted:
  b=os.pread(fd,wanted,offset)
  if not b:raise ValueError('Unexpected DB EOF')
  parts.append(b);wanted-=len(b);offset+=len(b)
 return b''.join(parts)
try:
 for i in range(min(window,total)):pending[i]=pool.submit(chunk,i)
 for i in range(total):
  state=json.loads(resolve('runs/f4_full_v2/state.json').read_text())
  if state['phase']!='running':reason='source_'+state['phase'];break
  remaining=limit-(time.monotonic()-started)
  if remaining<=0:reason='time_budget';break
  try:b=pending.pop(i).result(timeout=remaining)
  except TimeoutError:reason='time_budget';break
  digest.update(b);read+=len(b);del b
  if i+window<total:pending[i+window]=pool.submit(chunk,i+window)
  now=time.monotonic()
  if now-last>=10:
   atomic_json(root/'state.json',dict(phase='running',case='sequential_read',done=read,total=stat.st_size,elapsed_seconds=now-started,mean_mib_s=read/2**20/max(now-started,.001)))
   rss=int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
   current=int(Path('/sys/fs/cgroup/memory.current').read_text());cap=int(Path('/sys/fs/cgroup/memory.max').read_text())
   if rss>2**30 or cap-current<128*2**30:reason='memory_reserve';break
   last=now
 whole=read==stat.st_size
 if whole and (digest.hexdigest()!=m['database']['sha256'] or (target.stat().st_size,target.stat().st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns)):
  raise ValueError('Published NFS DB SHA/size/mtime differs')
 report=dict(status='passed_whole_file_read' if whole else 'stopped_prefetch',bytes=read,total=stat.st_size,elapsed_seconds=time.monotonic()-started,
             sha256=digest.hexdigest() if whole else None,stop_reason=reason,scope=contract['scope'])
 atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
 atomic_json(root/'state.json',dict(phase='complete',done=read,total=stat.st_size,reason=reason))
except BaseException as exc:
 atomic_json(root/'state.json',dict(phase='error',error=str(exc),done=read,total=stat.st_size));raise
finally:
 for f in pending.values():f.cancel()
 pool.shutdown(wait=True,cancel_futures=True);os.close(fd)
