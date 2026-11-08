"""Read-only actual token SHA/prefetch; never changes the active F6 verifier."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import resolve,artifact_path,identity,atomic_json,sha256_file


def main():
 from scripts.resource_admission import inspect
 from scripts.run_local_split import process_tree_rss
 root=Path(__file__).resolve().parent;(root/'pid').write_text(str(os.getpid()));start=time.monotonic()
 atomic_json(root/'state.json',dict(phase='running',case='prepare_readonly_token_inventory'))
 hw=inspect();tokens=resolve('data/tokens/final_v1');native=json.loads((tokens/'report.json').read_text());cp=tokens/'contract.json'
 if json.loads((tokens/'.done').read_text())!=dict(report_sha256=sha256_file(tokens/'report.json'),contract_sha256=sha256_file(cp)):raise ValueError('Native token completion proof differs')
 if native['status']!='complete' or native['shards']!=native['planned_shards']:raise ValueError('Require actual complete native output')
 if hw['memory']['cache_aware_admission']['headroom_gib']<384:raise RuntimeError('Read-only working-set headroom unavailable')
 rows=[];logical=[]
 for profile in native['files']:
  source=profile['source'];file=profile['path'];directory=tokens/source;r=json.loads((directory/(Path(file).stem+'.bin.done')).read_text());logical.append(dict(source=source,file=file,record=r))
  rows.extend(dict(path=artifact_path(directory/item['name']),bytes=item['bytes'],sha256=item['sha256']) for item in r['outputs'])
 if identity(logical)!=native['logical_id']:raise ValueError('Native output digest inventory differs')
 files=[artifact_path(Path(__file__)),'scripts/resource_admission.py','scripts/hardware_profile.py','common/data_contracts.py']
 contract=dict(hardware=hw,code={f:sha256_file(resolve(f)) for f in files},native_report_sha256=sha256_file(tokens/'report.json'),
   files=rows,workers=32,max_seconds=900,host_rss_gib=64,scope='Readonly complete token prefetch/SHA against immutable native output records. Original F6 verification remains unchanged and mandatory. No cold/warm speed guarantee.')
 atomic_json(root/'contract.json',contract)
 def read(row):
  p=resolve(row['path']);before=p.stat();digest=sha256_file(p);after=p.stat()
  if before.st_size!=row['bytes'] or digest!=row['sha256'] or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Token content mismatch:'+row['path'])
  return row['bytes']
 done=0;read_bytes=0;peak=0
 with ThreadPoolExecutor(max_workers=32) as pool:
  pending={pool.submit(read,row) for row in rows}
  while pending:
   ready,pending=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
   for future in ready:read_bytes+=future.result();done+=1
   peak=max(peak,process_tree_rss(os.getpid()))
   atomic_json(root/'state.json',dict(phase='running',case='readonly_token_prefetch',done=done,total=len(rows),read_bytes=read_bytes,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak))
   if peak>64*2**30 or time.monotonic()-start>900:raise RuntimeError('Prefetch time/RSS budget')
 for f,h in contract['code'].items():
  if sha256_file(resolve(f))!=h:raise ValueError('Read-only prefetch implementation changed')
 if sha256_file(tokens/'report.json')!=contract['native_report_sha256']:raise ValueError('Native token report changed')
 report=dict(status='passed_actual_token_prefetch',files=done,bytes=read_bytes,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak,scope=contract['scope'])
 atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')));atomic_json(root/'state.json',dict(phase='complete'))


if __name__=='__main__':
 try:main()
 except BaseException as exc:atomic_json(Path(__file__).resolve().parent/'state.json',dict(phase='error',error=str(exc)));raise
