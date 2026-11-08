import json,os,shutil,statistics,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import atomic_json,sha256_file
from scripts.hardware_profile import inspect
from scripts.run_local_split import process_tree_rss,stop,size_tree

root=Path(__file__).resolve().parent
hw=inspect();scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=True)
if hw['memory']['conservative_headroom_gib']<448:raise RuntimeError('Need original jobs future peaks plus32GiB probe')
tokenizer=Path('runs/tokenizer_resources_v1/bpe_16/experimental_tokenizer.json')
assert sha256_file(tokenizer)==json.loads(Path('runs/tokenizer_resources_v1/bpe_16/result.json').read_text())['tokenizer_sha256']
contract=dict(hardware=hw,scope='Experimental512MiB truncated real-text fixture,48shards; six-worker Rayon tuning only, not final tokenizer/corpus acceptance.',
 code={str(p):sha256_file(p) for p in [Path(__file__),Path('scripts/benchmark_next_stage.py'),Path('data/parallel_tokenize.py'),Path('data/verified_split_context.py')]},
 tokenizer_sha256=sha256_file(tokenizer),cases=[2,8,8,2],workers=6,batch_size=1024,arrow_threads=1,memory_gib=32,max_seconds=300)
atomic_json(root/'contract.json',contract);(Path('logs')/(root.name+'.pid')).write_text(str(os.getpid()))
results=[];started=time.monotonic();proc=None
try:
 for i,rayon in enumerate(contract['cases']):
  name=f'case_{i}_r{rayon}';out=root/name
  env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS=str(rayon),TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch))
  cmd=['taskset','-c',','.join(map(str,hw['cpu']['affinity'][:64])),sys.executable,'-u','scripts/benchmark_next_stage.py','--mode','encode','--frame','runs/next_stage_optimization_v1/prepare','--output',str(out),'--tokenizer',str(tokenizer),'--workers','6','--batch','1024','--arrow','1']
  peak=0;atomic_json(root/'state.json',dict(phase='running',case=name))
  with (root/(name+'.log')).open('w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   peak=max(peak,process_tree_rss(proc.pid))
   if peak>32*2**30 or time.monotonic()-started>300 or size_tree(root)>8*2**30 or shutil.disk_usage(root).free<200*2**30:raise RuntimeError('Probe resource budget')
   time.sleep(1)
  if proc.returncode:raise RuntimeError('Probe failed:'+name)
  r=json.loads((out/'result.json').read_text());r.update(rayon=rayon,peak_job_rss_bytes=peak,case=name)
  if results and r['logical_id']!=results[0]['logical_id']:raise ValueError('Thread configuration changed native output')
  results.append(r);atomic_json(root/'partial_results.json',results)
 for f,h in contract['code'].items():
  if sha256_file(Path(f))!=h:raise ValueError('Probe code changed')
 medians={str(n):statistics.median(r['end_to_end_seconds'] for r in results if r['rayon']==n) for n in [2,8]}
 report=dict(status='measurement_complete',results=results,median_seconds=medians,candidate_rayon=int(min(medians,key=medians.get)),elapsed_seconds=time.monotonic()-started,scope=contract['scope'])
 atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
 atomic_json(root/'state.json',dict(phase='complete'))
 print(json.dumps(dict(median_seconds=medians,candidate_rayon=report['candidate_rayon'])))
except BaseException as exc:
 if proc is not None and proc.poll() is None:stop(proc)
 atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise
