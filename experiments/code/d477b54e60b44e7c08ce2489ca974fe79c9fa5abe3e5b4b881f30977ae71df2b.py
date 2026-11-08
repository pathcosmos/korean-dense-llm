"""Final-tokenizer scheduler reconfirmation on the preserved experimental frame."""
import json,os,statistics,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import atomic_json,sha256_file,resolve
from scripts.hardware_profile import inspect
from scripts.run_local_split import process_tree_rss,size_tree,stop
from scripts.run_final_tokens import verify_tokenizer_run
root=Path(__file__).resolve().parent;hw=inspect()
if hw['memory']['conservative_headroom_gib']<448:raise RuntimeError('Reserve active token reader future peak plus32GiB experiment')
scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=True)
c=json.loads(resolve('runs/f5_tokenizer_v1/contract.json').read_text());trained,_=verify_tokenizer_run(resolve('runs/f5_tokenizer_v1'),c['sources_sha256'],c['canonical_split_sha256'])
files=['scripts/benchmark_next_stage.py','data/parallel_tokenize.py','data/verified_split_context.py','data/local_split_reader.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py']
contract=dict(hardware=hw,tokenizer_sha256=trained['tokenizer_sha256'],f5_report_sha256=sha256_file(resolve('runs/f5_tokenizer_v1/report.json')),
 code={p:sha256_file(resolve(p)) for p in files},driver_sha256=sha256_file(Path(__file__)),initial_cases=[[16,3],[24,2]],repeat_if_relative_gap_above=.03,
 memory_gib=32,max_seconds=300,scope='Actual final tokenizer on preserved512MiB/48shard truncated real-text experimental frame. Not full-source pilot/quality acceptance or global optimum.')
atomic_json(root/'contract.json',contract);Path('logs/final_tokenizer_encoding_probe_v1.pid').write_text(str(os.getpid()))
results=[];started=time.monotonic();proc=None
def case(workers,rayon):
 global proc
 name=f'case_{len(results)}_w{workers}_r{rayon}';out=root/name;start=time.monotonic();peak=0
 env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS=str(rayon),TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch))
 cmd=['taskset','-c',','.join(map(str,hw['cpu']['affinity'][:64])),sys.executable,'-u','scripts/benchmark_next_stage.py','--mode','encode',
      '--frame','runs/next_stage_optimization_v1/prepare','--output',str(out),'--tokenizer',trained['tokenizer'],'--workers',str(workers),'--batch','1024','--arrow','1']
 with (root/(name+'.log')).open('w') as stream:proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 while proc.poll() is None:
  peak=max(peak,process_tree_rss(proc.pid));atomic_json(root/'state.json',dict(phase='running',case=name,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=peak))
  if time.monotonic()-started>300 or peak>32*2**30 or size_tree(root)>8*2**30:raise RuntimeError('Final-tokenizer probe budget')
  time.sleep(1)
 if proc.returncode:raise RuntimeError('Final-tokenizer encoding probe failed:'+name)
 r=json.loads((out/'result.json').read_text());r.update(rayon_threads=rayon,peak_group_rss_bytes=peak,case=name)
 if results and r['logical_id']!=results[0]['logical_id']:raise ValueError('Resource settings changed native encoded bytes')
 results.append(r);atomic_json(root/'partial_results.json',results);return r
try:
 a=case(16,3);b=case(24,2)
 if abs(a['end_to_end_seconds']/b['end_to_end_seconds']-1)>.03:case(24,2);case(16,3)
 medians={str(w):statistics.median(r['end_to_end_seconds'] for r in results if r['workers']==w) for w in [16,24]}
 winner=16 if medians['16']<medians['24']*.97 else 24
 for p,h in contract['code'].items():
  if sha256_file(resolve(p))!=h:raise ValueError('Encoding probe producer changed')
 report=dict(status='measurement_complete',results=results,median_end_to_end_seconds=medians,candidate=dict(workers=winner,rayon_threads=3 if winner==16 else 2),
             scope=contract['scope'],production_adoption='pending_actual_whole_file_pilot_review')
 atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
 atomic_json(root/'state.json',dict(phase='complete'));print(json.dumps(dict(medians=medians,candidate=report['candidate'])))
except BaseException as exc:
 if proc is not None and proc.poll() is None:stop(proc)
 atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise
