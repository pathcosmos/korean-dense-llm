import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from common.data_contracts import atomic_json,sha256_file
from scripts.adopt_main_worker import stamp,AdoptedWorker

b=Path('runs/retention3_migration_v1');main=Path('runs/3b_full_v1')
assert 'Ran 34 tests' in (b/'capacity_revision_tests.log').read_text() and (b/'capacity_revision_tests.log').read_text().rstrip().endswith('OK')
copy=json.loads(Path('runs/tiered_retention_copy_v2/report.json').read_text());assert copy['cpu_state_exact'] and copy['all8_original_cpu_cuda_rng_exact']
c=json.loads((main/'contract.json').read_text())
for name,digest in {**c['code'],**c['evidence']}.items():assert sha256_file(Path(name))==digest,name
training=main/'training';assert not any(p.is_dir() and not (main/'checkpoint_certificates'/(p.name+'.json')).exists() for p in training.glob('ckpt_*'))
pid=int((main/'pid').read_text());old_start,old_cmd=stamp(pid);assert b'scripts/supervise_main_training.py' in old_cmd
state=json.loads((main/'state.json').read_text());worker=state['worker_pid'];worker_start,_=stamp(worker)
ranks={str(p):stamp(p)[0] for p in map(int,Path(f'/proc/{worker}/task/{worker}/children').read_text().split())};assert len(ranks)==8
receipt=dict(authorized=True,utc=datetime.now(timezone.utc).isoformat(),main_run=str(main),contract_sha256=sha256_file(main/'contract.json'),
             previous_supervisor={'pid':pid,'start_ticks':old_start},worker={'pid':worker,'start_ticks':worker_start},ranks=ranks,
             elapsed_before_seconds=state['elapsed_seconds'],training_contract_sha256=sha256_file(training/'training_contract.json'),
             config_sha256=sha256_file(main/'main.json'),authorization=str(b/'authorization.json'),scope='Supervisor handoff only; original rank PIDs, optimizer, RNG and attempt stay live')
AdoptedWorker(worker,main.resolve(),receipt)
with (main/'retention.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
atomic_json(b/'capacity_handoff.json',receipt)
os.kill(pid,signal.SIGTERM)  # Parent only; torchrun was launched in a separate session.
end=time.monotonic()+10
while time.monotonic()<end:
    try:stamp(pid)
    except (FileNotFoundError,ProcessLookupError):break
    time.sleep(.1)
else:raise RuntimeError('Previous supervisor did not exit; no second controller started')
assert stamp(worker)[0]==worker_start
env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
command=[sys.executable,'-u','scripts/supervise_main_training.py','--run-dir',str(main),'--run','--adopt-receipt',str(b/'capacity_handoff.json')]
with (b/'capacity_supervisor_console.log').open('x') as stream:
    proc=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(b/'capacity_new_supervisor.json',dict(pid=proc.pid,command=command))
deadline=time.monotonic()+300
while time.monotonic()<deadline:
    if proc.poll() is not None:raise RuntimeError('New supervisor exited; inspect preserved console and main protection state')
    if (main/'supervisor_adopted.json').exists() and json.loads((main/'supervisor_adopted.json').read_text()).get('receipt')==str(b/'capacity_handoff.json') and int((main/'pid').read_text())==proc.pid:
        current=json.loads((main/'state.json').read_text())
        retention=json.loads((main/'retention_state.json').read_text())
        if current.get('worker_pid')==worker and len(retention['nfs_checkpoints'])==3 and len(retention['local_checkpoints'])<=15:break
    time.sleep(1)
else:raise RuntimeError('Handoff observation deadline; inspect actual supervisor, never restart from timeout')
assert sha256_file(training/'training_contract.json')==receipt['training_contract_sha256']
assert sha256_file(main/'main.json')==receipt['config_sha256']
assert stamp(worker)[0]==worker_start
for rank,start in ranks.items():assert stamp(int(rank))[0]==start
atomic_json(b/'capacity_handoff_result.json',dict(status='passed_live_eight_rank_handoff',utc=datetime.now(timezone.utc).isoformat(),
          old_supervisor_pid=pid,new_supervisor_pid=proc.pid,worker_pid=worker,rank_pids=list(map(int,ranks)),
          ranks_restarted=False,training_contract_unchanged=True,config_unchanged=True,retention=retention))
print(json.dumps({'handoff':'passed','new_supervisor_pid':proc.pid,'worker_pid':worker,'nfs':retention['nfs_checkpoints'],'local':retention['local_checkpoints']}),flush=True)
