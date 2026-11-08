import json,os,signal,sys,time,shutil,datetime
from pathlib import Path
from common.data_contracts import atomic_json,sha256_file
from scripts.run_final_retention_audit import admission
from scripts.run_final_resume_probe import supervise_command
from scripts.tiered_checkpoint_retention import size_tree
from train.final_resume_probe import file_stats
root=Path('runs/main_step5500_review_v1');main=Path('runs/3b_full_v1');GIB=2**30
assert not root.exists(),'Preserve any previous review attempt'
assert (main/'checkpoint_certificates/ckpt_0005500.json').is_file(),'Wait for certified5500'
assert (main/'training/ckpt_0005500/.complete').is_file()
settled=json.loads((main/'retention_state.json').read_text());assert settled['nfs_checkpoints']==['ckpt_0004500','ckpt_0005000','ckpt_0005500'] and len(settled['local_checkpoints'])<=8,'Wait for5500 retention to settle'
main_state=json.loads((main/'state.json').read_text());assert main_state['phase']=='running' and main_state['certified_checkpoints']==11,'Main manager has not finished5500 certification'
audit_root=Path('runs/3b_final_retention_audit_v3');audit_contract=json.loads((audit_root/'contract.json').read_text())
hw,budget=admission(audit_root,audit_contract)
combined=budget['combined_memory_gib']+16
assert hw['memory']['cache_aware_admission']['headroom_gib']>=combined
assert budget['nfs_after_budget_reserve_bytes']>=64*2**20
root.mkdir()
atomic_json(root/'hardware.json',hw);atomic_json(root/'capacity.json',budget)
atomic_json(root/'review_budget.json',dict(cpu_affinity=list(range(32)),hash_workers=8,pss_gib=16,max_seconds=120,durable_bytes=64*2**20,additional_local_bytes=0,combined_future_plus_review_memory_gib=combined))
code=Path('runs/main_training_readiness_v1/review_checkpoint5500_worker.py').read_text()
compile(code,'step5500-readonly-review','exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='1',TMPDIR=str(Path('/tmp/llm-dense-d7135f5715/3b_full_v1')))
def guard(pid,elapsed,peak):
 used=size_tree(root)
 if elapsed>120 or peak>16*GIB or used>64*2**20:
  raise RuntimeError('Bounded5500review time/PSS/proof budget exceeded')
 if shutil.disk_usage(root).free < budget['nfs_remaining_bytes']+200*GIB+max(0,64*2**20-used):
  raise RuntimeError('Main/final NFS budgets and reserve unavailable')
 if shutil.disk_usage('/tmp').free < budget['local_remaining_bytes']+200*GIB:
  raise RuntimeError('Local future/temp/reserve unavailable')
def terminate_review(signum,frame):
 raise RuntimeError('Review parent received signal '+str(signum))
signal.signal(signal.SIGTERM,terminate_review)
try:
 result=supervise_command(['taskset','-c','0-31',sys.executable,'-u','-c',code],root=root,env=env,guard=guard)
 report=json.loads((root/'worker_result.json').read_text())
 assert report['through_step']==5500 and report['committed_tokens']==11534344000 and report['committed_step_rank_pairs']==44000
 assert report['replayed_tokens']==0 and report['storage']['nfs']==3 and report['storage']['local']<=8
 assert not report['storage']['residues'] and not report['storage']['issues']
 assert report['checkpoint']['full_payload_sha_refreshed'] is True
 for name,expected in report['evidence'].items():assert sha256_file(Path(name))==expected,name
 content=json.loads((root/'fresh_checkpoint_content.json').read_text())
 for name,expected in content['resident_stats'].items():assert file_stats(Path(name))==expected
 assert sha256_file(main/'checkpoint_certificates/ckpt_0005500.json')==content['certificate_sha256']
 hw_after,budget_after=admission(audit_root,audit_contract)
 assert hw_after['memory']['cache_aware_admission']['headroom_gib']>=combined
 assert budget_after['nfs_after_budget_reserve_bytes']>=64*2**20
 atomic_json(root/'capacity_after.json',budget_after)
 report['retained_checkpoint_sha_scope']='Fresh full SHA of every resident NFS/local checkpoint; adaptive local count checked against durable retirement/loss journal'
 report['supervision']=result;report['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 atomic_json(root/'report.json',report)
 atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),scope='Actual5500prefix/content review only; not final goal or GPU restore'))
 print(json.dumps(report),flush=True)
except BaseException as exc:
 atomic_json(root/'review_error.json',dict(type=type(exc).__name__,error=str(exc),goal_complete=False,source_unchanged=True))
 raise
