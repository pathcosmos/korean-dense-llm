
import json,time,resource
from pathlib import Path
from common.data_contracts import atomic_json,sha256_file
from scripts.audit_main_training import audit_run as prefix_audit
from scripts.audit_checkpoint_retention import audit_run as retention_audit,storage_inventory
root=Path('runs/main_step4500_review_v1');main=Path('runs/3b_full_v1');start=time.monotonic()
prefix=prefix_audit(main,4500);atomic_json(root/'committed_prefix.json',prefix)
config=json.loads((main/'main.json').read_text());parent=json.loads((main/'training/training_contract.json').read_text());cp=main/'training/ckpt_0004500';proof=json.loads((main/'checkpoint_certificates/ckpt_0004500.json').read_text())
binding=retention_audit(main,final=False);assert not binding['issues'] and not binding['residues'];atomic_json(root/'binding_audit.json',binding)
policy=json.loads((main/'contract.json').read_text())['tiered_retention'];begin=time.monotonic()
storage=storage_inventory(main,policy,parent,config,hashes=True,final=False,production=True)
assert not storage['issues'] and not storage['residues'] and set(storage['nfs_resume_candidates'])=={'ckpt_0003500','ckpt_0004000','ckpt_0004500'} and len(storage['local'])<=6 and storage['distinct_checkpoints']==3+len(storage['local'])
content=storage['nfs']['ckpt_0004500']
atomic_json(root/'fresh_checkpoint_content.json',dict(status='passed_actual_main_checkpoint_content',content=content,seconds=time.monotonic()-begin,certificate_sha256=sha256_file(main/'checkpoint_certificates/ckpt_0004500.json'),resident_stats=storage['resident_file_stats']))
atomic_json(root/'storage_snapshot.json',storage)
raw=(main/'training/log.jsonl').read_bytes(); complete=raw.split(b'\n')[:-1]; rows=[json.loads(s) for s in complete];evaluations={str(r['step']):r for r in rows if 'holdout_loss' in r};assert set(evaluations)=={str(step) for step in range(250,4501,250)}
atomic_json(root/'evaluations.json',evaluations)
result=dict(status='passed_actual_main_step4500_prefix_content_and_storage',through_step=4500,committed_step_rank_pairs=prefix['counts']['committed_step_rank_pairs'],committed_tokens=prefix['counts']['committed_tokens'],source_tokens=prefix['counts']['source_tokens'],category_tokens=prefix['category_tokens'],replayed_tokens=prefix['counts']['replayed_tokens_through_prefix'],generation_receipts=prefix['counts']['generation_sha256_by_step'],checkpoint=content,storage=dict(nfs=len(storage['nfs']),local=len(storage['local']),residues=storage['residues'],issues=storage['issues']),seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,evidence={str(p):sha256_file(p) for p in [root/'committed_prefix.json',root/'fresh_checkpoint_content.json',root/'storage_snapshot.json',root/'evaluations.json',root/'binding_audit.json']},actual_main_final_verified=False,actual_checkpoint_gpu_restore_executed=False,goal_complete=False)
assert result['committed_tokens']==9437184000 and result['committed_step_rank_pairs']==36000 and result['replayed_tokens']==0
atomic_json(root/'worker_result.json',result)
print(json.dumps(result))
