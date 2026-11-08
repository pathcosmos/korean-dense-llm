"""Actual pilot1B/FSDP8 source/optimizer ledger and greedy collective probe."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import resolve,artifact_path,atomic_json,sha256_file,identity


def worker(path):
 import torch
 import torch.distributed as dist
 from train.deterministic_train import configure
 configure()
 import train.train as trainer
 from train.data import HoldoutSet
 from train.rehearsal import SourceLedger,observed_sampler,distributed_generate
 from train.probe import evaluate_sources
 from tokenizers import Tokenizer
 config=json.loads(Path(path).read_text());mix=json.loads(resolve(config['mix']).read_text());rank=int(os.environ['RANK'])
 root=Path(path).parent;ledger=SourceLedger(mix['sources']);trainer.MixSampler=observed_sampler(trainer.MixSampler,ledger)
 trainer.HoldoutSet=lambda files,seq:HoldoutSet(files,seq,n_windows=4)
 groups=[dict(source=s,category=v['category'],start=i*4,count=4) for i,(s,v) in enumerate(mix['sources'].items())]
 probe=dict(seq_len=4096,total_windows=24,groups=groups,scope='Actual final-tokenizer pilot,4windows/source integration only')
 step=torch.optim.AdamW.step
 def update(opt,*args,**kwargs):
  value=step(opt,*args,**kwargs);ledger.commit();return value
 torch.optim.AdamW.step=update
 original=trainer.evaluate_holdout;checks=[]
 prompts=[json.loads(s) for s in resolve('runs/f7_preflight_v1/prompts.jsonl').read_text().splitlines()]
 tokenizer=Tokenizer.from_file(str(resolve('tokenizer/candidate_v1/tokenizer.json')));eos=tokenizer.token_to_id('<|eos|>')
 def evaluate(model,fwd,hold,device):
  result=evaluate_sources(original,model,fwd,hold,device,probe);outputs=[];started=time.monotonic()
  for p in prompts:
   value=distributed_generate(model,fwd,tokenizer.encode(p['text'],add_special_tokens=False).ids,eos_id=eos,context_length=4096,max_new_tokens=64,device=device)
   outputs.append(dict(id=p['id'],category=p['category'],**value))
  ledger.generations.append(dict(step=2,sha256=identity(outputs)));checks.append(dict(source_result=result,generation_sha256=identity(outputs),generation_seconds=time.monotonic()-started))
  return result
 trainer.evaluate_holdout=evaluate
 save=trainer.dcp.save
 def checkpoint(*args,**kwargs):
  result=save(*args,**kwargs)
  atomic_json(Path(config['out_dir'])/f'checkpoint_ledger_rank{rank}.json',dict(steps=ledger.steps,generations=ledger.generations))
  stamp=torch.tensor([1.,2.],dtype=torch.float64,device=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
  dist.all_reduce(stamp,op=dist.ReduceOp.MAX)
  if stamp.tolist()!=[1.,2.]:raise ValueError('Checkpoint timing collective changed')
  return result
 trainer.dcp.save=checkpoint
 sys.argv=['train/train.py','--config',path];trainer.main()
 tc=json.loads((Path(config['out_dir'])/'training_contract.json').read_text())
 if len(ledger.steps)!=2 or len(checks)!=1:raise ValueError('Incomplete actual hook trajectory')
 atomic_json(root/f'rank{rank}.json',dict(rank=rank,success=True,contract_id=tc['id'],steps=ledger.steps,generations=ledger.generations,checks=checks))


def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
 if a.worker:return worker(a.worker)
 from data.pilot_mix import verify_pilot_mix
 from scripts.resource_admission import inspect
 from scripts.parallel_job_support import run_cases,certificate
 from train.rehearsal import aggregate_ledgers
 root=Path(__file__).resolve().parent;hw=inspect();mix=verify_pilot_mix('runs/actual_pilot_input_v1')
 if len(hw['gpus'])!=8 or hw['memory']['cache_aware_admission']['headroom_gib']<128:raise RuntimeError('Actual hook integration hardware budget unavailable')
 lock=resolve('runs/.gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute occupied')
 scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(parents=True,exist_ok=False)
 files=['train/rehearsal.py','train/deterministic_train.py','train/probe.py','train/data.py','train/runtime.py','train/train.py','model/model.py','data/pilot_mix.py','scripts/resource_admission.py','scripts/hardware_profile.py',artifact_path(Path(__file__))]
 contract=dict(hardware=hw,code={f:sha256_file(resolve(f)) for f in files},mix_sha256=sha256_file('runs/actual_pilot_input_v1/mix.json'),
  host_rss_gib=64,max_case_seconds=360,local_budget_gib=32,scope='Actual final-tokenizer pilot1B/FSDP8 two-update source accounting/greedy EOS/checkpoint-journal integration. Not full-corpus or10B rehearsal completion.')
 atomic_json(root/'contract.json',contract);(root/'pid').write_text(str(os.getpid()))
 config=root/'config.json';atomic_json(config,dict(model='1b',vocab_size=65536,seq_len=4096,micro_bsz=8,global_tokens=2097152,max_steps=2,warmup=200,lr=.0006,
  decay_frac=.15,compile=False,mix=str(resolve('runs/actual_pilot_input_v1/mix.json')),holdout=mix['holdout'],eval_every=2,ckpt_every=2,save_final=True,log_every=1,keep_ckpts=3,sampler_version=2,out_dir=str(scratch/'training')))
 cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity'];env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',TMPDIR=str(scratch),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
 cmd=['taskset','-c',','.join(map(str,cpus)),'torchrun','--standalone','--nproc_per_node=8',str(Path(__file__).resolve()),'--worker',str(config)]
 execution=run_cases(root,[dict(name='actual_hooks',out=str(root/'execution'),cmd=cmd,env=env)],max_seconds=360,max_rss_gib=64)
 ranks=[root/f'rank{r}.json' for r in range(8)];reports=[json.loads(p.read_text()) for p in ranks]
 counts=aggregate_ledgers(ranks,world=8,steps=2,seq_len=4096,global_tokens=2097152,sources=mix['sources'])
 if len({r['generations'][0]['sha256'] for r in reports})!=1 or not all(r['checks'][0]['source_result']['valid_tokens']==98304 for r in reports):raise ValueError('Actual source/generation rank agreement differs')
 checkpoint=scratch/'training/ckpt_0000002'
 if not (checkpoint/'.complete').exists() or not all((scratch/'training'/f'checkpoint_ledger_rank{rank}.json').exists() for rank in range(8)):raise ValueError('Checkpoint published without source journals')
 verify_pilot_mix('runs/actual_pilot_input_v1')
 certificate(root,dict(status='passed_bounded_actual_hooks',execution=execution,counts=counts,generation_sha256=reports[0]['generations'][0]['sha256'],
   code_receipts={artifact_path(p):sha256_file(p) for p in ranks},checkpoint=artifact_path(checkpoint),scope=contract['scope']),contract)


if __name__=='__main__':main()
