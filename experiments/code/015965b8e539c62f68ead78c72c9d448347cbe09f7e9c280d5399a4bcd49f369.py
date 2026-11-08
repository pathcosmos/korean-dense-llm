"""Synthetic8GPU source evaluation1/7/9 windows against independent serial NLL."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import atomic_json,sha256_file,resolve,artifact_path


def worker(path):
 import torch
 import torch.distributed as dist
 import train.train as trainer
 from train.probe import evaluate_sources
 config=json.loads(Path(path).read_text());rank=int(os.environ['RANK']);original=trainer.evaluate_holdout;checks=[]
 probe=dict(seq_len=32,total_windows=17,scope='Synthetic grouped1/7/9 only',groups=[dict(source='a',category='ko',start=0,count=1),dict(source='b',category='ko',start=1,count=7),dict(source='c',category='en',start=8,count=9)])
 def evaluate(model,fwd,hold,device):
  r=evaluate_sources(original,model,fwd,hold,device,probe);previous=model.training;model.eval();expected={}
  try:
   with torch.no_grad():
    for group in probe['groups']:
     nll=0.
     for row in hold.wins[group['start']:group['start']+group['count']]:
      w=row[None].to(device)
      with torch.autocast('cuda',dtype=torch.bfloat16):loss=fwd(w[:,:-1].contiguous(),w[:,1:].contiguous())
      nll+=float(loss.item())*32
     expected[group['source']]=dict(nll_sum=nll,valid_tokens=group['count']*32)
  finally:model.train(previous)
  total_nll=sum(v['nll_sum'] for v in expected.values());total_tokens=sum(v['valid_tokens'] for v in expected.values())
  error=abs(r['holdout_loss']-total_nll/total_tokens)
  if error>1e-6 or r['valid_tokens']!=544:raise ValueError('Source evaluator differs from independent serial reference')
  for group in probe['groups']:
   actual=r['by_source'][group['source']];e=expected[group['source']]
   if actual['valid_tokens']!=e['valid_tokens'] or abs(actual['holdout_loss']-e['nll_sum']/e['valid_tokens'])>1e-6:raise ValueError('Per-source NLL differs')
  for category in ['ko','en']:
   names=[g['source'] for g in probe['groups'] if g['category']==category];count=sum(expected[n]['valid_tokens'] for n in names);loss=sum(expected[n]['nll_sum'] for n in names)/count
   if r['by_category'][category]['valid_tokens']!=count or abs(r['by_category'][category]['holdout_loss']-loss)>1e-6:raise ValueError('Category NLL weighting differs')
  checks.append(dict(reference_error=error,**r));return r
 trainer.evaluate_holdout=evaluate;sys.argv=['train/train.py','--config',str(path)];trainer.main()
 if len(checks)!=2:raise ValueError('Expected two source-aware evaluations')
 atomic_json(Path(config['out_dir'])/f'rank{rank}.json',dict(status='passed_bounded_synthetic',checks=checks))


def main():
 p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
 if a.worker:return worker(a.worker)
 from scripts.hardware_profile import inspect
 from scripts.parallel_job_support import run_cases,certificate
 import numpy as np
 root=Path(__file__).resolve().parent;hw=inspect();scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=True)
 if len(hw['gpus'])!=8 or hw['memory']['conservative_headroom_gib']<64:raise RuntimeError('Synthetic source eval GPU/RAM budget unavailable')
 lock=resolve('runs/.gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute occupied')
 rng=np.random.default_rng(231);train=root/'train.bin';hold=root/'hold.bin';rng.integers(0,128,4096,dtype=np.uint16).tofile(train);rng.integers(0,128,17*33,dtype=np.uint16).tofile(hold)
 mix=root/'mix.json';atomic_json(mix,dict(sampler_version=2,tokenizer_sha256='synthetic-source-probe-not-final',sources=dict(synthetic=dict(weight=1,files=[str(train)])),holdout=[str(hold)]))
 files=['train/probe.py','train/data.py','train/runtime.py','train/train.py','model/model.py',artifact_path(Path(__file__))]
 contract=dict(hardware=hw,code={f:sha256_file(resolve(f)) for f in files},host_rss_gib=32,max_case_seconds=180,
  scope='Synthetic micro model8GPU1/7/9 source grouping and category NLL. Not actual final-tokenizer/1B3B/rehearsal validation.')
 atomic_json(root/'contract.json',contract);Path('logs/source_probe_evaluation_v1.pid').write_text(str(os.getpid()))
 out=root/'evaluation';config=root/'config.json';atomic_json(config,dict(model='micro',vocab_size=128,seq_len=32,micro_bsz=1,global_tokens=512,max_steps=2,warmup=1,
  lr=.001,decay_frac=.5,compile=False,mix=str(mix),holdout=[str(hold)],eval_every=1,ckpt_every=0,save_final=False,log_every=1,keep_ckpts=3,sampler_version=2,out_dir=str(out)))
 cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity'];env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TMPDIR=str(scratch),TORCHINDUCTOR_COMPILE_THREADS='1')
 cmd=['taskset','-c',','.join(map(str,cpus)),'torchrun','--standalone','--nproc_per_node=8',str(Path(__file__).resolve()),'--worker',str(config)]
 execution=run_cases(root,[dict(name='source_evaluation',out=str(out),cmd=cmd,env=env)],max_seconds=180,max_rss_gib=32)
 reports=[json.loads((out/f'rank{r}.json').read_text()) for r in range(8)]
 certificate(root,dict(status='passed_bounded_synthetic',execution=execution,reports=reports,scope=contract['scope']),contract)


if __name__=='__main__':main()
