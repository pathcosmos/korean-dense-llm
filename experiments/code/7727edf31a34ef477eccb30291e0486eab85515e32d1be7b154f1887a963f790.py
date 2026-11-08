"""Synthetic micro8GPU fault checks; never final-tokenizer/model acceptance."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common.data_contracts import atomic_json,sha256_file,resolve
from scripts.hardware_profile import inspect
from scripts.parallel_job_support import run_cases,certificate
from scripts.validate_distributed import loss_fault_after_microbatches
root=Path(__file__).resolve().parent;hw=inspect()
if len(hw['gpus'])!=8 or hw['memory']['conservative_headroom_gib']<64:raise RuntimeError('Need8GPUs and64GiB spare for bounded regression')
scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=True)
lock=resolve('runs/.gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute active')
import numpy as np
train=root/'train.bin';np.random.default_rng(123).integers(0,128,4096,dtype=np.uint16).tofile(train)
mix=root/'mix.json';atomic_json(mix,dict(sampler_version=2,tokenizer_sha256='synthetic-accumulation-regression',sources=dict(synthetic=dict(weight=1,files=[str(train)])),holdout=[]))
cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity']
files=['scripts/validate_distributed.py','scripts/validate_actual_training.py','scripts/parallel_job_support.py','model/model.py','train/train.py','train/runtime.py','train/data.py']
contract=dict(hardware=hw,code={f:sha256_file(resolve(f)) for f in files},driver_sha256=sha256_file(Path(__file__)),accumulations=[1,4],world=8,model='micro',vocab=128,
 host_rss_gib=32,max_case_seconds=180,scope='Synthetic micro8GPU finite-fault timing regression only. Actual1B/3B/final-tokenizer acceptance remains pending.')
atomic_json(root/'contract.json',contract);Path('logs/accumulation_fault_regression_v1.pid').write_text(str(os.getpid()))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TMPDIR=str(scratch),TORCHINDUCTOR_COMPILE_THREADS='1')
results=[]
try:
 for accumulation,mode in [(1,'nan_loss'),(4,'nan_loss'),(4,'nan_gradient')]:
  name=f'a{accumulation}_{mode}';out=root/name;config=dict(model='micro',vocab_size=128,seq_len=32,micro_bsz=1,global_tokens=8*32*accumulation,
   max_steps=4,warmup=1,lr=.001,decay_frac=.5,compile=False,mix=str(mix),holdout=[],log_every=1,eval_every=0,ckpt_every=1,keep_ckpts=3,sampler_version=2,out_dir=str(out))
  path=root/(name+'.json');atomic_json(path,config)
  cmd=['taskset','-c',','.join(map(str,cpus)),'torchrun','--standalone','--nproc_per_node=8','scripts/validate_distributed.py','--worker',str(path),'--mode',mode]
  execution=run_cases(root,[dict(name=name,out=str(out),cmd=cmd,env=env)],max_seconds=180,max_rss_gib=32)
  faults=[json.loads((out/f'fault_rank{rank}.json').read_text()) for rank in range(8)]
  if any(r['optimizer_updates']!=1 for r in faults) or sorted(p.name for p in out.glob('ckpt_*'))!=['ckpt_0000001']:raise ValueError('Fault crossed optimizer/checkpoint boundary')
  results.append(dict(case=name,accumulation=accumulation,loss_fault_forward=loss_fault_after_microbatches(config,8),faults=faults,execution=execution));atomic_json(root/'partial_results.json',results)
 certificate(root,dict(status='passed_bounded_synthetic',results=results,scope=contract['scope']),contract)
except BaseException as exc:
 atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise
