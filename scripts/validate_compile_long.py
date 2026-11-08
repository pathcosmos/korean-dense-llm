"""Isolated candidate validation: paired seeds/long contexts and FSDP8.
CPU reference tensors bound GPU memory. This is synthetic validation, not training.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file
from scripts.validate_compile_pair import TOLERANCES,judge


def tensor_metric(actual,expected):
    import torch
    a=actual.float();b=expected.float();d=a-b
    return [d.double().square().sum().item(),b.double().square().sum().item(),
            d.abs().max().item(),float(bool(torch.isfinite(a).all() and torch.isfinite(b).all()))]


def summarize_metrics(items):
    return dict(rel_l2=math.sqrt(sum(x[0] for x in items)/max(sum(x[1] for x in items),1e-30)),
                max_abs=max(x[2] for x in items),finite=all(x[3] for x in items))


def worker(a):
    os.environ['TORCHINDUCTOR_EMULATE_PRECISION_CASTS']='1'
    import torch
    import torch.distributed as dist
    import torch._inductor.config as ic
    from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
    from model.model import Transformer,get_config,RMSNorm,Attention
    from train.runtime import local_tensor
    from scripts.diagnose_compile_precision import groups
    assert ic.emulate_precision_casts
    RMSNorm.forward=torch.compiler.disable(RMSNorm.forward)
    Attention.forward=torch.compiler.disable(Attention.forward)
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.set_num_threads(1 if world>1 else 2);torch.cuda.set_device(local);dev=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world>1:dist.init_process_group('nccl')
    out=resolve(a.output)
    cfg=get_config(a.model,max_seq_len=a.sequence,**({'vocab_size':128} if a.model=='micro' else {}))
    if world>1:
        with torch.device('meta'):m=Transformer(cfg)
        policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
        for b in m.blocks:fully_shard(b,mp_policy=policy)
        fully_shard(m,mp_policy=policy);m.to_empty(device=dev)
        torch.manual_seed(a.seed+rank)
        with torch.no_grad():m.init_weights()
    else:
        torch.manual_seed(a.seed);m=Transformer(cfg).to(dev)
    initial={n:local_tensor(p).detach().cpu().clone() for n,p in m.named_parameters()}
    torch.manual_seed(a.seed+201+rank)
    x=torch.randint(0,cfg.vocab_size,(1,a.sequence),device=dev);y=torch.randint(0,cfg.vocab_size,x.shape,device=dev)
    input_sha=hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
    compiled=torch.compile(m)

    def progress(phase):
        atomic_json(out/f'state_rank{rank}.json',dict(phase=phase,unix=time.time(),rank=rank))
        if rank==0:print(phase,flush=True)
    def finite(tensors,label):
        flag=torch.stack([torch.isfinite(local_tensor(t)).all() for t in tensors]).all().to(dtype=torch.int32,device=dev)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if not flag.item():raise FloatingPointError('Nonfinite '+label+' before optimizer')
    def reset():
        with torch.no_grad():
            for n,p in m.named_parameters():local_tensor(p).copy_(initial[n])
        m.zero_grad(set_to_none=True);torch.manual_seed(a.seed+400+rank)
    def optimizer():return torch.optim.AdamW(groups(m),lr=3e-4,betas=(.9,.95),eps=1e-8,fused=True)
    def step(fn,opt,collect=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=fn(x);loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,cfg.vocab_size),y.flatten())
        finite([loss],'loss');loss.backward()
        finite([p.grad for p in m.parameters() if p.grad is not None],'gradients')
        grads={n:local_tensor(p.grad).detach().cpu().clone() for n,p in m.named_parameters()} if collect else None
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);finite([norm],'norm');opt.step()
        result=None
        if collect:
            result=dict(logits=logits.detach().float().cpu(),loss=loss.item(),gradient=grads,
                        update={n:local_tensor(p).detach().cpu()-initial[n] for n,p in m.named_parameters()})
        opt.zero_grad(set_to_none=True)
        return result
    def once(fn,name):
        progress(name);reset();opt=optimizer();value=step(fn,opt,True)
        del opt;torch.cuda.empty_cache();return value
    def aggregate(items):
        values=torch.tensor([sum(x[0] for x in items),sum(x[1] for x in items),max(x[2] for x in items),float(all(x[3] for x in items))],dtype=torch.float64,device=dev)
        if world>1:
            dist.all_reduce(values[:2],op=dist.ReduceOp.SUM);dist.all_reduce(values[2:3],op=dist.ReduceOp.MAX);dist.all_reduce(values[3:],op=dist.ReduceOp.MIN)
        d,r,mx,ok=values.tolist();return dict(rel_l2=math.sqrt(d/max(r,1e-30)),max_abs=mx,finite=bool(ok))
    def compare(actual,ref):
        result=dict(logits=aggregate([tensor_metric(actual['logits'],ref['logits'])]),
                    gradient=aggregate([tensor_metric(actual['gradient'][n],ref['gradient'][n]) for n in initial]),
                    update=aggregate([tensor_metric(actual['update'][n],ref['update'][n]) for n in initial]))
        loss=torch.tensor(abs(actual['loss']-ref['loss']),device=dev,dtype=torch.float64)
        if world>1:dist.all_reduce(loss,op=dist.ReduceOp.MAX)
        result['loss_abs']=loss.item();return result
    torch.cuda.reset_peak_memory_stats()
    base=once(m,'eager_reference');repeat=once(m,'eager_repeat')
    rep=compare(repeat,base);del repeat
    warm=once(compiled,'compile_warmup');del warm
    candidate=once(compiled,'compiled_pair');comparison=compare(candidate,base);del candidate,base
    profiles={}
    for name,fn in [('eager',m),('candidate',compiled)]:
        progress('timing_'+name);reset();opt=optimizer();timings=[]
        for i in range(5):
            if world>1:dist.barrier()
            torch.cuda.synchronize();start=time.perf_counter();step(fn,opt);torch.cuda.synchronize()
            elapsed=torch.tensor(time.perf_counter()-start,device=dev,dtype=torch.float64)
            if world>1:dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            if i>=2:timings.append(elapsed.item())
        del opt;m.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        profiles[name]=dict(step_seconds=timings,mean_seconds=sum(timings)/len(timings),tokens_per_second=world*a.sequence/(sum(timings)/len(timings)))
    report=dict(status='passed_bounded_pair' if judge(rep) and judge(comparison) else 'needs_review',
                model=a.model,sequence=a.sequence,seed=a.seed,rank=rank,world=world,tolerances=TOLERANCES,
                eager_repeat=rep,compiled_vs_eager=comparison,profiles=profiles,input_sha256=input_sha,
                candidate='cast preservation + eager RMSNorm + eager Attention',production_modified=False,
                optimizer='production ndim>=2 wd.1 / ndim<2 wd0, AdamW fused .9/.95 eps1e-8 lr3e-4 clip1',
                torch=torch.__version__,max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                scope='Fixed synthetic paired one-step numerics plus 2warm/3timed steps. No actual-token checkpoint/resume or multi-step convergence approval.')
    atomic_json(out/f'rank{rank}.json',report);progress('complete')
    if world>1:dist.barrier();dist.destroy_process_group()


def main(a):
    from scripts.hardware_profile import inspect
    from scripts.parallel_job_support import run_cases,certificate
    root=resolve(a.output);root.mkdir(parents=True,exist_ok=False);hw=inspect();cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity']
    if len(hw['gpus'])<8 or hw['memory']['conservative_headroom_gib']<384:raise RuntimeError('GPU/host-memory budget unavailable')
    scratch=Path(hw['storage']['scratch_dir'])/'compile_long';scratch.mkdir(parents=True,exist_ok=True)
    files=['scripts/validate_compile_long.py','scripts/parallel_job_support.py','scripts/validate_compile_pair.py',
           'scripts/diagnose_compile_precision.py','model/model.py','train/runtime.py','train/train.py','common/data_contracts.py']
    contract=dict(code={p:sha256_file(resolve(p)) for p in files},hardware=hw,seeds=[711,712],
                  stages=[dict(world=1,sequence=512),dict(world=1,sequence=4096),dict(world=8,sequence=4096)],
                  max_case_group_seconds=900,max_case_group_host_rss_gib=256,cpu_affinity=cpus,
                  scope='Candidate diagnostic only; complete short-context stage before long context, then FSDP8. Numeric failure remains needs_review, not hidden by tolerance changes.')
    atomic_json(root/'contract.json',contract);summaries=[]
    with resolve('runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for world,sequence in [(1,512),(1,4096),(8,4096)]:
            groups=[[711,712]] if world==1 else [[711],[712]]
            for seeds in groups:
                cases=[]
                for index,seed in enumerate(seeds):
                    name=f'w{world}_s{sequence}_seed{seed}';out=root/name
                    env=dict(os.environ,OMP_NUM_THREADS='1' if world>1 else '2',NUMEXPR_NUM_THREADS='1',
                             TORCHINDUCTOR_COMPILE_THREADS='1' if world>1 else '2',
                             TORCHINDUCTOR_CACHE_DIR=str(scratch),TMPDIR=str(scratch),
                             CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))) if world>1 else str(index))
                    cmd=['taskset','-c',','.join(map(str,cpus))]
                    cmd+=['torchrun','--standalone','--nproc_per_node=8'] if world>1 else [sys.executable]
                    cmd += [str(Path(__file__).resolve()),'--worker','--model','3b','--sequence',str(sequence),'--seed',str(seed),'--output',str(out)]
                    cases.append(dict(name=name,out=str(out),cmd=cmd,env=env))
                execution=run_cases(root,cases,max_seconds=900,max_rss_gib=256)
                for case in cases:
                    reports=[json.loads((Path(case['out'])/f'rank{rank}.json').read_text()) for rank in range(world)]
                    summaries.append(dict(case=case['name'],reports=reports,execution=execution))
                atomic_json(root/'partial_results.json',summaries)
    certificate(root,dict(status='needs_review',cases=summaries,production_adoption='pending',scope=contract['scope']),contract)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--model',default='micro');p.add_argument('--sequence',type=int,default=32);p.add_argument('--seed',type=int,default=711)
    a=p.parse_args()
    if not 1<=a.sequence<=4096:p.error('Sequence must be1..4096')
    worker(a) if a.worker else main(a)
