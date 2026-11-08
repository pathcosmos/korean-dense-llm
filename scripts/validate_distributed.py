"""Bounded 8-GPU C2 regression suite; isolated synthetic inputs and micro model.
Run: python scripts/validate_distributed.py --run-dir runs/c2_<new_name>
The worker injects faults only inside this test harness, never in production config.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json, sha256_file


def event(state, **extra):
    with (Path(PROJECT_ROOT)/'logs/pipeline_events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage='c2_distributed',
                                    event='validation_state',state=state,**extra))+'\n')


def loss_fault_after_microbatches(config,world):
    values=[world,config['seq_len'],config['micro_bsz'],config['global_tokens']]
    if any(type(v) is not int or v<1 for v in values):raise ValueError('Positive integer batch geometry required')
    denominator=world*config['seq_len']*config['micro_bsz']
    if config['global_tokens']%denominator:raise ValueError('Global tokens must divide batch geometry')
    return config['global_tokens']//denominator+1


def worker(config_path, mode):
    import torch
    import torch.distributed as dist
    import train.train as trainer
    from train.runtime import local_tensor
    rank=int(os.environ['RANK']);config=json.loads(Path(config_path).read_text());out=Path(config['out_dir'])
    out.mkdir(parents=True,exist_ok=True)
    steps=[0];opt_ref=[]
    original_step=torch.optim.AdamW.step
    def counted_step(self,*args,**kwargs):
        result=original_step(self,*args,**kwargs);steps[0]+=1;opt_ref[:]=[self]
        return result
    torch.optim.AdamW.step=counted_step
    original_gradient=trainer.ensure_finite_gradients;calls=[0]
    def gradient_check(model,device):
        calls[0]+=1
        if mode=='nan_gradient' and calls[0]==2 and rank==3:
            local_tensor(next(p.grad for p in model.parameters() if p.grad is not None)).flatten()[0]=float('nan')
        return original_gradient(model,device)
    trainer.ensure_finite_gradients=gradient_check
    if mode=='nan_loss':
        fault_at=loss_fault_after_microbatches(config,int(os.environ['WORLD_SIZE']))
        base=trainer.Transformer
        class FaultModel(base):
            seen=0
            def forward(self,idx,targets=None):
                loss=super().forward(idx,targets)
                if self.training and targets is not None:
                    self.seen+=1
                    if self.seen==fault_at and rank==3:loss=loss*float('nan')
                return loss
        trainer.Transformer=FaultModel
    if mode=='eval_suite':
        original_eval=trainer.evaluate_holdout
        def eval_suite(model,fwd,hold,device):
            results=[]
            for count in [1,7,9]:
                subset=copy.copy(hold);subset.wins=hold.wins[:count]
                result=original_eval(model,fwd,subset,device)
                # Independent serial reference: every rank executes the same full list.
                previous=model.training;model.eval();total=0.0
                with torch.no_grad():
                    for w in subset.wins:
                        w=w[None].to(device)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            loss=fwd(w[:,:-1].contiguous(),w[:,1:].contiguous())
                        total+=float(loss.item())
                model.train(previous)
                error=abs(result['holdout_loss']-total/count)
                if error>1e-6 or result['valid_tokens']!=count*config['seq_len']:
                    raise AssertionError(f'Uneven evaluation mismatch: {result}, reference={total/count}')
                results.append(dict(count=count,reference_error=error,**result))
            if rank==0:
                with (out/'eval_checks.jsonl').open('a') as stream:stream.write(json.dumps(results)+'\n')
            return results[-1]
        trainer.evaluate_holdout=eval_suite
    sys.argv=['train/train.py','--config',config_path]
    try:
        trainer.main()
    except FloatingPointError as exc:
        if mode not in ('nan_loss','nan_gradient'):raise
        atomic_json(out/f'fault_rank{rank}.json',dict(optimizer_updates=steps[0],error=str(exc)))
        if steps[0]!=1:raise AssertionError('Fault reached optimizer update')
        return
    if mode in ('nan_loss','nan_gradient'):raise AssertionError('Injected fault did not abort')


def checkpoint_compare(left,right):
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    converted=[]
    for directory in [left,right]:
        file=directory.parent/(directory.name+'.comparison.pt')
        dcp_to_torch_save(str(directory),str(file));converted.append(torch.load(file,map_location='cpu',weights_only=True))
    report=dict(tensors=0,max_abs=0.0,exact=True)
    def compare(a,b):
        if isinstance(a,torch.Tensor):
            if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Tensor metadata differs')
            report['tensors']+=1
            diff=(a.double()-b.double()).abs().max().item() if a.numel() else 0.0
            report['max_abs']=max(report['max_abs'],diff)
            report['exact'] &= torch.equal(a,b)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        elif isinstance(a,dict):
            if a.keys()!=b.keys():raise AssertionError('State keys differ')
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):raise AssertionError('State lengths differ')
            for x,y in zip(a,b):compare(x,y)
        elif a!=b:raise AssertionError(f'State values differ: {a} != {b}')
    compare(*converted)
    for rank in range(8):
        a=torch.load(left/f'rng_rank{rank}.pt',weights_only=True)
        b=torch.load(right/f'rng_rank{rank}.pt',weights_only=True)
        if any(not torch.equal(a[k],b[k]) for k in a):raise AssertionError('Rank RNG differs')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir');ap.add_argument('--worker');ap.add_argument('--mode',default='normal')
    ap.add_argument('--compile',action='store_true');ap.add_argument('--case-timeout',type=int)
    args=ap.parse_args()
    if args.worker:return worker(args.worker,args.mode)
    if not args.run_dir:ap.error('--run-dir required')
    timeout=args.case_timeout or (900 if args.compile else 240)
    if timeout<1:ap.error('--case-timeout must be positive')
    root=Path(PROJECT_ROOT);run=(root/args.run_dir).resolve();run.relative_to(root/'runs')
    run.mkdir(exist_ok=False,parents=True)
    lock=(root/'runs/.gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('GPU compute processes already active')
    (run/'pid').write_text(str(os.getpid()))
    import numpy as np
    rng=np.random.default_rng(123)
    data=run/'train.bin';rng.integers(0,128,size=4096,dtype=np.uint16).tofile(data)
    hold=run/'hold.bin';rng.integers(0,128,size=9*33,dtype=np.uint16).tofile(hold)
    mix=run/'mix.json';atomic_json(mix,dict(sampler_version=2,tokenizer_sha256='synthetic-v1',
                    sources={'synthetic':dict(weight=1,files=[str(data)])},holdout=[str(hold)]))
    base=dict(model='micro',vocab_size=128,seq_len=32,micro_bsz=1,global_tokens=512,
              max_steps=4,warmup=1,lr=.001,decay_frac=.5,compile=args.compile,mix=str(mix),holdout=[str(hold)],
              log_every=1,eval_every=2,ckpt_every=2,keep_ckpts=3,sampler_version=2)
    env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='2')
    results=[]
    def launch(name,config,mode='normal',expected_error=None):
        path=run/(name+'.json');atomic_json(path,config)
        start=time.monotonic();print('START',name,flush=True);event('running',case=name,run_dir=str(run))
        with (run/(name+'.log')).open('w') as stream:
            entry = ['train/train.py', '--config', str(path)] if mode == 'normal' else [
                'scripts/validate_distributed.py', '--worker', str(path), '--mode', mode]
            proc=subprocess.run(['timeout','--signal=TERM','--kill-after=15s',f'{timeout}s','torchrun','--standalone',
                '--nproc_per_node=8', *entry],
                cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        output=(run/(name+'.log')).read_text()
        success=proc.returncode==0 if expected_error is None else proc.returncode not in (0,124,137) and expected_error in output
        if not success:raise RuntimeError(f'{name} failed rc={proc.returncode}; inspect {run/(name+".log")}')
        result=dict(case=name,returncode=proc.returncode,wall_s=time.monotonic()-start,expected_error=expected_error,compile=args.compile,timeout_s=timeout)
        results.append(result);atomic_json(run/'progress.json',results);print('DONE',json.dumps(result),flush=True)
    try:
        baseline=dict(base,out_dir=str(run/'continuous'))
        launch('continuous',baseline,'eval_suite')
        resumed=dict(base,out_dir=str(run/'resumed'))
        launch('stop_at_2',dict(resumed,stop_after=2))
        incomplete=run/'resumed/ckpt_0000003';incomplete.mkdir();(incomplete/'partial').write_text('interrupted fixture')
        launch('resume',resumed)
        comparison=checkpoint_compare(run/'continuous/ckpt_0000004',run/'resumed/ckpt_0000004')
        rows=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
        a=[x for x in rows(run/'continuous/log.jsonl') if 'loss' in x]
        b=[x for x in rows(run/'resumed/log.jsonl') if 'loss' in x]
        if len(a)!=4 or len(b)!=4:raise AssertionError('Unexpected training step count')
        for x,y in zip(a,b):
            if x['step']!=y['step'] or x['lr']!=y['lr'] or abs(x['loss']-y['loss'])>1e-6:
                raise AssertionError('Continuous/resume trajectory differs')
        # Explicitly compare every rank/micro batch at the resumed steps.
        from train.data import MixSampler
        one=MixSampler(str(mix),32,version=2);two=MixSampler(str(mix),32,version=2)
        for step in [2,3]:
            for rank in range(8):
                for micro in range(2):
                    for x,y in zip(one.batch(step,rank,micro,1),two.batch(step,rank,micro,1)):
                        if not (x==y).all():raise AssertionError('Resumed batch differs')
        changed=copy.deepcopy(json.loads(mix.read_text()));changed['tokenizer_sha256']='synthetic-v2'
        changed_mix=run/'changed_mix.json';atomic_json(changed_mix,changed)
        launch('reject_changed_tokenizer',dict(resumed,mix=str(changed_mix)),expected_error='Training identity mismatch')
        for mode in ['nan_loss','nan_gradient']:
            out=run/mode
            launch(mode,dict(base,out_dir=str(out),ckpt_every=1,eval_every=0),mode)
            faults=[json.loads((out/f'fault_rank{rank}.json').read_text()) for rank in range(8)]
            if not all(x['optimizer_updates']==1 for x in faults):raise AssertionError('Fault not synchronized')
            if sorted(p.name for p in out.glob('ckpt_*'))!=['ckpt_0000001']:raise AssertionError('Fault published a checkpoint')
        report=dict(status='passed',scope=f'synthetic micro model, 8 GPU, compile={args.compile}',cases=results,
                    continuous_resume=comparison,uneven_eval_windows=[1,7,9],
                    remaining=['real Korean/English holdout','production-size trajectory'] + ([] if args.compile else ['compile=true resume']),
                    code={p:sha256_file(root/p) for p in ['train/__init__.py','train/train.py','train/runtime.py','train/data.py','model/model.py','scripts/validate_distributed.py']})
        atomic_json(run/'summary.json',report);(run/'.done').touch();event('complete',run_dir=str(run))
        print('PASS',json.dumps(report),flush=True)
    except Exception as exc:
        atomic_json(run/'failure.json',dict(error=str(exc)));event('error',run_dir=str(run),error=str(exc));raise


if __name__=='__main__':main()
