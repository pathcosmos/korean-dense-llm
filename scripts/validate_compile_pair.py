"""Bounded paired eager/compile validation and synchronized stage timing."""
import argparse,json,os,sys,time,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file,identity

TOLERANCES=dict(logits_rel_l2=.03,gradient_rel_l2=.05,update_rel_l2=.10,loss_abs=.03)
def metrics(actual,expected):
    import torch
    # Compare on CPU in double for stable sums; called with bounded parameter tensors.
    d=actual.float()-expected.float()
    return dict(diff_sq=d.double().square().sum().item(),ref_sq=expected.double().square().sum().item(),max_abs=d.abs().max().item(),finite=bool(torch.isfinite(actual).all()))
def combine(items):
    import math
    return dict(rel_l2=math.sqrt(sum(x['diff_sq'] for x in items)/max(sum(x['ref_sq'] for x in items),1e-30)),max_abs=max(x['max_abs'] for x in items),finite=all(x['finite'] for x in items))
def judge(result):
    return all(result[k]['finite'] and result[k]['rel_l2']<=TOLERANCES[k+'_rel_l2'] for k in ['logits','gradient','update']) and result['loss_abs']<=TOLERANCES['loss_abs']
def worker(args):
    import torch,torch.distributed as dist
    from model.model import Transformer,get_config
    from train.runtime import local_tensor,require_finite,ensure_finite_gradients
    from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);dev=torch.device('cuda',local);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if world>1:dist.init_process_group('nccl')
    cfg=get_config(args.model,max_seq_len=4096,**({'vocab_size':128} if args.model=='micro' else {}))
    if world>1:
        with torch.device('meta'):m=Transformer(cfg)
        mp=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
        for b in m.blocks:fully_shard(b,mp_policy=mp)
        fully_shard(m,mp_policy=mp);m.to_empty(device=dev);torch.manual_seed(711+rank);m.init_weights()
    else:
        torch.manual_seed(711);m=Transformer(cfg).to(dev)
    initial={n:local_tensor(p).detach().cpu().clone() for n,p in m.named_parameters()}
    torch.manual_seed(912+rank);x=torch.randint(0,cfg.vocab_size,(1,32),device=dev);y=torch.randint(0,cfg.vocab_size,x.shape,device=dev)
    compiled=torch.compile(m)
    def reset():
        with torch.no_grad():
            for n,p in m.named_parameters():local_tensor(p).copy_(initial[n])
        m.zero_grad(set_to_none=True);torch.manual_seed(513+rank)
    def once(forward,amp):
        reset();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            logits=forward(x);loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,cfg.vocab_size),y.flatten())
        loss.backward();grads={n:local_tensor(p.grad).detach().cpu().clone() for n,p in m.named_parameters()}
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        updates={n:local_tensor(p).detach().cpu()-initial[n] for n,p in m.named_parameters()}
        r=dict(logits=logits.detach().float().cpu(),loss=loss.item(),gradient=grads,update=updates)
        del opt,logits,loss;m.zero_grad(set_to_none=True);torch.cuda.empty_cache();return r
    def compare(a,b):
        return dict(logits=combine([metrics(a['logits'],b['logits'])]),loss_abs=abs(a['loss']-b['loss']),gradient=combine([metrics(a['gradient'][n],b['gradient'][n]) for n in initial]),update=combine([metrics(a['update'][n],b['update'][n]) for n in initial]))
    report=dict(model=args.model,world=world,rank=rank,sequence=32,tolerances=TOLERANCES,initial_input_shared=True)
    eager=once(m,True);repeat=once(m,True);report['eager_repeat']=compare(repeat,eager);del repeat
    comp=once(compiled,True);report['compiled_vs_eager']=compare(comp,eager);del comp
    if args.model=='micro' and world==1:
        fp32=once(m,False);report['bf16_vs_fp32']=compare(eager,fp32);del fp32
    del eager
    report['status']='passed_bounded_pair' if judge(report['compiled_vs_eager']) and judge(report['eager_repeat']) else 'needs_review'
    # Stage timings include synchronization overhead; local maxima are reduced across ranks.
    if args.profile:
        reset();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        host=torch.randint(0,cfg.vocab_size,(2,4097));timings=[]
        def timed(name,fn,row):
            torch.cuda.synchronize();start=time.perf_counter();v=fn();torch.cuda.synchronize();row[name]=row.get(name,0)+time.perf_counter()-start;return v
        for step in range(5):
            row={}
            for micro in range(2):
                if world>1:m.set_requires_gradient_sync(micro==1);m.set_requires_all_reduce(micro==1)
                inputs=timed('h2d',lambda:(host[:,:-1].contiguous().to(dev),host[:,1:].contiguous().to(dev)),row)
                def fwd():
                    with torch.autocast('cuda',dtype=torch.bfloat16):return compiled(*inputs)/2
                loss=timed('forward',fwd,row)
                timed('loss_check',lambda:require_finite(loss,'loss',dev),row)
                timed('backward',loss.backward,row)
            timed('gradient_check',lambda:ensure_finite_gradients(m,dev),row)
            gn=timed('clip',lambda:torch.nn.utils.clip_grad_norm_(m.parameters(),1.),row)
            timed('norm_check',lambda:require_finite(gn,'norm',dev),row)
            timed('optimizer',lambda:(opt.step(),opt.zero_grad(set_to_none=True)),row)
            if step>=2:timings.append(row)
        report['profile']=dict(sequence=4096,micro_batch=2,accumulation=2,measured_steps=3,scope='Synchronized stage timings, overhead included; collectives included within stages',seconds={k:sum(r[k] for r in timings)/len(timings) for k in timings[0]})
    out=Path(args.output);atomic_json(out/f'rank{rank}.json',report)
    if world>1:dist.barrier();dist.destroy_process_group()
    print(json.dumps({k:v for k,v in report.items() if k not in ['bf16_vs_fp32']}),flush=True)
def main(args):
    root=Path(PROJECT_ROOT);out=root/args.output;out.mkdir(parents=True,exist_ok=False)
    with (root/'runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=['scripts/validate_compile_pair.py','model/model.py','train/runtime.py','train/data.py','common/data_contracts.py']
        contract=dict(code={p:sha256_file(root/p) for p in files},tolerances=TOLERANCES,scope='Paired seq32 full models; profile seq4096 FSDP only')
        atomic_json(out/'contract.json',contract)
        cases=[('micro',1,False),('1b',1,False),('3b',1,False),('3b',8,True)]
        env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2')
        summaries=[]
        for model,world,profile in cases:
            name=f'{model}_{world}gpu';case=out/name;case.mkdir();cmd=['timeout','--kill-after=30s','900s']
            cmd+=([sys.executable] if world==1 else ['torchrun','--standalone','--nproc_per_node=8'])
            cmd+=['scripts/validate_compile_pair.py','--worker','--model',model,'--output',str(case)]
            if profile:cmd+=['--profile']
            atomic_json(out/'state.json',dict(status='running',case=name,utc=datetime.now(timezone.utc).isoformat()));print('START',name,flush=True)
            with (case/'console.log').open('w') as log:p=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:atomic_json(out/'state.json',dict(status='error',case=name,returncode=p.returncode));raise RuntimeError(f'{name}: {p.returncode}')
            reports=[json.loads((case/f'rank{i}.json').read_text()) for i in range(world)]
            summaries.append(dict(case=name,status='passed_bounded_pair' if all(r['status']=='passed_bounded_pair' for r in reports) else 'needs_review',reports=reports));atomic_json(out/'summary.json',summaries);print('DONE',name,summaries[-1]['status'],flush=True)
        assert all(sha256_file(root/p)==h for p,h in contract['code'].items())
        atomic_json(out/'.done',dict(summary_sha256=sha256_file(out/'summary.json'),contract_sha256=sha256_file(out/'contract.json')))
        atomic_json(out/'state.json',dict(status='completed',cases=len(summaries),utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--model',default='micro');ap.add_argument('--output',required=True);a=ap.parse_args();worker(a) if a.worker else main(a)
