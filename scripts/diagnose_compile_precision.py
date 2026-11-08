"""Paired FP32/bf16 and production AdamW groups; single GPU diagnostic only."""
import argparse,json,os,sys,subprocess,fcntl,time,math
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file

def groups(model):
    return [{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},
            {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
def worker(args):
    import torch
    from model.model import Transformer,get_config
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.manual_seed(711)
    c=get_config(args.model,max_seq_len=4096,**({'vocab_size':128} if args.model=='micro' else {}));m=Transformer(c).cuda()
    initial={n:p.detach().clone() for n,p in m.named_parameters()};torch.manual_seed(912)
    x=torch.randint(0,c.vocab_size,(1,32),device='cuda');y=torch.randint(0,c.vocab_size,x.shape,device='cuda');compiled=torch.compile(m)
    out=Path(args.output)
    def progress(phase):atomic_json(out/'state.json',dict(phase=phase,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat()));print(phase,flush=True)
    def metric(a,b):
        d=(a.float()-b.float());return dict(diff_sq=d.double().square().sum().item(),ref_sq=b.double().square().sum().item(),max_abs=d.abs().max().item(),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    def total(items):return dict(rel_l2=math.sqrt(sum(z['diff_sq'] for z in items)/max(sum(z['ref_sq'] for z in items),1e-30)),max_abs=max(z['max_abs'] for z in items),finite=all(z['finite'] for z in items))
    def reset():
        with torch.no_grad():
            for n,p in m.named_parameters():p.copy_(initial[n])
        m.zero_grad(set_to_none=True);torch.manual_seed(513)
    def once(fn,amp,name):
        progress(name);reset();opt=torch.optim.AdamW(groups(m),lr=3e-4,betas=(.9,.95),eps=1e-8,fused=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            logits=fn(x);loss=torch.nn.functional.cross_entropy(logits.float().reshape(-1,c.vocab_size),y.flatten())
        loss.backward();grads={n:p.grad.detach().clone() for n,p in m.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        updates={n:p.detach()-initial[n] for n,p in m.named_parameters()}
        r=dict(logits=logits.detach().float(),loss=loss.item(),gradient=grads,update=updates,norm=norm.item())
        del opt,loss,logits;m.zero_grad(set_to_none=True);torch.cuda.empty_cache();return r
    def compare(a,b):
        gstats=[metric(a['gradient'][n],b['gradient'][n]) for n in initial]
        per={n:total([v]) for n,v in zip(initial,gstats)}
        # Global sign-disagreement count and mass locate small-gradient sensitivity.
        count=0;flips=0;flipmass=0.;allmass=0.
        for n in initial:
            ga,gb=a['gradient'][n],b['gradient'][n];flip=(ga*gb<0);count+=gb.numel();flips+=flip.sum().item();flipmass+=gb[flip].double().square().sum().item();allmass+=gb.double().square().sum().item()
        return dict(logits=total([metric(a['logits'],b['logits'])]),loss_abs=abs(a['loss']-b['loss']),gradient=total(gstats),update=total([metric(a['update'][n],b['update'][n]) for n in initial]),grad_norms=[a['norm'],b['norm']],gradient_sign_flip_fraction=flips/count,sign_flip_reference_gradient_energy_fraction=flipmass/max(allmass,1e-30),worst_gradient_parameters=sorted(per.items(),key=lambda kv:kv[1]['rel_l2'],reverse=True)[:12])
    report=dict(model=args.model,world=1,sequence=32,seed=711,optimizer='production ndim>=2 decay0.1 / ndim<2 decay0; AdamW fused betas.9/.95 eps1e-8 lr3e-4 clip1',scope='Single-GPU one-step diagnostics; not FSDP/long context adoption proof',comparisons={})
    base=once(m,False,'fp32_eager');candidate=once(compiled,False,'fp32_compiled');report['comparisons']['fp32_compile_vs_eager']=compare(candidate,base);del candidate;atomic_json(out/'report.json',report)
    bf=once(m,True,'bf16_eager');report['comparisons']['bf16_eager_vs_fp32_eager']=compare(bf,base);del base;atomic_json(out/'report.json',report)
    candidate=once(compiled,True,'bf16_compiled');report['comparisons']['bf16_compile_vs_eager']=compare(candidate,bf);del candidate,bf;torch.cuda.empty_cache();atomic_json(out/'report.json',report)
    # Extra outputs can change fusion: explicitly diagnostic trace, not original graph proof.
    reset();progress('layer_trace')
    def trace(ids):
        h=m.tok(ids);states=[h];cos,sin=m.cos.to(h.device),m.sin.to(h.device)
        for b in m.blocks:h=b(h,cos,sin);states.append(h)
        states.append(m.norm(h));states.append(m.out(states[-1]));return tuple(states)
    traced=torch.compile(trace)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        e=trace(x);cc=traced(x);report['layer_trace']=[dict(stage=('embedding' if i==0 else 'block'+str(i-1) if i<=c.n_layers else 'final_norm' if i==c.n_layers+1 else 'logits'),**total([metric(a,b)])) for i,(a,b) in enumerate(zip(cc,e))]
        # Compare production modules on exactly the same first-block inputs.
        h=e[0];b=m.blocks[0];norm=b.attn_norm(h);att=b.attn(norm,m.cos,m.sin);res=h+att;norm2=b.ffn_norm(res)
        checks=[('attn_norm',b.attn_norm,(h,)),('attention',b.attn,(norm,m.cos,m.sin)),('ffn_norm',b.ffn_norm,(res,)),('mlp',b.ffn,(norm2,))]
        report['same_input_first_block']={name:total([metric(torch.compile(fn)(*inputs),fn(*inputs))]) for name,fn,inputs in checks}
        report['trace_vs_uninstrumented_compiled_logits']=total([metric(cc[-1],compiled(x))])
    report['trace_caveat']='Returning intermediates can change compiler fusion. Local module and trace results are localization evidence, not identical executable graphs.'
    report['status']='diagnostic_complete';report['max_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    atomic_json(out/'report.json',report);progress('complete')
def main(args):
    root=Path(PROJECT_ROOT);out=root/args.output;out.mkdir(parents=True,exist_ok=False)
    with (root/'runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=['scripts/diagnose_compile_precision.py','model/model.py','train/train.py','common/data_contracts.py']
        contract=dict(code={p:sha256_file(root/p) for p in files},scope='micro/1B/3B singleGPU seq32 production groups')
        atomic_json(out/'contract.json',contract);summaries=[];env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2')
        for model in ['micro','1b','3b']:
            case=out/model;case.mkdir();atomic_json(out/'state.json',dict(status='running',case=model));print('START',model,flush=True)
            with (case/'console.log').open('w') as log:p=subprocess.run(['timeout','--kill-after=30s','900s',sys.executable,'scripts/diagnose_compile_precision.py','--worker','--model',model,'--output',str(case)],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root)
            if p.returncode:atomic_json(out/'state.json',dict(status='error',case=model,returncode=p.returncode));raise RuntimeError(model)
            summaries.append(json.loads((case/'report.json').read_text()));atomic_json(out/'summary.json',summaries);print('DONE',model,flush=True)
        assert all(sha256_file(root/p)==h for p,h in contract['code'].items())
        atomic_json(out/'.done',dict(summary_sha256=sha256_file(out/'summary.json'),contract_sha256=sha256_file(out/'contract.json')));atomic_json(out/'state.json',dict(status='completed',cases=3))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--model',default='micro');a=ap.parse_args();worker(a) if a.worker else main(a)
