"""Source-aware deterministic rehearsal hooks without changing the native trainer.

Sampling is observed at the immutable memmap slice; the original RNG and batch
implementation remain in use. Only optimizer-committed batches enter the ledger.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,identity,sha256_file


class SourceLedger:
    def __init__(self,names):
        self.names=list(names);self.pending={};self.current=None;self.steps=[];self.generations=[];self.optimizer_initializations=0

    def begin(self,step):
        if self.current is not None and self.current!=step:raise ValueError('Uncommitted sampled update')
        if self.steps and step+1<=self.steps[-1]['step']:raise ValueError('Sampler step did not advance')
        self.current=step

    def record(self,source):
        if self.current is None or source not in self.names:raise ValueError('Unbound sampled source')
        self.pending[source]=self.pending.get(source,0)+1

    def commit(self):
        if self.current is None or not self.pending:raise ValueError('Optimizer update has no sampled batch')
        self.steps.append(dict(step=self.current+1,source_sequences=dict(self.pending)))
        self.current=None;self.pending={}


class ObservedTokens:
    def __init__(self,tokens,source,ledger):self.tokens=tokens;self.source=source;self.ledger=ledger
    def __getitem__(self,key):
        result=self.tokens[key]
        self.ledger.record(self.source)
        return result


def step_and_record(original,ledger,optimizer,*args,**kwargs):
    sampled=ledger.current is not None and bool(ledger.pending)
    # DCP _init_optim_state invokes step(closure=None) with zero LR/grads before
    # loading the checkpoint. It creates optimizer buffers, not a data update.
    initializing=(not sampled and not ledger.steps and not optimizer.state and kwargs.get('closure',False) is None
                  and all(float(group['lr'])==0 for group in optimizer.param_groups))
    if not sampled and not initializing:raise ValueError('Unexpected optimizer step without sampled input or zero-LR DCP initialization')
    result=original(optimizer,*args,**kwargs)
    if sampled:ledger.commit()
    else:ledger.optimizer_initializations+=1
    return result


def observed_sampler(base,ledger):
    class Sampler(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.names!=ledger.names:raise ValueError('Observed sampler source order differs')
            self.shards=[[(ObservedTokens(tokens,name,ledger),n) for tokens,n in shards] for name,shards in zip(self.names,self.shards)]
        def batch(self,step,rank,micro,bsz):
            ledger.begin(step)
            return super().batch(step,rank,micro,bsz)
    return Sampler


def aggregate_ledgers(paths,*,world,steps,seq_len,global_tokens,sources):
    rows={};contracts=set();generation_checks={};attempted_sequences=0
    for path in paths:
        r=json.loads(Path(path).read_text());rank=r['rank']
        if not r['steps'] and not r['generations']:continue
        if type(rank) is not int or not 0<=rank<world or not r['contract_id']:raise ValueError('Invalid rehearsal ledger rank/contract')
        contracts.add(r['contract_id'])
        for row in r['steps']:
            step=row['step'];counts=row['source_sequences']
            if type(step) is not int or not 1<=step<=steps or not counts or set(counts)-set(sources):raise ValueError('Invalid sampled update/source')
            if any(type(v) is not int or v<1 for v in counts.values()) or sum(counts.values())*seq_len*world!=global_tokens:
                raise ValueError('Committed source tokens differ from update geometry')
            key=rank,step;previous=rows.setdefault(key,counts)
            if previous!=counts:raise ValueError('Replayed update source choices changed')
            attempted_sequences+=sum(counts.values())
        for generated in r['generations']:
            key=generated['step'],rank;previous=generation_checks.setdefault(key,generated['sha256'])
            if previous!=generated['sha256']:raise ValueError('Replayed fixed generation changed')
    if len(contracts)!=1 or len(rows)!=world*steps:raise ValueError('Incomplete committed update/rank coverage')
    totals={s:0 for s in sources}
    for counts in rows.values():
        for s,n in counts.items():totals[s]+=n*seq_len
    if sum(totals.values())!=steps*global_tokens:raise ValueError('Actual source token total differs')
    for step in [4,2500,5000]:
        if step<=steps and (set(rank for s,rank in generation_checks if s==step)!=set(range(world)) or
                           len({v for (s,_),v in generation_checks.items() if s==step})!=1):
            raise ValueError('Fixed generation rank coverage/agreement differs')
    return dict(contract_id=next(iter(contracts)),source_tokens=totals,source_fractions={s:n/(steps*global_tokens) for s,n in totals.items()},
                committed_tokens=steps*global_tokens,attempted_committed_tokens_including_replays=attempted_sequences*seq_len,
                replayed_tokens=attempted_sequences*seq_len-steps*global_tokens,
                generation_sha256_by_step={str(s):v for (s,rank),v in generation_checks.items() if rank==0},
                scope='Exact source counts of unique optimizer-committed step/rank batches; replayed successful updates counted separately.')


def distributed_generate(model,fwd,prompt,*,eos_id,context_length,max_new_tokens,device):
    import torch
    import torch.distributed as dist
    from train.runtime import require_finite
    if not prompt or max_new_tokens<1 or context_length<1:raise ValueError('Invalid distributed generation geometry')
    tokens=list(prompt);generated=[];previous=model.training;model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x=torch.tensor([tokens[-context_length:]],dtype=torch.long,device=device)
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16):logits=fwd(x)
                require_finite(logits,'fixed generation logits',device)
                token=logits[0,-1].argmax().reshape(1)
                # Broadcast every decision, including EOS, so all FSDP ranks
                # execute exactly the same number of forward collectives.
                dist.broadcast(token,src=0);value=int(token.item());generated.append(value);tokens.append(value)
                if value==eos_id:break
    finally:model.train(previous)
    return dict(generated_ids=generated,stopped_on_eos=generated[-1]==eos_id,
                scope='FSDP-balanced bf16 greedy base probe; no KV cache. Separate from native fp32 full-target PPL.')


def worker(root,config_path,attempt):
    import os
    import torch
    import torch.distributed as dist
    from train.deterministic_train import configure
    from train import train as trainer
    from train.parallel_contract import training_contract as parallel_contract
    from train.probe import verify_probe,evaluate_sources
    from tokenizers import Tokenizer
    root=resolve(root);contract=json.loads((root/'contract.json').read_text());config_path=resolve(config_path)
    if attempt!=Path(attempt).name or attempt in ('.','..'):raise ValueError('Invalid rehearsal attempt name')
    allowed={root/'early.json',root/'main.json'}
    if config_path not in allowed:raise ValueError('Unknown rehearsal phase configuration')
    for path,digest in {**contract['code'],**contract['evidence']}.items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Rehearsal implementation/input changed:'+path)
    config=json.loads(config_path.read_text())
    if config['model']!='1b' or config['max_steps']!=5000 or config['compile'] or config['keep_ckpts']!=5001:
        raise ValueError('Rehearsal worker is restricted to non-pruning1B5000step configuration')
    def fresh_contract(c,world,hold_files):
        return parallel_contract(c,world,hold_files,expected_files=json.loads((root/'expected_tokens.json').read_text()),
                                 observe=lambda metrics:atomic_json(root/(attempt+'-input_hash.json'),metrics))
    trainer.training_contract=fresh_contract
    configure()
    probe=verify_probe(root/'probe');mix=json.loads(resolve(config['mix']).read_text())
    if config['holdout']!=probe['holdout'] or config['seq_len']!=probe['seq_len'] or mix['tokenizer_sha256']!=probe['tokenizer_sha256']:
        raise ValueError('Rehearsal probe configuration differs')
    ledger=SourceLedger(mix['sources']);trainer.MixSampler=observed_sampler(trainer.MixSampler,ledger)
    rank=int(os.environ['RANK'])
    def persist(success=False,checkpoint_step=None):
        cp=resolve(config['out_dir'])/'training_contract.json'
        cid=json.loads(cp.read_text())['id'] if cp.exists() else None
        atomic_json(root/'ledgers'/(attempt+f'-rank{rank}.json'),dict(rank=rank,success=success,contract_id=cid,steps=ledger.steps,
                                                                 generations=ledger.generations,checkpoint_step=checkpoint_step,
                                                                 optimizer_initializations=ledger.optimizer_initializations))
    original_step=torch.optim.AdamW.step
    def optimizer_step(optimizer,*args,**kwargs):
        return step_and_record(original_step,ledger,optimizer,*args,**kwargs)
    torch.optim.AdamW.step=optimizer_step
    original_contract=trainer.training_contract
    def training_contract(c,world,hold_files):
        value=original_contract(c,world,hold_files);value.pop('id')
        value['rehearsal_evaluation']=dict(probe_sha256=sha256_file(root/'probe/probe.json'),
                                         prompts_sha256=sha256_file(resolve(contract['prompts'])),code_sha256=sha256_file(Path(__file__)),
                                         input_validation_code_sha256=sha256_file(resolve('train/parallel_contract.py')))
        value['id']=identity(value);return value
    trainer.training_contract=training_contract
    prompts=[json.loads(s) for s in resolve(contract['prompts']).read_text().splitlines() if s.strip()]
    if len(prompts)!=6 or len({r['id'] for r in prompts})!=6 or {r['category'] for r in prompts}!={'ko','en'}:raise ValueError('Fixed prompt coverage differs')
    tokenizer=Tokenizer.from_file(str(resolve(contract['tokenizer'])));eos=tokenizer.token_to_id('<|eos|>')
    if eos is None:raise ValueError('Final tokenizer lacks EOS')
    original_eval=trainer.evaluate_holdout
    def evaluate(model,fwd,hold,device):
        start=time.monotonic()
        def timed(*args):
            t=time.monotonic();result=original_eval(*args);return dict(result,elapsed_seconds=time.monotonic()-t)
        result=evaluate_sources(timed,model,fwd,hold,device,probe);result['probe_eval_seconds']=time.monotonic()-start
        step=ledger.steps[-1]['step']
        if step in [4,2500,5000]:
            t=time.monotonic();generated=[]
            for prompt in prompts:
                output=distributed_generate(model,fwd,tokenizer.encode(prompt['text'],add_special_tokens=False).ids,
                                            eos_id=eos,context_length=config['seq_len'],max_new_tokens=64,device=device)
                generated.append(dict(id=prompt['id'],category=prompt['category'],prompt=prompt['text'],
                                      text=tokenizer.decode(output['generated_ids'],skip_special_tokens=False),**output))
            result['fixed_generations']=generated;result['generation_seconds']=time.monotonic()-t
            ledger.generations.append(dict(step=step,sha256=identity(generated)))
        return result
    trainer.evaluate_holdout=evaluate
    original_save=trainer.dcp.save
    def save(*args,**kwargs):
        start=time.monotonic();result=original_save(*args,**kwargs)
        payload_seconds=time.monotonic()-start;snapshot_start=time.monotonic()
        # Persist all source receipts before the trainer publishes .complete.
        # A killed job can then recover every committed checkpoint prefix.
        persist(checkpoint_step=ledger.steps[-1]['step'])
        seconds=torch.tensor([payload_seconds,time.monotonic()-snapshot_start],dtype=torch.float64,device=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
        dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
        if dist.get_rank()==0:
            with (root/'checkpoint_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(checkpoint=Path(kwargs['checkpoint_id']).name,payload_seconds_max_rank=float(seconds[0].item()),
                                             source_ledger_seconds_max_rank=float(seconds[1].item()),
                                             scope='DCP payload and source-ledger journal measured separately; excludes subsequent per-rank RNG and metadata publication'))+'\n')
        return result
    trainer.dcp.save=save
    success=False
    try:
        sys.argv=['train/rehearsal.py','--config',str(config_path)];trainer.main();success=True
    finally:
        persist(success)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--config',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    worker(a.run_dir,a.config,a.attempt)
