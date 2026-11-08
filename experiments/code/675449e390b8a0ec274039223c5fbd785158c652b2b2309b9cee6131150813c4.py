"""Source-aware3B entry, sharing the original validated trainer and hooks.

The controller supplies immutable configuration/input/code bindings. Bounded
validation is limited to four updates without pruning. Main execution requires
an explicit authorization record from the user-facing launch controller.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, atomic_json, identity, sha256_file


def validate_configuration(config, contract):
    from scripts.plan_main_training import geometry
    g = geometry(config)
    if (config['model'] != '3b' or config['vocab_size'] != 65536 or
            config['seq_len'] != 4096 or config['micro_bsz'] != 4 or
            config['global_tokens'] != 2097152 or config['compile'] or
            config['sampler_version'] != 2 or config['lr'] != .0003):
        raise ValueError('Monitored entry requires validated3B eager deterministic geometry')
    purpose = contract['purpose']
    if purpose == 'bounded_validation':
        if config['max_steps'] != 4 or config['keep_ckpts'] != 5 or config['stop_after'] not in (2, None):
            raise ValueError('Bounded validation is restricted to4 non-pruning updates')
    elif purpose == 'main_training':
        if contract.get('main_training_authorized') is not True:
            raise ValueError('Main training requires separate user authorization')
        if config['max_steps'] != 57000:
            raise ValueError('Unreviewed main training horizon')
        if config['keep_ckpts'] <= config['max_steps'] and contract.get('new_checkpoint_pruning_authorized') is not True:
            raise ValueError('New-run checkpoint pruning requires explicit authorization')
    else:
        raise ValueError('Unknown monitored training purpose')
    if config['eval_every'] <= 0 or config['ckpt_every'] <= 0:
        raise ValueError('Monitored training requires evaluation and checkpoints')
    schedule = contract['generation_steps']
    if (not schedule or sorted(set(schedule)) != schedule or
            any(type(s) is not int or not 1 <= s <= config['max_steps'] or s % config['eval_every'] for s in schedule)):
        raise ValueError('Generation schedule must match evaluation boundaries')
    return g


def worker(root, config_path, attempt):
    import torch
    import torch.distributed as dist
    from train import train as trainer
    from train.deterministic_train import configure
    from train.parallel_contract import training_contract as parallel_contract
    from train.probe import verify_probe, evaluate_sources
    from train.rehearsal import SourceLedger, observed_sampler, step_and_record, distributed_generate
    from tokenizers import Tokenizer
    root, config_path = resolve(root), resolve(config_path)
    contract = json.loads((root/'contract.json').read_text())
    if attempt != Path(attempt).name or attempt in ('.', '..'):
        raise ValueError('Invalid monitored attempt')
    expected = contract['configurations'].get(config_path.name)
    if config_path.parent != root or expected != sha256_file(config_path):
        raise ValueError('Unbound monitored configuration')
    for path, digest in {**contract['code'], **contract['evidence']}.items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Monitored producer/input changed: '+path)
    config = json.loads(config_path.read_text()); validate_configuration(config, contract)
    probe = verify_probe(root/'probe')
    mix = json.loads(resolve(config['mix']).read_text())
    if (config['holdout'] != probe['holdout'] or config['seq_len'] != probe['seq_len'] or
            mix['tokenizer_sha256'] != probe['tokenizer_sha256']):
        raise ValueError('Monitored probe/tokenizer differs')
    def fresh(c, world, hold_files):
        if world != 8:
            raise ValueError('Actual monitored3B requires8GPU world')
        return parallel_contract(c, world, hold_files,
                    expected_files=json.loads((root/'expected_tokens.json').read_text()),
                    observe=lambda metrics:atomic_json(root/(attempt+'-input_hash.json'), metrics))
    trainer.training_contract = fresh; configure()
    original_contract = trainer.training_contract
    def training_contract(c, world, hold_files):
        value = original_contract(c, world, hold_files); value.pop('id')
        value['monitored_evaluation'] = dict(probe_sha256=sha256_file(root/'probe/probe.json'),
                    prompts_sha256=sha256_file(resolve(contract['prompts'])),
                    generation_steps=contract['generation_steps'], code_sha256=sha256_file(Path(__file__)),
                    source_hooks_sha256=sha256_file(resolve('train/rehearsal.py')))
        value['id'] = identity(value); return value
    trainer.training_contract = training_contract
    ledger = SourceLedger(mix['sources']); rank = int(os.environ['RANK'])
    trainer.MixSampler = observed_sampler(trainer.MixSampler, ledger)
    original_step = torch.optim.AdamW.step
    def step(optimizer, *args, **kwargs):
        return step_and_record(original_step, ledger, optimizer, *args, **kwargs)
    torch.optim.AdamW.step = step
    def persist(success=False, checkpoint_step=None):
        path = resolve(config['out_dir'])/'training_contract.json'
        cid = json.loads(path.read_text())['id'] if path.exists() else None
        atomic_json(root/'ledgers'/(attempt+f'-rank{rank}.json'),
                    dict(rank=rank, success=success, contract_id=cid, steps=ledger.steps,
                         generations=ledger.generations, checkpoint_step=checkpoint_step,
                         optimizer_initializations=ledger.optimizer_initializations))
    prompts = [json.loads(s) for s in resolve(contract['prompts']).read_text().splitlines() if s.strip()]
    if len(prompts) != 6 or len({r['id'] for r in prompts}) != 6 or {r['category'] for r in prompts} != {'ko', 'en'}:
        raise ValueError('Six bilingual fixed prompts required')
    tokenizer = Tokenizer.from_file(str(resolve(contract['tokenizer'])))
    eos = tokenizer.token_to_id('<|eos|>')
    if eos is None:
        raise ValueError('Final tokenizer lacks EOS')
    original_eval = trainer.evaluate_holdout
    def evaluate(model, fwd, hold, device):
        start = time.monotonic()
        def timed(*args):
            t = time.monotonic(); r = original_eval(*args)
            return dict(r, elapsed_seconds=time.monotonic()-t)
        result = evaluate_sources(timed, model, fwd, hold, device, probe)
        result['probe_eval_seconds'] = time.monotonic()-start
        step = ledger.steps[-1]['step']
        if step in contract['generation_steps']:
            t = time.monotonic(); generations = []
            for p in prompts:
                generated = distributed_generate(model, fwd, tokenizer.encode(p['text'], add_special_tokens=False).ids,
                            eos_id=eos, context_length=config['seq_len'], max_new_tokens=64, device=device)
                generations.append(dict(id=p['id'], category=p['category'], prompt=p['text'],
                            text=tokenizer.decode(generated['generated_ids'], skip_special_tokens=False), **generated))
            result.update(fixed_generations=generations, generation_seconds=time.monotonic()-t)
            ledger.generations.append(dict(step=step, sha256=identity(generations)))
        return result
    trainer.evaluate_holdout = evaluate
    original_save = trainer.dcp.save
    def save(*args, **kwargs):
        t = time.monotonic(); result = original_save(*args, **kwargs); payload = time.monotonic()-t
        t = time.monotonic(); persist(checkpoint_step=ledger.steps[-1]['step'])
        seconds = torch.tensor([payload, time.monotonic()-t], dtype=torch.float64,
                               device=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        if rank == 0:
            with (root/'checkpoint_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(checkpoint=Path(kwargs['checkpoint_id']).name,
                     payload_seconds_max_rank=float(seconds[0].item()),
                     source_ledger_seconds_max_rank=float(seconds[1].item()),
                     scope='Payload/ledger only; later RNG/meta publication excluded.'))+'\n')
        return result
    trainer.dcp.save = save
    success = False
    try:
        sys.argv = ['train/monitored.py', '--config', str(config_path)]
        trainer.main(); success = True
    finally:
        persist(success)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True); p.add_argument('--config', required=True)
    p.add_argument('--attempt', required=True); a = p.parse_args()
    worker(a.run_dir, a.config, a.attempt)
