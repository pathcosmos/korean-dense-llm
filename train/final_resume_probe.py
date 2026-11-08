"""Isolated full-state resume probe; never extends or writes the parent run.

The production entry requires an audited, certified final checkpoint and a
separate diagnostic contract. Continuous two updates are compared with one
update, process exit, load and one update. Parent max_steps remains unchanged.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, atomic_json, identity, sha256_file

MODES = {'continuous': (0, 2), 'first': (0, 1), 'resumed': (1, 2)}
OUTPUTS = {'continuous': 'continuous_final', 'first': 'split_first', 'resumed': 'resumed_final'}


def check_spec(spec):
    plan = spec['plan']
    if spec['plan_id'] != identity(plan) or plan['purpose'] != 'isolated_final_resume_probe':
        raise ValueError('Diagnostic plan identity differs')
    if (plan['parent_step'] != 57000 or plan['world_size'] != 8 or plan['updates'] != 2 or
            plan['lr_policy'] != 'freeze_last_parent_update' or plan['main_target_modified'] is not False):
        raise ValueError('Unreviewed final diagnostic scope')
    main, root = resolve(plan['main_run']), resolve(plan['run_dir'])
    mc = json.loads((main/'contract.json').read_text())
    scratch = Path(mc['tiered_retention']['temp_root'])/root.name
    if (Path(plan['scratch']) != scratch or root.is_relative_to(main) or main.is_relative_to(root) or not root.is_relative_to(resolve('runs')) or
            scratch.is_relative_to(main) or main.is_relative_to(scratch)):
        raise ValueError('Diagnostic path overlaps parent or differs from reserved temporary namespace')
    for path, digest in {**plan['code'], **plan['inputs']}.items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Diagnostic code/input binding differs: '+path)
    if not mc.get('main_training_authorized') or not mc.get('new_checkpoint_pruning_authorized'):
        raise ValueError('Parent authorization missing')
    checkpoint = main/'training/ckpt_0057000'
    if resolve(spec['parent_checkpoint']) != checkpoint:
        raise ValueError('Only exact final NFS parent is permitted')
    for name in ['parent_proof', 'audit']:
        if sha256_file(resolve(spec[name])) != spec[name+'_sha256']:
            raise ValueError('Final proof/audit changed')
    audit = json.loads(resolve(spec['audit']).read_text())
    if (audit['validation_source'] or audit['counts']['committed_tokens'] != 119537664000 or
            audit['counts']['successful_target_ranks'] != list(range(8))):
        raise ValueError('Actual final8rank target audit missing')
    # Full payload SHA is refreshed once by the live controller; workers check
    # its exact stat snapshot as well as the certified inventory before loading.
    from scripts.tiered_checkpoint_retention import inventory
    proof = json.loads(resolve(spec['parent_proof']).read_text())
    inventory(checkpoint, proof)
    if spec['parent_stats'] != file_stats(checkpoint):
        raise ValueError('Parent changed after fresh controller SHA verification')
    meta = json.loads((checkpoint/'meta.json').read_text())
    parent = json.loads(resolve(plan['parent_contract']).read_text())
    config = json.loads((main/'main.json').read_text())
    verify_parent_metadata(meta, config, parent, plan['parent_training_contract_id'])
    if spec['input_stats'] != {p: stat_record(resolve(p)) for p in parent['files']}:
        raise ValueError('Training input changed after controller fresh verification')
    if json.loads(resolve(config['mix']).read_text()) != parent['mix']:
        raise ValueError('Current sampler mix differs')
    for path in [scratch, root]:
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError('Diagnostic output symlink rejected')
    return plan, config, parent


def verify_parent_metadata(meta, config, parent, contract_id):
    # The frozen native CLI resolves only out_dir before saving its full config.
    saved_config = dict(config, out_dir=str(resolve(config['out_dir'])))
    if (meta['step'] != 57000 or meta['world_size'] != 8 or meta['config'] != saved_config or
            meta['training_contract'] != parent or parent['id'] != contract_id):
        raise ValueError('Parent metadata/config/training identity differs')


def stat_record(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected regular immutable file')
    s = path.stat()
    return {k: getattr(s, k) for k in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']}


def file_stats(path):
    rows = {}
    for p in sorted(Path(path).rglob('*')):
        if p.is_symlink():
            raise ValueError('Checkpoint symlink rejected')
        if p.is_file():
            rows[str(p.relative_to(path))] = stat_record(p)
    if not rows:
        raise ValueError('Empty checkpoint')
    return rows


def diagnostic_lr(config):
    from train.train import lr_at
    # Last actual update's LR, not an altered parent WSD horizon.
    return lr_at(config['max_steps']-1, config)


def verify_optimizer_step(optimizer, expected):
    import torch
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if not parameters or len(optimizer.state) != len(parameters):
        raise ValueError('Loaded optimizer state coverage differs')
    for parameter in parameters:
        state = optimizer.state[parameter]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Loaded AdamW fields differ')
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != expected:
            raise ValueError('Loaded optimizer step differs')
        for key in ['exp_avg', 'exp_avg_sq']:
            if state[key].shape != parameter.shape or state[key].dtype != parameter.dtype:
                raise ValueError('Loaded optimizer tensor geometry differs')


def rng_state(device):
    import torch
    result = {'cpu': torch.get_rng_state()}
    if torch.device(device).type == 'cuda':
        result['cuda'] = torch.cuda.get_rng_state(device)
    return result


def restore_state(model, optimizer, checkpoint, rank, device, expected_step):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from scripts.supervise_main_training import validate_optimizer_map
    path = Path(checkpoint)
    if not (path/'.complete').is_file():
        raise ValueError('Incomplete source checkpoint')
    validate_optimizer_map(dcp.FileSystemReader(path).read_metadata())
    msd, osd = get_state_dict(model, optimizer)
    state = {'model': msd, 'optim': osd}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, optimizer, model_state_dict=state['model'], optim_state_dict=state['optim'])
    verify_optimizer_step(optimizer, expected_step)
    rng = torch.load(path/f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
    keys = {'cpu', 'cuda'} if torch.device(device).type == 'cuda' else {'cpu'}
    if set(rng) != keys or any(not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1 for v in rng.values()):
        raise ValueError('Invalid rank RNG state')
    torch.set_rng_state(rng['cpu'])
    if 'cuda' in rng:
        torch.cuda.set_rng_state(rng['cuda'], device)
    if any(not torch.equal(v, rng_state(device)[k]) for k, v in rng.items()):
        raise ValueError('Rank RNG failed exact restore')


def save_state(model, optimizer, path, rank, device, metadata):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    path = Path(path)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)  # No overwrite or implicit cleanup.
    dist.barrier()
    msd, osd = get_state_dict(model, optimizer)
    dcp.save({'model': msd, 'optim': osd}, checkpoint_id=path)
    torch.save(rng_state(device), path/f'rng_rank{rank}.pt')
    dist.barrier()
    if rank == 0:
        atomic_json(path/'meta.json', metadata)
        (path/'.complete').touch()
    dist.barrier()
    # .complete is only a candidate; controller fresh SHA/fsync certification
    # and independent state comparisons are required for final acceptance.


def run(spec, mode):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from model.model import Transformer, get_config
    from train.data import MixSampler
    from train.rehearsal import SourceLedger, observed_sampler
    from train.runtime import require_finite, ensure_finite_gradients

    plan, config, parent = check_spec(spec)
    if plan['framework'] != {'torch': torch.__version__, 'cuda': torch.version.cuda}:
        raise ValueError('Diagnostic framework differs from parent')
    if mode not in MODES or dist.get_world_size() != 8:
        raise ValueError('Probe requires a reviewed mode and actual8GPU ranks')
    rank = dist.get_rank()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.backends.cuda.matmul.allow_tf32 = True
    root, scratch = resolve(plan['run_dir']), Path(plan['scratch'])
    begin, end = MODES[mode]
    source = resolve(spec['parent_checkpoint']) if begin == 0 else scratch/'split_first'
    if begin:
        from scripts.verify_final_resume_probe import verify_inventory
        proof = json.loads((root/'certificates/split_first.json').read_text())
        verify_inventory(source, proof)
        meta = json.loads((source/'meta.json').read_text())
        if meta != metadata_for(spec, 1):
            raise ValueError('Intermediate diagnostic resume identity differs')
    # Every mode has its own process. No incomplete prior attempt is overwritten.
    destinations = [scratch/OUTPUTS[mode]]
    if mode == 'continuous':
        destinations.append(scratch/'restored_parent')
    if any(p.exists() for p in destinations) or (root/'ledgers'/f'{mode}-rank{rank}.json').exists():
        raise ValueError('Existing diagnostic attempt is preserved; use reviewed recovery')
    cfg = get_config(config['model'], vocab_size=config['vocab_size'], max_seq_len=config['seq_len'])
    with torch.device('meta'):
        model = Transformer(cfg)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    model.to_empty(device=device)
    torch.manual_seed(config['seed']+rank)
    with torch.no_grad():
        model.init_weights()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': config['wd']},
                                  {'params': nodecay, 'weight_decay': 0.0}], lr=config['lr'],
                                 betas=(config['beta1'], config['beta2']), fused=True)
    restore_state(model, optimizer, source, rank, device, plan['parent_step']+begin)
    if mode == 'continuous':
        save_state(model, optimizer, scratch/'restored_parent', rank, device, metadata_for(spec, 0))
    ledger = SourceLedger(parent['source_order'])
    sampler = observed_sampler(MixSampler, ledger)(config['mix'], config['seq_len'], config['seed'], version=config['sampler_version'])
    micro_tokens = config['micro_bsz']*config['seq_len']*8
    if config['global_tokens'] % micro_tokens:
        raise ValueError('Nonintegral accumulation')
    accum = config['global_tokens']//micro_tokens
    lr = diagnostic_lr(config)
    records = []
    def persist(success):
        atomic_json(root/'ledgers'/f'{mode}-rank{rank}.json', dict(rank=rank, mode=mode, success=success,
                    plan_id=spec['plan_id'], steps=ledger.steps, metrics=records,
                    parent_step=plan['parent_step'], diagnostic_only=True))
    model.train()
    persist(False)
    for offset in range(begin, end):
        step = plan['parent_step']+offset
        start = time.monotonic()
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss_acc = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = sampler.batch(step, rank, micro, config['micro_bsz'])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.set_requires_gradient_sync(micro == accum-1)
            model.set_requires_all_reduce(micro == accum-1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, y)/accum
            require_finite(loss, 'diagnostic loss', device)
            loss.backward()
            loss_acc += loss.detach()
        ensure_finite_gradients(model, device)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        require_finite(norm, 'diagnostic gradient norm', device)
        optimizer.step()
        ledger.commit()
        optimizer.zero_grad(set_to_none=True)
        verify_optimizer_step(optimizer, step+1)
        dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        records.append(dict(step=step+1, loss=loss_acc.item(), gnorm=norm.item(), lr=lr,
                            elapsed_seconds=time.monotonic()-start))
        persist(False)
    save_state(model, optimizer, scratch/OUTPUTS[mode], rank, device, metadata_for(spec, end))
    persist(True)


def metadata_for(spec, offset):
    p = spec['plan']
    return dict(kind='final_resume_diagnostic', step=p['parent_step']+offset, diagnostic_updates=offset,
                world_size=p['world_size'], diagnostic_contract_id=spec['plan_id'],
                parent_step=p['parent_step'], parent_checkpoint=spec['parent_checkpoint'],
                parent_training_contract_id=p['parent_training_contract_id'],
                parent_config_sha256=p['inputs'][p['parent_config']], main_target_modified=False,
                lr_policy=p['lr_policy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--mode', choices=sorted(MODES), required=True)
    args = parser.parse_args()
    from train.deterministic_train import configure
    configure()
    import torch.distributed as dist
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    try:
        run(json.loads(resolve(args.spec).read_text()), args.mode)
    finally:
        dist.destroy_process_group()
