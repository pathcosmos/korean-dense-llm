"""Training identities, synchronized failure checks, and balanced FSDP probes."""
import json
import math
from pathlib import Path
import torch
import torch.distributed as dist
from common.data_contracts import resolve, identity, sha256_file

# Changing these changes the optimization trajectory; operational stop/log paths do not.
RESUME_KEYS = ('model', 'vocab_size', 'seq_len', 'micro_bsz', 'global_tokens',
               'max_steps', 'lr', 'min_lr_ratio', 'warmup', 'decay_frac', 'wd',
               'beta1', 'beta2', 'clip', 'seed', 'compile', 'sampler_version')


def training_contract(config, world, hold_files):
    mix_path = resolve(config['mix'])
    mix = json.loads(mix_path.read_text())
    paths = [p for source in mix['sources'].values() for p in source['files']] + list(hold_files)
    records = {}
    # Only rank 0 calls this; immutable token files are verified once at startup.
    for name in paths:
        path = resolve(name).resolve()
        records[str(path)] = dict(bytes=path.stat().st_size, sha256=sha256_file(path))
    value = dict(version=1, world_size=world,
                 config={k: config[k] for k in RESUME_KEYS}, mix=mix,
                 source_order=list(mix['sources']), holdout=list(hold_files), files=records,
                 code={p: sha256_file(resolve(p)) for p in
                       ('model/model.py', 'train/__init__.py', 'train/train.py', 'train/data.py', 'train/runtime.py')})
    value['id'] = identity(value)
    return value


def validate_resume(metadata, contract):
    previous = metadata.get('training_contract')
    if previous is None:
        raise ValueError('Checkpoint lacks training identity; use a new run directory (sampler version alone is insufficient)')
    if previous != contract:
        raise ValueError('Training identity mismatch: data/tokenizer/mix/config/world/code changed')


def complete_checkpoints(directory):
    return sorted(p for p in Path(directory).glob('ckpt_[0-9]*')
                  if p.is_dir() and (p / '.complete').is_file())


def local_tensor(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def require_finite(value, label, device):
    bad = (~torch.isfinite(local_tensor(value))).any().to(device=device, dtype=torch.int32)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f'non-finite {label}: abort before optimizer update on all ranks')


def ensure_finite_gradients(model, device):
    flags = [torch.isfinite(local_tensor(p.grad)).all() for p in model.parameters() if p.grad is not None]
    valid = torch.stack(flags).all() if flags else torch.ones((), dtype=torch.bool, device=device)
    require_finite(torch.where(valid, 0.0, float('nan')), 'gradient', device)


def evaluation_indices(count, world, rank):
    if count < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError('Evaluation requires targets and a valid rank/world')
    return [(min(i, count - 1), i < count) for i in range(rank, math.ceil(count / world) * world, world)]


@torch.no_grad()
def evaluate_holdout(model, fwd, hold, device):
    if hold is None:
        return None
    rank, world = dist.get_rank(), dist.get_world_size()
    indices = evaluation_indices(len(hold), world, rank)
    previous = model.training
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    try:
        for index, valid in indices:
            w = hold.wins[index:index + 1].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = fwd(w[:, :-1].contiguous(), w[:, 1:].contiguous())
            require_finite(loss, 'evaluation loss', device)
            if valid:
                count = w.shape[1] - 1
                totals[0] += loss.double() * count
                totals[1] += count
        dist.all_reduce(totals)
    finally:
        model.train(previous)
    nll, count = totals.tolist()
    if count == 0:
        raise ValueError('Evaluation has no valid targets')
    return dict(holdout_loss=nll/count, nll_sum=nll, valid_tokens=int(count),
                windows=len(hold), forwards_per_rank=len(indices))
