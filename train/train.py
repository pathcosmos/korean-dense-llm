"""Pretraining loop: FSDP2 + bf16 + WSD schedule + DCP checkpoints with exact resume.
single GPU : torchrun --standalone --nproc_per_node=1 train/train.py --config train/configs/smoke.json
8 GPU      : torchrun --standalone --nproc_per_node=8 train/train.py --config train/configs/3b.json
"""
import os, sys, json, time, math, argparse, shutil
from pathlib import Path
from datetime import timedelta
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, torch.distributed as dist, torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from model.model import Transformer, get_config
from train.data import MixSampler, HoldoutSet, validate_sampler_resume
from train.runtime import (training_contract, validate_resume, complete_checkpoints,
                           require_finite, ensure_finite_gradients, evaluate_holdout)
from common.data_contracts import resolve, atomic_json

DEFAULT = dict(model="tiny", vocab_size=65536, seq_len=4096, micro_bsz=8, global_tokens=2_097_152,
               max_steps=1000, lr=3e-4, min_lr_ratio=0.1, warmup=500, decay_frac=0.15,
               wd=0.1, beta1=0.9, beta2=0.95, clip=1.0, seed=1234, compile=True,
               mix="data/tokens/mix.json", holdout=[], out_dir="runs/smoke",
               ckpt_every=500, eval_every=250, log_every=10, keep_ckpts=2, save_final=True, sampler_version=2, stop_after=None)


def lr_at(s, c):  # WSD: linear warmup -> constant -> linear decay to min_lr
    if s < c["warmup"]: return c["lr"] * (s + 1) / c["warmup"]
    dstart = int(c["max_steps"] * (1 - c["decay_frac"]))
    if s < dstart: return c["lr"]
    p = (s - dstart) / max(1, c["max_steps"] - dstart)
    return c["lr"] * (1 - p * (1 - c["min_lr_ratio"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    c = dict(DEFAULT)
    if a.config: c.update(json.load(open(a.config)))
    for kv in a.set: k, v = kv.split("=", 1); c[k] = json.loads(v)

    if c['stop_after'] is not None and (type(c['stop_after']) is not int or not 0 < c['stop_after'] <= c['max_steps']):
        raise ValueError('stop_after must be within 1..max_steps; max_steps retains the LR schedule')
    if c['keep_ckpts'] < 1 or c['log_every'] < 1:
        raise ValueError('keep_ckpts and log_every must be positive')
    c['out_dir'] = str(resolve(c['out_dir']))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        run(c)
    finally:
        dist.destroy_process_group()


def run(c):
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(c["out_dir"], exist_ok=True)
    log = lambda *x: print(*x, flush=True) if rank == 0 else None

    mix = json.loads(resolve(c['mix']).read_text())
    hold_files = c['holdout'] or mix.get('holdout', [])
    contract_message = [None]
    if rank == 0:
        try:
            contract_message[0] = dict(contract=training_contract(c, world, hold_files))
        except Exception as exc:
            contract_message[0] = dict(error=f'{type(exc).__name__}: {exc}')
    dist.broadcast_object_list(contract_message, src=0, device=dev)
    if 'error' in contract_message[0]:
        raise ValueError(contract_message[0]['error'])
    contract = contract_message[0]['contract']
    ck = complete_checkpoints(c['out_dir'])
    resume_meta = json.loads((ck[-1] / 'meta.json').read_text()) if ck else None
    if resume_meta:
        validate_sampler_resume(resume_meta, c['sampler_version'])
        validate_resume(resume_meta, contract)
    if rank == 0:
        atomic_json(Path(c['out_dir']) / 'training_contract.json', contract)

    cfg = get_config(c["model"], vocab_size=c["vocab_size"], max_seq_len=c["seq_len"])
    with torch.device("meta"): model = Transformer(cfg)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in model.blocks: fully_shard(b, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    model.to_empty(device=dev)
    # Rank-local initialization is deterministic for a fixed world size (part of the contract).
    torch.manual_seed(c["seed"] + rank)
    with torch.no_grad():
        model.init_weights()
    nparams = model.num_params()
    log(f"model={c['model']} params={nparams/1e9:.3f}B world={world}")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["wd"]}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=c["lr"], betas=(c["beta1"], c["beta2"]), fused=True)
    fwd = torch.compile(model) if c["compile"] else model

    # global batch bookkeeping
    tok_per_micro = c["micro_bsz"] * c["seq_len"] * world
    accum = max(1, round(c["global_tokens"] / tok_per_micro))
    tok_per_step = tok_per_micro * accum
    log(f"accum={accum} tokens/step={tok_per_step:,} total_tokens={tok_per_step*c['max_steps']/1e9:.2f}B")

    sampler = MixSampler(c["mix"], c["seq_len"], c["seed"], version=c["sampler_version"])
    hold = HoldoutSet(hold_files, c["seq_len"]) if hold_files else None
    if hold is not None and len(hold) == 0:
        raise ValueError('Configured holdout has no valid windows')

    step = 0
    if resume_meta:
        directory = ck[-1]
        msd, osd = get_state_dict(model, opt)
        state = {"model": msd, "optim": osd}
        dcp.load(state, checkpoint_id=directory)
        set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
        rng = torch.load(directory / f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'], dev)
        step = resume_meta['step']
        log(f'resumed from {directory} @ step {step}')

    def save(s):
        directory = Path(c['out_dir']) / f'ckpt_{s:07d}'
        if rank == 0 and directory.exists():
            if (directory / '.complete').exists():
                raise ValueError('Refusing to overwrite a complete checkpoint')
            directory.rename(directory.with_name(directory.name + f'.incomplete-{time.time_ns()}'))
        dist.barrier()
        msd, osd = get_state_dict(model, opt)
        dcp.save({'model': msd, 'optim': osd}, checkpoint_id=directory)
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(dev)),
                   directory / f'rng_rank{rank}.pt')
        dist.barrier()  # Publish only after every rank's DCP and RNG state are durable to the OS.
        if rank == 0:
            atomic_json(directory / 'meta.json', dict(step=s, config=c, sampler_version=sampler.version,
                        training_contract=contract, tokens_per_step=tok_per_step, world_size=world))
            (directory / '.complete').touch()
            for old in complete_checkpoints(c['out_dir'])[:-c['keep_ckpts']]:
                shutil.rmtree(old)
        dist.barrier()

    fpt = model.flops_per_token(c["seq_len"]); peak = 989e12
    logf = open(f"{c['out_dir']}/log.jsonl", "a") if rank == 0 else None
    model.train(); t0 = time.time(); tl = 0.0
    end_step = c["stop_after"] or c["max_steps"]
    while step < end_step:
        lr = lr_at(step, c)
        for g in opt.param_groups: g["lr"] = lr
        loss_acc = torch.zeros((), device=dev)
        for m in range(accum):
            x, y = sampler.batch(step, rank, m, c["micro_bsz"])
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            model.set_requires_gradient_sync(m == accum - 1); model.set_requires_all_reduce(m == accum - 1)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = fwd(x, y) / accum
            require_finite(loss, 'loss', dev)
            loss.backward(); loss_acc += loss.detach()
        ensure_finite_gradients(model, dev)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"])
        require_finite(gn, 'gradient norm', dev)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % c["log_every"] == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG); torch.cuda.synchronize()
            dt = (time.time() - t0) / c["log_every"]; t0 = time.time()
            tps = tok_per_step / dt; mfu = tps * fpt / (peak * world)
            mem = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                               dtype=torch.int64, device=dev)
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
            rec = dict(step=step, loss=loss_acc.item(), lr=lr, gnorm=gn.item(), tok_s=tps, mfu=mfu,
                       tokens_b=step * tok_per_step / 1e9,
                       max_allocated_gib=mem[0].item() / 2**30,
                       max_reserved_gib=mem[1].item() / 2**30)
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if rank == 0: logf.write(json.dumps(rec) + "\n"); logf.flush()
        if c["eval_every"] > 0 and step % c["eval_every"] == 0:
            e = evaluate_holdout(model, fwd, hold, dev)
            if e is not None:
                log(f"eval step={step} holdout_loss={e['holdout_loss']:.4f} tokens={e['valid_tokens']}")
                if rank == 0:
                    logf.write(json.dumps(dict(step=step, contract_id=contract['id'], **e)) + "\n"); logf.flush()
        if (c["ckpt_every"] > 0 and step % c["ckpt_every"] == 0) or (c["save_final"] and step == end_step): save(step)
    if logf is not None: logf.close()


if __name__ == "__main__":
    main()
