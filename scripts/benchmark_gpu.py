"""Bounded 8-GPU synthetic throughput sweep; no production data or checkpoints.

Run from the project root: python scripts/benchmark_gpu.py --run-dir runs/<new-name>
Successful cases have .done markers; reruns reuse them. Interrupted case logs
are archived before retry. Each case has a 15-minute timeout.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
import numpy as np
from scripts.benchmark_contract import code_hashes,environment,bind,certify,verify
from common.data_contracts import atomic_json,sha256_file,identity
from datetime import datetime,timezone

def event(run,status,**kw):
    item=dict(utc=datetime.now(timezone.utc).isoformat(),stage="gpu_benchmark",event=status,run=str(run),**kw)
    atomic_json(run/"state.json",item)
    with (Path(PROJECT_ROOT)/"logs/pipeline_events.jsonl").open("a") as f:f.write(json.dumps(item)+"\n")



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    root = Path(PROJECT_ROOT)
    run = (root / args.run_dir).resolve()
    run.relative_to(root / 'runs')
    run.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runs/.gpu_benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run / 'pid').write_text(str(os.getpid()) + '\n')
    data = run / 'synthetic.bin'
    if not data.exists():
        rng = np.random.default_rng(0)
        base = rng.integers(0, 65000, size=(2000, 16))
        tmp = run / 'synthetic.tmp'
        base[rng.integers(0, 2000, size=400000)].astype(np.uint16).tofile(tmp)
        tmp.replace(data)
    mix = run / 'mix.json'
    mix.write_text(json.dumps({'sources': {'synthetic': {'weight': 1, 'files': [str(data.relative_to(root))]}}}))
    cases = [('tiny', True, 2, 128, 6)]
    for model, batch in [('1b', 2), ('3b', 2), ('3b', 4)]:
        for compile_on in (False, True):
            cases.append((model, compile_on, batch, 4096, 40))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1', TORCHINDUCTOR_COMPILE_THREADS='2')
    contract=dict(version=2,code=code_hashes(root),environment=environment(),data_sha256=sha256_file(data),mix_sha256=sha256_file(mix),cases=[list(x) for x in cases],env_overrides={k:env[k] for k in ['OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS']})
    bind(run/'contract.json',contract)
    contract_id=identity(contract)
    event(run,'started',contract_id=contract_id)
    numerical=run/'numerics.json'
    subprocess.run(['timeout','180s',sys.executable,'scripts/check_gpu_numerics.py','--output',str(numerical)],cwd=root,env=env,check=True)
    summary = []
    for model, compile_on, batch, seq, steps in cases:
        name = f'{model}_compile{int(compile_on)}_b{batch}'
        case = run / name
        case.mkdir(exist_ok=True)
        config = dict(model=model, seq_len=seq, micro_bsz=batch,
                      global_tokens=131072 if seq == 4096 else 2048,
                      max_steps=steps, warmup=10, lr=3e-4, log_every=5 if steps > 6 else 1,
                      eval_every=0, ckpt_every=0, save_final=False,
                      compile=compile_on, holdout=[], mix=str(mix.relative_to(root)),
                      out_dir=str(case.relative_to(root)))
        config_path = case / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError(f'Configuration changed: {case}')
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        result_path = case / 'result.json'
        if (case / '.done').exists():
            verify(case/'.done',[result_path,config_path,case/'log.jsonl'],contract_id)
            summary.append(json.loads(result_path.read_text()))
            continue
        for filename in ['log.jsonl', 'console.log']:
            p = case / filename
            if p.exists():
                p.rename(case / (filename + f'.previous-{time.time_ns()}'))
        event(run,'case_started',case=name)
        print(f'START {name}', flush=True)
        start = time.monotonic()
        with (case / 'console.log').open('w') as log:
            proc = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30s', '900s',
                                     'torchrun', '--standalone', '--nproc_per_node=8',
                                     'train/train.py', '--config', str(config_path)],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            (case / 'pid').write_text(str(proc.pid) + '\n')
            code = proc.wait()
        wall = time.monotonic() - start
        rows = [json.loads(s) for s in (case / 'log.jsonl').read_text().splitlines()] if (case / 'log.jsonl').exists() else []
        rows = [r for r in rows if 'tok_s' in r]
        valid = code == 0 and rows and rows[-1]['step'] == steps and all(
            math.isfinite(r[k]) for r in rows for k in ['loss', 'gnorm', 'tok_s'])
        if not valid:
            (case / 'failure.json').write_text(json.dumps({'returncode': code, 'wall_s': wall}))
            event(run,'error',case=name,returncode=code)
            raise RuntimeError(f'{name} failed; inspect {case / "console.log"}')
        measured = [r for r in rows if r['step'] > (10 if steps > 6 else 2)]
        result = dict(name=name, model=model, compile=compile_on, micro_bsz=batch,
                      seq_len=seq, steps=steps, global_tokens=config['global_tokens'],
                      median_tok_s=statistics.median(r['tok_s'] for r in measured),
                      aggregate_tok_s=len(measured) / sum(1/r['tok_s'] for r in measured),
                      max_allocated_gib=max(r['max_allocated_gib'] for r in rows),
                      max_reserved_gib=max(r['max_reserved_gib'] for r in rows),
                      first_log_tok_s=rows[0]['tok_s'], wall_s=wall,
                      final_loss=rows[-1]['loss'])
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        if code_hashes(root)!=contract['code']:raise RuntimeError('Benchmark code changed during execution')
        certify(case/'.done',[result_path,config_path,case/'log.jsonl'],contract_id)
        event(run,'case_complete',case=name,tok_s=result['aggregate_tok_s'])
        summary.append(result)
        (run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('DONE ' + json.dumps(result), flush=True)
    (run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    certify(run/'.done',[run/'summary.json',run/'contract.json',numerical],contract_id)
    event(run,'complete',cases=len(summary))


if __name__ == '__main__':
    main()
