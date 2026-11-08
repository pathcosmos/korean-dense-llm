"""Two independent GPU counterfactuals; production model code is never edited."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json, sha256_file


def main(args):
    root = Path(PROJECT_ROOT)
    out = root / args.output
    if args.worker:
        os.environ['TORCHINDUCTOR_EMULATE_PRECISION_CASTS'] = '1'
        import torch
        import torch._inductor.config as config
        from model.model import RMSNorm, Attention, FeedForward
        from scripts.diagnose_compile_precision import worker
        assert config.emulate_precision_casts
        RMSNorm.forward = torch.compiler.disable(RMSNorm.forward)
        target = Attention if args.worker == 'attention' else FeedForward
        target.forward = torch.compiler.disable(target.forward)
        worker(argparse.Namespace(model='3b', output=str(out)))
        return
    out.mkdir(parents=True, exist_ok=False)
    with (root / 'runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = ['scripts/probe_compile_boundaries.py', 'scripts/diagnose_compile_precision.py',
                 'model/model.py', 'train/train.py']
        contract = dict(code={p: sha256_file(root / p) for p in files},
                        devices=[0, 1], model='3b', sequence=32, timeout_seconds=900,
                        base='emulate_precision_casts + eager RMSNorm',
                        variants=['additional eager attention', 'additional eager MLP'],
                        production_modified=False,
                        scope='Single GPU/seed/step counterfactuals; graph breaks change fusion. Not adoption or FSDP proof.')
        atomic_json(out / 'contract.json', contract)
        jobs = []
        start = time.monotonic()
        try:
            for device, case in enumerate(['attention', 'mlp']):
                target = out / case
                target.mkdir()
                stream = (target / 'console.log').open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device),
                           OMP_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                           TORCHINDUCTOR_COMPILE_THREADS='2')
                proc = subprocess.Popen(['timeout', '--kill-after=30s', '900s', sys.executable,
                                         str(Path(__file__).resolve()), '--worker', case,
                                         '--output', str(target)], env=env, cwd=root,
                                        stdout=stream, stderr=subprocess.STDOUT)
                stream.close()
                jobs.append((case, proc))
            while any(p.poll() is None for _, p in jobs):
                atomic_json(out / 'state.json', dict(phase='running', elapsed_seconds=time.monotonic()-start,
                            cases={c: p.poll() for c, p in jobs}, pid=os.getpid()))
                time.sleep(10)
            if any(p.returncode for _, p in jobs):
                raise RuntimeError('GPU diagnostic failed: ' + str({c: p.returncode for c, p in jobs}))
            assert all(sha256_file(root / p) == h for p, h in contract['code'].items())
            reports = {case: json.loads((out / case / 'report.json').read_text()) for case, _ in jobs}
            report = dict(status='needs_review', cases=reports, scope=contract['scope'],
                          elapsed_seconds=time.monotonic()-start,
                          case_report_sha256={c: sha256_file(out/c/'report.json') for c in reports})
            atomic_json(out / 'report.json', report)
            atomic_json(out / '.done', dict(report_sha256=sha256_file(out/'report.json'),
                                          contract_sha256=sha256_file(out/'contract.json')))
            atomic_json(out / 'state.json', dict(phase='complete', pid=os.getpid()))
        except BaseException as exc:
            for _, proc in jobs:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=35)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            atomic_json(out / 'state.json', dict(phase='error', error=str(exc)))
            raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--worker', choices=['attention', 'mlp'])
    main(ap.parse_args())
