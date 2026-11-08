"""Temporary eager RMSNorm + cast preservation diagnostic, no production edits."""
import argparse,sys,fcntl,os
from pathlib import Path
os.environ['TORCHINDUCTOR_EMULATE_PRECISION_CASTS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch._inductor.config as inductor_config
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file
from scripts.diagnose_compile_precision import worker
from model.model import RMSNorm
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(PROJECT_ROOT);out=root/a.output;out.mkdir(parents=True,exist_ok=False)
    with (root/'runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert inductor_config.emulate_precision_casts
        files=['scripts/probe_eager_norm_casts.py','scripts/diagnose_compile_precision.py','model/model.py','train/train.py']
        contract=dict(code={p:sha256_file(root/p) for p in files},torch=torch.__version__,inductor_config_sha256=sha256_file(Path(torch.__file__).parent/'_inductor/config.py'),emulate_precision_casts=True,eager_rmsnorm=True,production_modified=False)
        atomic_json(out/'contract.json',contract);RMSNorm.forward=torch.compiler.disable(RMSNorm.forward);worker(argparse.Namespace(model='3b',output=str(out)))
        assert all(sha256_file(root/p)==h for p,h in contract['code'].items())
        atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
