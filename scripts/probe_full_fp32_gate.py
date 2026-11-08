"""Temporary full-model FP32 SwiGLU candidate; production code is unchanged."""
import argparse,sys,fcntl,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file
from scripts.diagnose_compile_precision import worker
from model.model import FeedForward
import torch.nn.functional as F

def fp32_gate(self,x):
    a,b=self.w1(x),self.w3(x)
    return self.w2((F.silu(a.float())*b.float()).to(a.dtype))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(PROJECT_ROOT);out=root/a.output;out.mkdir(parents=True,exist_ok=False)
    with (root/'runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=['scripts/probe_full_fp32_gate.py','scripts/diagnose_compile_precision.py','model/model.py','train/train.py']
        contract=dict(code={p:sha256_file(root/p) for p in files},candidate='Temporary FeedForward.forward: FP32 silu and product, cast once before w2',production_modified=False)
        atomic_json(out/'contract.json',contract);FeedForward.forward=fp32_gate
        worker(argparse.Namespace(model='3b',output=str(out)))
        assert all(sha256_file(root/p)==h for p,h in contract['code'].items())
        atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
