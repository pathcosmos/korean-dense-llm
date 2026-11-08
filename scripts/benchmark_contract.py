"""Content-bound certificates for bounded synthetic GPU measurements."""
import hashlib,json,platform,subprocess
from pathlib import Path
import torch
from common.data_contracts import atomic_json,sha256_file,identity

CODE=['scripts/benchmark_gpu.py','scripts/benchmark_contract.py','scripts/check_gpu_numerics.py','model/model.py','train/train.py','train/data.py','train/runtime.py','eval/ppl.py','common/paths.py']
def code_hashes(root):
    return {p:sha256_file(root/p) for p in CODE if (root/p).exists()}
def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip())
def bind(path,contract):
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError('Benchmark contract changed; use a new run directory')
    else:atomic_json(path,contract)
def certify(marker,paths,contract_id):
    atomic_json(marker,dict(contract_id=contract_id,artifacts={str(p.name):sha256_file(p) for p in paths}))
def verify(marker,paths,contract_id):
    d=json.loads(marker.read_text())
    if d!=dict(contract_id=contract_id,artifacts={str(p.name):sha256_file(p) for p in paths}):
        raise ValueError('Benchmark result certificate mismatch')
