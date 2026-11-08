"""Environment report. Run first on a new machine:  python scripts/check_env.py"""
import os, shutil, subprocess, sys
def cg(p):
    try: return open(p).read().strip()
    except Exception: return "n/a"
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    pr = torch.cuda.get_device_properties(i); print(f"  gpu{i} {pr.name} {pr.total_memory/2**30:.0f} GiB")
print("cpu cores (affinity)", len(os.sched_getaffinity(0)))
print("cgroup memory.max", cg("/sys/fs/cgroup/memory.max"), "(NOTE: bounds total RAM of all worker procs; this caused OOM before)")
from common.paths import PROJECT_ROOT, DATA
print("project root", PROJECT_ROOT)
for d in ["raw", "clean", "tokens"]:
    p = os.path.join(DATA, d)
    if os.path.isdir(p): print(f"  data/{d}: {subprocess.run(['du','-sh',p],capture_output=True,text=True).stdout.split()[0]}")
print("disk free (project):", f"{shutil.disk_usage(PROJECT_ROOT).free/2**30:.0f} GiB")
print("nccl", torch.distributed.is_nccl_available())
