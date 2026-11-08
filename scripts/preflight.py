"""Preflight: checks every prerequisite for continuing, prints OK/MISSING + the exact fix command. Run first on a fresh machine.
   python scripts/preflight.py          (exit code 1 if anything REQUIRED is missing)"""
import os, sys, glob, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import PROJECT_ROOT as R, DATA
miss = []
def item(ok, name, fix="", required=True):
    print(f"[{'OK' if ok else ('MISSING' if required else 'todo')}] {name}" + ("" if ok or not fix else f"\n        fix: {fix}"))
    if not ok and required: miss.append(name)
def cnt(pat): return len(glob.glob(os.path.join(R, pat), recursive=True))

print("== machine ==")
try:
    import torch
    item(True, f"torch {torch.__version__} cuda {torch.version.cuda}")
    n = torch.cuda.device_count(); item(n == 8, f"GPUs visible = {n} (expected 8)", "check CUDA_VISIBLE_DEVICES / container GPU allocation; STOP and ask the user if <8")
    item(torch.distributed.is_nccl_available(), "NCCL available")
except ImportError:
    item(False, "torch installed", "bash scripts/setup_env.sh")
for mod in ["tokenizers", "pyarrow", "xxhash", "numpy"]:
    try: __import__(mod); item(True, f"import {mod}")
    except ImportError: item(False, f"import {mod}", "bash scripts/setup_env.sh")
cores = len(os.sched_getaffinity(0)); print(f"      cpu cores = {cores}  (set clean.py workers <= min(cores-2, cgroup_mem/25GB))")
try: print("      cgroup memory.max =", open("/sys/fs/cgroup/memory.max").read().strip())
except Exception: pass
free = shutil.disk_usage(R).free / 2**30
item(free > 200, f"disk free {free:.0f} GiB (200 GiB reserve; future allocation budget still required)", "calculate new token/split/checkpoint demand with scripts/storage_budget.py; see docs/RUNBOOK.md", required=False)

print("== data pipeline (expected counts) ==")
E = {"fineweb2_ko": 25, "wiki_ko": 3, "hplt_ko": 464, "fineweb_edu": 56, "owm": 114}
for s, n in E.items():
    raw = cnt(f"data/raw/{s}/**/*.parquet"); clean = cnt(f"data/clean/{s}/*.parquet.done"); tok = cnt(f"data/tokens/{s}/*.bin.done")
    item(raw >= n, f"raw/{s}: {raw}/{n}", f"python data/download.py {s}" + (" 56" if s == "fineweb_edu" else ""), required=False)
    item(clean >= n, f"clean/{s}: {clean}/{n}", f"python data/clean.py {s} {'en' if s in ('fineweb_edu','owm') else 'ko'} <workers>", required=False)
    item(tok >= n, f"tokens/{s}: {tok}/{n}", f"python data/tokenize_shards.py {s} [holdout_docs]", required=False)
item(os.path.exists(f"{R}/tokenizer/tokenizer.json"), "tokenizer/tokenizer.json (FINAL, not the *_test200mb one)", "python tokenizer/train_tokenizer.py --mb 3000 && python tokenizer/check_tokenizer.py", required=False)
item(os.path.exists(f"{DATA}/tokens/mix.json"), "data/tokens/mix.json", "python data/build_mix.py <src:weight> ...", required=False)
item(os.path.exists(f"{DATA}/MANIFEST.json"), "data/MANIFEST.json (integrity manifest of moved files)", "git pull / copy from old machine", required=False)
print("\nREQUIRED missing:", miss or "none"); print("Next: read docs/STATUS.md and docs/SETUP_CHECKLIST.md; the 'todo' items are the work to do.")
sys.exit(1 if miss else 0)
