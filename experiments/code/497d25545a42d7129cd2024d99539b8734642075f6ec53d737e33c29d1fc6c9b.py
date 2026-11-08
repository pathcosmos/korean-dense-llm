"""Read container limits and size throughput-oriented stage starting points.
Worker counts are benchmark candidates, not claims of measured optimum.
"""
import argparse,json,math,os,shutil,subprocess,sys,tempfile,hashlib
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json
GIB=2**30

def effective_cpu(affinity,quota):
    fields=quota.split()
    return float(affinity) if not fields or fields[0]=='max' else min(float(affinity),int(fields[0])/int(fields[1]))
def mount_info(path):
    p=Path(path)
    while not p.exists():p=p.parent
    d=shutil.disk_usage(p)
    try:
        fs=json.loads(subprocess.check_output(['findmnt','-J','-T',str(p),'-o','TARGET,FSTYPE'],text=True))['filesystems'][0]
    except (OSError,subprocess.CalledProcessError,ValueError,KeyError):fs=dict(target=str(p),fstype='unknown')
    return dict(path=str(p),fstype=fs['fstype'],target=fs['target'],device=p.stat().st_dev,total_bytes=d.total,free_bytes=d.free,network=fs['fstype'] in ['nfs','nfs4','cifs','ceph','fuse.sshfs'])
def presets(cpu,memory_headroom_gib):
    cores=max(1,math.floor(cpu));reserved=min(8,max(0,cores//8));usable=max(1,cores-reserved)
    # Threads below are enforced by the launcher; worker counts are explicit CLI planning values.
    token_workers=max(1,min(8,usable//8,int(max(1,memory_headroom_gib//16))));rust=max(1,min(8,usable//(token_workers+2)));arrow=max(1,min(4,usable//(token_workers*4)));prep_workers=max(1,min(16,usable//4,int(max(1,memory_headroom_gib//25))))
    return dict(cpu_effective=cpu,reserved_cpu=reserved,launch_permitted=memory_headroom_gib>=4,minimum_memory_headroom_gib=4,stages={
        'cpu-prep':dict(start_workers=prep_workers,max_workers=max(prep_workers,min(32,usable//2,int(max(1,memory_headroom_gib//25)))),arrow_threads=arrow,env=dict(OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2' if cores>=2 else '1',RAYON_NUM_THREADS='1'),job_memory_budget_gib=min(1024,max(0,memory_headroom_gib*.75)),worker_candidates=[4,8,16,32]),
        'tokenization':dict(start_workers=token_workers,max_workers=max(token_workers,min(8,usable//(rust+2))),arrow_threads=arrow,env=dict(OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2' if cores>=2 else '1',RAYON_NUM_THREADS=str(rust),TOKENIZERS_PARALLELISM='true'),job_memory_budget_gib=min(256,max(0,memory_headroom_gib))),
        'tokenizer-train':dict(start_workers=1,max_workers=1,arrow_threads=max(1,min(8,usable)),env=dict(OMP_NUM_THREADS='2' if cores>=2 else '1',NUMEXPR_NUM_THREADS='2' if cores>=2 else '1',RAYON_NUM_THREADS=str(min(64,usable)),TOKENIZERS_PARALLELISM='true'),job_memory_budget_gib=min(1024,max(0,memory_headroom_gib*.75)),rayon_candidates=[8,16,32,64]),
        'sqlite':dict(start_workers=1,max_workers=1,arrow_threads=max(1,min(4,usable)),env=dict(OMP_NUM_THREADS='2' if cores>=2 else '1',NUMEXPR_NUM_THREADS='2' if cores>=2 else '1'),validated_cache_mib=131072,validated_cache_candidates_mib=[512,8192,131072],hash_workers=min(32,usable),candidate_note='3M same-input cache sweep verified logical identity; 128GiB is a capacity ceiling for the estimated100GiB full DB, not proof of full-scale speed or resident128GiB usage. Use fresh RSS headroom and runtime guards.',job_memory_budget_gib=min(384,max(0,memory_headroom_gib*.75))),
        'gpu':dict(start_workers=1,max_workers=1,arrow_threads=max(1,min(4,usable)),env=dict(OMP_NUM_THREADS='2' if cores>=2 else '1',NUMEXPR_NUM_THREADS='2' if cores>=2 else '1',TORCHINDUCTOR_COMPILE_THREADS=str(max(1,min(4,usable//8)))),job_memory_budget_gib=min(256,max(0,memory_headroom_gib)),gpu_policy='Use available required GPUs for compatible kernels/training. Eight-GPU main training still requires user confirmation and numerical gates.')})
def inspect():
    cg=Path('/sys/fs/cgroup');affinity=sorted(os.sched_getaffinity(0));quota=(cg/'cpu.max').read_text() if (cg/'cpu.max').exists() else 'max 100000';cpu=effective_cpu(len(affinity),quota)
    lim=(cg/'memory.max').read_text().strip() if (cg/'memory.max').exists() else 'max';meminfo={k:int(v.strip().split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if v.strip().split() and v.strip().split()[0].isdigit()}
    limit=int(lim) if lim!='max' else meminfo['MemTotal'];current=int((cg/'memory.current').read_text()) if (cg/'memory.current').exists() else 0;reserve=min(128*GIB,limit//8);headroom=max(0,min(limit-current,meminfo['MemAvailable'])-reserve)/GIB
    stat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines()) if (cg/'memory.stat').exists() else {}
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    except (OSError,subprocess.CalledProcessError):gpu=[]
    scratch=Path(tempfile.gettempdir())/('llm-dense-'+hashlib.sha256(str(PROJECT_ROOT).encode()).hexdigest()[:10]);project=mount_info(PROJECT_ROOT);local=mount_info(scratch)
    return dict(utc=datetime.now(timezone.utc).isoformat(),cpu=dict(affinity=affinity,effective=cpu,quota=quota.strip()),memory=dict(limit_bytes=limit,current_bytes=current,file_cache_bytes=int(stat.get('file',0)),reserve_bytes=reserve,conservative_headroom_gib=headroom),gpus=gpu,storage=dict(project=project,scratch_parent=local,scratch_dir=str(scratch)),presets=presets(cpu,headroom),policy='Measured-container limits, not host core counts. Current memory includes reclaimable file cache. No global kernel/mount/GPU clock changes. Worker and memory budgets are planning values, not automatically enforced by arbitrary child commands.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();r=inspect();atomic_json(Path(a.output),r);print(json.dumps(dict(cpu=r['cpu']['effective'],memory_gib=round(r['memory']['limit_bytes']/GIB),project_fs=r['storage']['project']['fstype'],local_free_gib=round(r['storage']['scratch_parent']['free_bytes']/GIB),gpu_count=len(r['gpus']))))
