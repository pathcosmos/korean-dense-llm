"""Take a small resource snapshot for conversation reports; never change jobs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json


def snapshot():
    cg = Path('/sys/fs/cgroup')
    def counters(name):
        return {k: int(v) for k, v in
                (line.split() for line in (cg / name).read_text().splitlines())}
    before = counters('cpu.stat')
    start = time.monotonic()
    time.sleep(5)
    after = counters('cpu.stat')
    elapsed = time.monotonic() - start
    cores = (after['usage_usec'] - before['usage_usec']) / 1e6 / elapsed
    quota, period = (cg / 'cpu.max').read_text().split()
    capacity = len(os.sched_getaffinity(0))
    if quota != 'max':
        capacity = min(capacity, int(quota) / int(period))
    mem = counters('memory.stat')
    limit = (cg / 'memory.max').read_text().strip()
    disks = {}
    for name, path in [('nfs', PROJECT_ROOT), ('local', '/tmp')]:
        usage = shutil.disk_usage(path)
        disks[name] = dict(path=path, device=os.stat(path).st_dev,
                           total_gib=usage.total / 2**30,
                           used_gib=usage.used / 2**30, free_gib=usage.free / 2**30)
    gpu = subprocess.run([
        'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    gpus = []
    if gpu.returncode == 0:
        for line in gpu.stdout.splitlines():
            idx, util, used, total, watts = [v.strip() for v in line.split(',')]
            def number(value):
                try:
                    return float(value)
                except ValueError:
                    return None
            gpus.append(dict(index=int(idx), utilization_percent=number(util),
                             memory_used_mib=number(used), memory_total_mib=number(total),
                             power_watts=number(watts)))
    return dict(utc=datetime.now(timezone.utc).isoformat(), sample_seconds=elapsed,
                cpu=dict(used_cores=cores, capacity_cores=capacity,
                         utilization_percent=100 * cores / capacity,
                         throttled_usec_delta=after.get('throttled_usec', 0) - before.get('throttled_usec', 0)),
                memory=dict(current_gib=int((cg / 'memory.current').read_text()) / 2**30,
                            limit_gib=None if limit == 'max' else int(limit) / 2**30,
                            **{k + '_gib': mem.get(k, 0) / 2**30 for k in
                               ['anon', 'file', 'shmem', 'inactive_file', 'kernel']}),
                gpus=gpus, gpu_error=gpu.stderr.strip() if gpu.returncode else None,
                disks=disks, root_and_tmp_same_filesystem=os.stat('/').st_dev == os.stat('/tmp').st_dev,
                oom=counters('memory.events'),
                scope='CPU: container cgroup five-second average; memory includes file cache; GPU/disk: snapshots. shmem is part of file, not additive. No job or resource settings changed.')


if __name__ == '__main__':
    result = snapshot()
    atomic_json(Path(PROJECT_ROOT) / 'logs/resource_usage_latest.json', result)
    print(json.dumps(result, ensure_ascii=False))
