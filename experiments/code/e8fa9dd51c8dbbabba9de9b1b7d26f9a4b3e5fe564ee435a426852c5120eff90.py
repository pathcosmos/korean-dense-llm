"""Cache-aware container admission; no cache eviction or kernel-setting changes.

Preserve hardware_profile's conservative counter. Separately credit only clean
inactive file cache, exclude shmem/dirty/writeback, clamp to host MemAvailable,
then subtract the existing reserve. Actual job RSS caps remain independent.
"""
from pathlib import Path
from scripts.hardware_profile import inspect as hardware_inspect

GIB=2**30


def memory_admission(*,limit,current,file_cache,inactive_file,shmem,dirty,writeback,host_available,reserve):
    values=[limit,current,file_cache,inactive_file,shmem,dirty,writeback,host_available,reserve]
    if any(type(v) is not int or v<0 for v in values) or limit<1:raise ValueError('Invalid measured memory inputs')
    clean=max(0,min(current,inactive_file,max(0,file_cache-shmem))-dirty-writeback)
    cgroup_available=max(0,limit-current+clean)
    headroom=max(0,min(cgroup_available,host_available)-reserve)
    return dict(clean_inactive_file_credit_bytes=clean,cgroup_available_after_credit_bytes=cgroup_available,
                host_available_bytes=host_available,reserve_bytes=reserve,headroom_gib=headroom/GIB,
                scope='Measured clean inactive file cache can be reclaimed by the kernel. Shmem/dirty/writeback receive no credit; no global cache flush. Job RSS caps still apply.')


def inspect():
    report=hardware_inspect();cg=Path('/sys/fs/cgroup');stat={k:int(v) for k,v in (row.split() for row in (cg/'memory.stat').read_text().splitlines())}
    memory=report['memory'];current=int((cg/'memory.current').read_text())
    host={key:int(value.split()[0])*1024 for key,value in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if value.strip() and value.split()[0].isdigit()}
    admission=memory_admission(limit=memory['limit_bytes'],current=current,file_cache=stat.get('file',0),inactive_file=stat.get('inactive_file',0),
                               shmem=stat.get('shmem',0),dirty=stat.get('file_dirty',0),writeback=stat.get('file_writeback',0),host_available=host['MemAvailable'],reserve=memory['reserve_bytes'])
    memory['cache_aware_admission']=admission
    return report
