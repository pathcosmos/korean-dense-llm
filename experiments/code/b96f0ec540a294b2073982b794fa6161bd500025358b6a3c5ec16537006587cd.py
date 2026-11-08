"""Process-tree RSS and proportional resident memory without double-counting.

Shared token memmaps can appear in each rank's RSS. PSS apportions shared pages;
it is a job observation, separate from cgroup cache-aware start admission.
"""
from pathlib import Path


def tree_pids(pid):
    pending=[pid];seen=set()
    while pending:
        current=pending.pop()
        if current in seen:continue
        seen.add(current)
        try:pending.extend(map(int,Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except FileNotFoundError:continue
    return sorted(seen)


def parse_rollup(text):
    fields={}
    for line in text.splitlines():
        if ':' not in line:continue
        key,value=line.split(':',1)
        if key in ('Rss','Pss','Pss_Anon','Pss_File','Pss_Shmem'):
            number=int(value.split()[0])
            if number<0:raise ValueError('Negative process memory counter')
            fields[key]=number*1024
    if not {'Rss','Pss'}<=fields.keys() or fields['Pss']>fields['Rss']:
        raise ValueError('Missing or inconsistent RSS/PSS counter')
    return fields


def proportional_tree(pid):
    rows=[]
    for child in tree_pids(pid):
        try:values=parse_rollup(Path(f'/proc/{child}/smaps_rollup').read_text())
        except FileNotFoundError:continue
        rows.append(dict(pid=child,**values))
    if not rows:raise RuntimeError('No live readable process memory records')
    totals={key:sum(r.get(key,0) for r in rows) for key in ('Rss','Pss','Pss_Anon','Pss_File','Pss_Shmem')}
    return dict(processes=rows,totals_bytes=totals,
                scope='Sampled process tree PSS,not cgroup usage or exclusive ownership of all cached files. No per-process RSS limit imposed.')
