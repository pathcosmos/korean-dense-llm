"""One indexed traversal per nonoverlapping group range, after fresh parent validation.
No extra corpus-wide scan for each source and no full-sort materialization.
"""
from concurrent.futures import ProcessPoolExecutor
import heapq
import multiprocessing
from pathlib import Path
import sqlite3
import time
from common.data_contracts import resolve

_DB=None


def _initialize(database,cache_mib):
    global _DB
    _DB=sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)
    _DB.execute(f'PRAGMA cache_size={-cache_mib*1024}')


def _keep(heap,item,size):
    if len(heap)<size:heapq.heappush(heap,item)
    elif item>heap[0]:heapq.heapreplace(heap,item)


def _scan(bounds):
    low,high=bounds;clauses=[];args=[]
    if low is not None:clauses.append('group_id>=?');args.append(low)
    if high is not None:clauses.append('group_id<?');args.append(high)
    where=(' WHERE '+' AND '.join(clauses)) if clauses else ''
    query='SELECT group_id,source,split FROM docs INDEXED BY doc_group'+where+' ORDER BY group_id'
    sources={};global_counts={};largest=[];groups=0;last=None;members={};rows=0
    def finish(group,members):
        nonlocal groups
        if group is None:return
        labels={label for source,label in members}
        if len(labels)!=1:raise ValueError('Group crosses train/eval')
        total=sum(members.values());groups+=1;label=next(iter(labels));global_counts[label]=global_counts.get(label,0)+total
        _keep(largest,(total,group,len({s for s,_ in members})),10)
        for (source,label),n in members.items():
            item=sources.setdefault(source,dict(splits={},largest=[]));v=item['splits'].setdefault(label,dict(documents=0,groups=0))
            v['documents']+=n;v['groups']+=1;_keep(item['largest'],(n,group),5)
    for group,source,label in _DB.execute(query,args):
        if not isinstance(group,str) or not group or label not in ('train','eval'):raise ValueError('Invalid group/label')
        if group!=last:finish(last,members);last=group;members={}
        key=(source,label);members[key]=members.get(key,0)+1;rows+=1
    finish(last,members)
    return dict(bounds=bounds,rows=rows,groups=groups,counts=global_counts,sources=sources,largest=largest)


def distribution(context,*,workers=8,worker_cache_mib=16384):
    if type(workers) is not int or not 1<=workers<=16 or type(worker_cache_mib) is not int or not 1<=worker_cache_mib<=262144:
        raise ValueError('Invalid distribution resource settings')
    context.assert_control_files();context.selection.db.execute('SELECT 1');start=time.perf_counter()
    database=resolve(context.selection.report['database']['path'])
    # Contiguous ranges cover every BINARY-collated string, including repo roots.
    cuts=[None]+['text:'+c for c in '02468ace']+[None];bounds=list(zip(cuts,cuts[1:]))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,
                             initargs=(str(database),worker_cache_mib)) as pool:parts=list(pool.map(_scan,bounds))
    expected={};sources={name:dict(splits={},largest=[]) for name in context.records};counts={};largest=[]
    for f in context.selection.report['contract']['files']:expected[f['source']]=expected.get(f['source'],0)+f['rows']
    for part in parts:
        for label,n in part['counts'].items():counts[label]=counts.get(label,0)+n
        for name,item in part['sources'].items():
            if name not in sources:raise ValueError('Unknown source in split DB')
            for label,v in item['splits'].items():
                dest=sources[name]['splits'].setdefault(label,dict(documents=0,groups=0))
                for key in ['documents','groups']:dest[key]+=v[key]
            for pair in item['largest']:_keep(sources[name]['largest'],tuple(pair),5)
        for item in part['largest']:_keep(largest,tuple(item),10)
    if counts!=context.selection.report['counts']:raise ValueError('Distribution counts differ from validated split')
    for name,item in sources.items():
        if sum(v['documents'] for v in item['splits'].values())!=expected[name]:raise ValueError('Distribution source coverage differs')
        item['largest_components']=[dict(group_id=g,documents=n) for n,g in sorted(item.pop('largest'),reverse=True)]
        total=sum(v['documents'] for v in item['splits'].values());item['eval_document_fraction']=item['splits'].get('eval',{}).get('documents',0)/max(total,1)
    context.assert_control_files()
    return dict(split_contract_id=context.selection.report['contract_id'],source_sha256=context.source_sha,split_manifest_sha256=context.split_sha,
                database=context.selection.report['database'],counts=counts,sources=sources,global_groups=sum(p['groups'] for p in parts),
                largest_components=[dict(group_id=g,documents=n,sources=c) for n,g,c in sorted(largest,reverse=True)],
                elapsed_seconds=time.perf_counter()-start,workers=workers,worker_cache_mib=worker_cache_mib,range_coverage=[p['bounds'] for p in parts],
                rows_scanned=sum(p['rows'] for p in parts),validation='Fresh verified parent context; disjoint group ranges; global and per-source row totals rechecked.',
                tie_order='documents descending, then group ID descending',limitation='Document/group distribution only; byte/token distribution requires source/token pilot.')
