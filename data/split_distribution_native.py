"""Materialize source/group sizes once in SQLite TEMP, then reuse for reports.
The original published database is read only. Explicit memory/cache budgets belong
to the caller; this uses RAM temp tables to avoid repeated original-table scans.
"""
import time
import uuid
from common.data_contracts import resolve


def distribution(context,*,sort_threads=8):
    if type(sort_threads) is not int or not 0<=sort_threads<=8:raise ValueError('Sort threads must be0..8')
    context.assert_control_files();db=context.selection.db;start=time.perf_counter()
    if db.in_transaction and db.execute('PRAGMA temp_store').fetchone()[0]!=2:raise ValueError('Set RAM TEMP before opening the shared read snapshot')
    db.execute('PRAGMA temp_store=MEMORY');db.execute(f'PRAGMA threads={sort_threads}')
    table='sizes_'+uuid.uuid4().hex
    try:
        db.execute(f'CREATE TEMP TABLE {table} AS SELECT source,group_id,MIN(split) AS split,COUNT(*) AS documents FROM docs GROUP BY source,group_id')
        grouped=time.perf_counter();counts={};sources={name:dict(splits={}) for name in context.records}
        for name,label,n,g in db.execute(f'SELECT source,split,SUM(documents),COUNT(*) FROM {table} GROUP BY source,split'):
            if name not in sources or label not in ('train','eval'):raise ValueError('Unknown source/label')
            sources[name]['splits'][label]=dict(documents=n,groups=g);counts[label]=counts.get(label,0)+n
        if counts!=context.selection.report['counts']:raise ValueError('Distribution total differs')
        expected={}
        for f in context.selection.report['contract']['files']:expected[f['source']]=expected.get(f['source'],0)+f['rows']
        for name,item in sources.items():
            total=sum(v['documents'] for v in item['splits'].values())
            if total!=expected[name]:raise ValueError('Source row coverage differs')
            largest=db.execute(f'SELECT group_id,documents FROM {table} WHERE source=? ORDER BY documents DESC,group_id DESC LIMIT 5',(name,)).fetchall()
            item['largest_components']=[dict(group_id=g,documents=n) for g,n in largest]
            item['eval_document_fraction']=item['splits'].get('eval',{}).get('documents',0)/max(total,1)
        largest=db.execute(f'SELECT group_id,SUM(documents) AS n,COUNT(DISTINCT source) FROM {table} GROUP BY group_id ORDER BY n DESC,group_id DESC LIMIT 10').fetchall()
        context.assert_control_files()
        return dict(split_contract_id=context.selection.report['contract_id'],source_sha256=context.source_sha,split_manifest_sha256=context.split_sha,
                    database=context.selection.report['database'],counts=counts,sources=sources,
                    largest_components=[dict(group_id=g,documents=n,sources=c) for g,n,c in largest],
                    original_rows_materialized=sum(counts.values()),materialization_seconds=grouped-start,elapsed_seconds=time.perf_counter()-start,
                    sort_threads=db.execute('PRAGMA threads').fetchone()[0],temp_store='MEMORY',
                    validation='Fresh verified parent context; native source/group TEMP aggregation, source/global totals checked.',
                    tie_order='documents descending, group ID descending',limitation='Document/group distribution only; bytes/tokens require later pilot. Fresh full consumer validation is separate.')
    finally:db.execute(f'DROP TABLE IF EXISTS temp.{table}')
