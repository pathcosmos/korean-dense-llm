"""Resumable read-only all-column audit with certified per-file fingerprints.
Receipts commit after fingerprint rename. Resume hashes completed source/artifacts;
file size/mtime alone never permit cache reuse. No data promotion or source edits.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import resolve,artifact_path,identity,sha256_file,atomic_json
from common.source_inventory import definition,content_inventory,load_audit
from data.probe_code_math_overlap import fenced_blocks,block_key

FINGERPRINT_SCHEMA=pa.schema([('row',pa.int64()),('raw_sha256',pa.binary(32)),
    ('code_file_block_sha256',pa.binary(32)),('web_block_sha256',pa.list_(pa.binary(32)))])
CODE_COLUMNS=['text','h','repo_name','path','license','language','source_row','content_sha256']


class AuditBudgetReached(RuntimeError):pass


def run(manifest,output_dir,*,pilot=False,batch_size=1024,max_seconds=7200,max_rss_gib=16,min_free_gib=200,
        event_log=None,_checkpoint=None):
    if type(batch_size) is not int or batch_size<1 or max_seconds<=0 or max_rss_gib<=0 or min_free_gib<0:
        raise ValueError('Invalid audit resource limits')
    manifest=resolve(manifest);inventory=json.loads(manifest.read_text())
    if inventory.get('status') not in ('draft','ready'):raise ValueError('Invalid candidate status')
    spec=definition(inventory);descriptors=[]
    for source,e in spec['sources'].items():
        if not resolve(e['completion_marker']).is_file():raise ValueError('Source completion evidence missing')
        for file in e['files']:
            path=resolve(file);marker=Path(str(path)+'.done')
            if not path.is_file() or not marker.is_file():raise ValueError('Input incomplete: '+file)
            p=pq.ParquetFile(path);stat=path.stat();code=e['text_policy']=='raw-utf8-code-v1'
            columns=CODE_COLUMNS if code else ['text','h']
            if any(c not in p.schema_arrow.names for c in columns):raise ValueError('Audit required column missing: '+file)
            raw_rows=None;stats_sha=None
            if code:
                stats_path=Path(str(path)+'.stats.json')
                if not stats_path.is_file():raise ValueError('Code cleaning receipt missing')
                counts=json.loads(stats_path.read_text())['counts']
                if any(type(v) is not int or v<0 for v in counts.values()):raise ValueError('Invalid code cleaning counts')
                raw_rows=sum(counts.values());stats_sha=sha256_file(stats_path)
            marker_value=json.loads(marker.read_text())
            expected_rows=marker_value.get('kept') if not code else counts.get('kept')
            if expected_rows!=p.metadata.num_rows:raise ValueError('Audit footer/completion receipt mismatch')
            descriptors.append(dict(source=source,path=file,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,rows=p.metadata.num_rows,
                marker_sha256=sha256_file(marker),code=code,columns=columns,raw_rows=raw_rows,stats_sha256=stats_sha))
    selected=descriptors
    if pilot:
        selected=[]
        for source in spec['sources']:
            files=sorted([f for f in descriptors if f['source']==source],key=lambda f:(f['bytes'],f['path']))
            selected.append(files[len(files)//2])
    code_files=['data/source_audit.py','common/source_inventory.py','common/data_contracts.py','data/probe_code_math_overlap.py']
    code_hashes={p:sha256_file(resolve(p)) for p in code_files}
    contract=dict(version=1,manifest_sha256=sha256_file(manifest),definition_id=identity(spec),files=selected,
        scope='pilot' if pilot else 'full',batch_size=batch_size,code=code_hashes,
        fingerprint_policy='raw UTF8 SHA256; code block key strips outer CR/LF only; web closed fences min80 chars',
        pilot_selection='Median compressed-size file/source' if pilot else None)
    cid=identity(contract);out=resolve(output_dir);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_path=out/'contract.json';state_path=out/'state.json';report_path=out/'report.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=contract:raise ValueError('Audit input/policy/code changed; use a new output directory')
    else:
        if any(p.name!='.lock' for p in out.iterdir()):raise ValueError('Unrecognized audit output directory')
        atomic_json(config_path,contract)
    (out/'files').mkdir(exist_ok=True);(out/'fingerprints').mkdir(exist_ok=True)
    started=time.monotonic();completed=[];last_heartbeat=0;state={}
    def notify(event,**fields):
        if event_log is not None:
            p=resolve(event_log);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('a') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage='source_audit',event=event,
                run=artifact_path(out),scope=contract['scope'],**fields),ensure_ascii=False)+'\n')
    def status(phase,**fields):
        nonlocal last_heartbeat
        state.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            contract_id=cid,scope=contract['scope'],done=len(completed),total=len(selected),**fields)
        atomic_json(state_path,state);last_heartbeat=time.monotonic()
    def limits():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>max_rss_gib*1024**2:raise MemoryError('Audit RSS budget exceeded')
        if time.monotonic()-started>=max_seconds:raise AuditBudgetReached('Audit time budget reached; committed files can resume')
        if shutil.disk_usage(out).free<min_free_gib*1024**3:raise AuditBudgetReached('Audit free-space reserve reached')
    def checkpoint(event):
        if _checkpoint is not None:_checkpoint(event)
    def stable(desc):
        p=resolve(desc['path']);s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(desc['bytes'],desc['mtime_ns']) or sha256_file(Path(str(p)+'.done'))!=desc['marker_sha256']:
            raise ValueError('Audited input changed: '+desc['path'])
    status('running',active_file=None,processed_rows=0);notify('started',total=len(selected))
    try:
        for index,desc in enumerate(selected):
            limits();stable(desc);path=resolve(desc['path']);receipt=out/'files'/f'{index:06}.json';fp=out/'fingerprints'/f'{index:06}.parquet'
            if receipt.exists():
                old=json.loads(receipt.read_text())
                if old['descriptor']!=desc or old['contract_id']!=cid or sha256_file(path)!=old['sha256'] or sha256_file(fp)!=old['fingerprints']['sha256']:
                    raise ValueError('Committed audit input/artifact content changed')
                if pq.ParquetFile(fp).metadata.num_rows!=old['rows']:raise ValueError('Fingerprint row coverage changed')
                stable(desc);completed.append(old);status('running',active_file=desc['path'],processed_rows=old['rows']);continue
            status('running',active_file=desc['path'],processed_rows=0);notify('file_started',file=desc['path'],index=index+1)
            file_start=time.monotonic();tmp=Path(str(fp)+'.tmp');rows=chars=utf8=blocks=0;last_source_row=-1
            if fp.exists():
                orphan=out/'orphans';orphan.mkdir(exist_ok=True);fp.replace(orphan/f'{index:06}-{time.time_ns()}.parquet')
            with pq.ParquetWriter(tmp,FINGERPRINT_SCHEMA,compression='zstd') as writer:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,columns=desc['columns'],use_threads=False):
                    limits();raws=[];code_keys=[];web_keys=[];positions=[]
                    for item in batch.to_pylist():
                        text=item['text']
                        if not isinstance(text,str) or not text:raise ValueError(f'Invalid text at {desc["path"]}:{rows}')
                        if type(item['h']) is not int:raise ValueError('Invalid h field')
                        encoded=text.encode();digest=hashlib.sha256(encoded).digest();c_key=None;w_keys=[]
                        if desc['code']:
                            if any(not isinstance(item[k],str) or not item[k].strip() for k in ['repo_name','path','license','language']):raise ValueError('Missing code provenance')
                            source_row=item['source_row']
                            if type(source_row) is not int or not last_source_row<source_row<desc['raw_rows']:raise ValueError('Code source row out of range/order')
                            last_source_row=source_row
                            if item['content_sha256']!=digest.hex():raise ValueError('Code raw body SHA256 mismatch')
                            if len(text.strip('\r\n'))>=80:c_key=block_key(text)
                        elif '```' in text or '~~~' in text:
                            w_keys=[block_key(body) for body in fenced_blocks(text) if len(body.strip('\r\n'))>=80]
                        positions.append(rows);raws.append(digest);code_keys.append(c_key);web_keys.append(w_keys)
                        rows+=1;chars+=len(text);utf8+=len(encoded);blocks+=len(w_keys)
                    table=pa.Table.from_arrays([pa.array(positions,pa.int64()),pa.array(raws,pa.binary(32)),
                        pa.array(code_keys,pa.binary(32)),pa.array(web_keys,pa.list_(pa.binary(32)))],schema=FINGERPRINT_SCHEMA)
                    writer.write_table(table)
                    if time.monotonic()-last_heartbeat>=5:status('running',active_file=desc['path'],processed_rows=rows)
            if rows!=desc['rows']:raise ValueError('Audit row count mismatch')
            stable(desc);status('running',active_file=desc['path'],processed_rows=rows,subphase='source_sha256')
            digest=sha256_file(path);stable(desc);tmp.replace(fp);checkpoint('fingerprint_published')
            record=dict(status='passed',contract_id=cid,descriptor=desc,path=desc['path'],source=desc['source'],rows=rows,
                bytes=desc['bytes'],sha256=digest,characters=chars,utf8_bytes=utf8,eligible_web_blocks=blocks,
                fingerprints=dict(path=artifact_path(fp),sha256=sha256_file(fp),rows=rows),seconds=time.monotonic()-file_start,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            atomic_json(receipt,record);checkpoint('receipt_committed');completed.append(record)
            status('running',active_file=None,processed_rows=0,subphase=None);notify('file_complete',file=desc['path'],done=len(completed),total=len(selected))
        if any(sha256_file(resolve(p))!=s for p,s in code_hashes.items()):raise ValueError('Audit code changed during execution')
        if sha256_file(manifest)!=contract['manifest_sha256']:raise ValueError('Candidate manifest changed during audit')
        sources={s:dict(rows=0,files=[]) for s in spec['sources']}
        for r in completed:
            sources[r['source']]['rows']+=r['rows'];sources[r['source']]['files'].append({k:r[k] for k in ['path','rows','bytes','sha256','fingerprints']})
        # Restore original source-file order, independent of pilot selection order.
        for name,e in sources.items():e['files'].sort(key=lambda f:spec['sources'][name]['files'].index(f['path']))
        report=dict(format='source-audit-v1',run_status='completed',scope=contract['scope'],structural_status='passed_selected_files' if pilot else 'passed',
            quality_status='not_certified_by_this_tool',contract=contract,contract_id=cid,definition=spec,
            definition_id=identity(spec),manifest_sha256=sha256_file(manifest),sources=sources,
            files_verified=len(completed),rows_verified=sum(x['rows'] for x in completed),input_bytes=sum(x['bytes'] for x in completed),
            cumulative_file_seconds=sum(x['seconds'] for x in completed),elapsed_this_run_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if not pilot:report['content_inventory']=content_inventory(spec,sources);report['content_inventory_id']=identity(report['content_inventory'])
        if report_path.exists():
            old=json.loads(report_path.read_text())
            # Timing is execution evidence from the original completion; all data fields must agree.
            for k in ['elapsed_this_run_seconds','max_rss_kib']:report[k]=old[k]
            if old!=report:raise ValueError('Completed audit report differs')
        else:atomic_json(report_path,report)
        checkpoint('report_published');atomic_json(out/'.done',dict(report_sha256=sha256_file(report_path),contract_id=cid))
        status('complete',active_file=None,processed_rows=0);notify('completed',done=len(completed),total=len(selected));return report
    except BaseException as exc:
        phase='paused' if isinstance(exc,AuditBudgetReached) else 'error'
        status(phase,error_type=type(exc).__name__,error=str(exc));notify(phase,error_type=type(exc).__name__,done=len(completed),total=len(selected));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
