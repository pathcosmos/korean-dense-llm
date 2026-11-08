"""Bounded synthetic split scalability preparation, max 3M rows/case.
Parent enforces time/RSS/disk limits, preserves partial DBs and allows same-contract
resume. Big cases are explicit; do not run them alongside a production I/O audit.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import resolve,atomic_json,sha256_file,identity
from data.split_manifest import build,SplitSelection


def worker(case,size,cache):
    case=Path(case);sources={};last=0
    def progress(phase):
        nonlocal last
        if time.monotonic()-last>=1 or phase in ['assign','finalize','publish_db','publish_report']:
            atomic_json(case/'worker_state.json',dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat()))
            last=time.monotonic()
    progress('generate')
    for name,n in [('web',size//2),('code',size-size//2)]:
        path=case/(name+'.parquet');marker=Path(str(path)+'.done')
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved.get('sha256')!=sha256_file(path) or saved.get('rows')!=n:raise ValueError('Synthetic input changed')
        else:
            if path.exists():raise ValueError('Uncertified synthetic input; preserve output')
            fields=[('text',pa.string())]+([('repo_name',pa.string())] if name=='code' else [])
            tmp=Path(str(path)+'.tmp')
            with pq.ParquetWriter(tmp,pa.schema(fields),compression='zstd') as writer:
                for start in range(0,n,4096):
                    ids=range(start,min(start+4096,n));texts=[f'document {name} {i}' if i%101 else f'duplicate {i//101}' for i in ids]
                    cols=dict(text=texts)
                    if name=='code':cols['repo_name']=[f'org/repo{i//5}' for i in ids]
                    writer.write_table(pa.table(cols));progress('generate')
            tmp.replace(path);atomic_json(marker,dict(rows=n,sha256=sha256_file(path)))
        sources[name]=dict(files=[str(path)],completion_marker=str(marker),text_policy='raw-utf8-code-v1' if name=='code' else 'nfc-web-exact-v1')
    manifest=case/'sources.json';inventory=dict(schema_version=1,status='ready',scope='synthetic-scale-fixture-only',sources=sources)
    if manifest.exists() and json.loads(manifest.read_text())!=inventory:raise ValueError('Synthetic manifest changed')
    if not manifest.exists():atomic_json(manifest,inventory)
    resumed=(case/'split').exists();progress('build');started=time.perf_counter()
    report=build(manifest,case/'split',.1,7,_checkpoint=progress,sqlite_cache_mib=cache)
    build_seconds=time.perf_counter()-started;progress('verify');started=time.perf_counter()
    select=SplitSelection(case/'split/manifest.json',sqlite_cache_mib=cache);select.close();verify_seconds=time.perf_counter()-started
    result=dict(documents=size,build_seconds_current_attempt=build_seconds,verification_seconds=verify_seconds,
        resumed_split=resumed,counts=report['counts'],split_contract_id=report['contract_id'],sqlite_cache_mib=cache,
        database_bytes=(case/'split/split.sqlite').stat().st_size,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(case/'worker_result.json',result);progress('complete')


def artifact_hashes(case):
    paths=['sources.json','web.parquet','code.parquet','web.parquet.done','code.parquet.done','split/manifest.json','split/.done','split/split.sqlite']
    return {name:sha256_file(case/name) for name in paths}


def validate_case(case):
    marker=json.loads((case/'case.done').read_text())
    if marker['result_sha256']!=sha256_file(case/'result.json') or marker['artifacts']!=artifact_hashes(case):raise ValueError('Measured case content changed')
    return json.loads((case/'result.json').read_text())


def stop(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def validate_sizes(sizes):
    if not sizes or len(set(sizes))!=len(sizes) or any(type(n) is not int or not 100<=n<=3000000 for n in sizes):
        raise ValueError('Unique case sizes must be in 100..3000000')


def case_usage(case):
    total=journal=0
    for p in case.rglob('*'):
        try:
            if not p.is_file():continue
            size=p.stat().st_size
        except FileNotFoundError:continue  # Worker may atomically rename a journal/output.
        total+=size
        if p.name.endswith('-journal'):journal+=size
    return total,journal


def audit_running():
    registration=resolve('logs/source_audit_job.json')
    if not registration.exists():return False
    r=json.loads(registration.read_text());state=resolve(r['state_file'])
    if not state.exists():return False
    s=json.loads(state.read_text());pid=s.get('pid',0);cmd=Path(f'/proc/{pid}/cmdline')
    try:return s.get('phase')=='running' and b'audit_source_inventory.py' in cmd.read_bytes()
    except (FileNotFoundError,ProcessLookupError):return False


def run(output,sizes,*,max_case_seconds=1800,max_rss_gib=4,max_case_gib=8,min_free_gib=200,sqlite_cache_mib=64):
    import math
    validate_sizes(sizes)
    if any(n>100000 for n in sizes) and audit_running():
        raise RuntimeError('Large split measurement waits until the active source audit finishes')
    if any(not math.isfinite(x) or x<=0 for x in [max_case_seconds,max_rss_gib,max_case_gib]) or min_free_gib<0 or type(sqlite_cache_mib) is not int or not 1<=sqlite_cache_mib<=1024:
        raise ValueError('Invalid measurement resource limits')
    out=resolve(output);out.mkdir(parents=True,exist_ok=True);lock=(out/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:lock.close();raise
    try:
        config=dict(version=2,sizes=sizes,sqlite_cache_mib=sqlite_cache_mib,generator='streamed-short-text+repo5+cross-duplicate101-v2',
            environment=dict(python=platform.python_version(),sqlite=sqlite3.sqlite_version,pyarrow=pa.__version__),
            code={p:sha256_file(resolve(p)) for p in ['scripts/measure_split.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py']})
        signature=out/'run.json'
        if signature.exists():
            if json.loads(signature.read_text())!=config:raise ValueError('Measurement input/code/config changed; use a new output')
        else:
            if any(p.name!='.lock' for p in out.iterdir()):raise ValueError('Unrecognized measurement output')
            atomic_json(signature,config)
        if (out/'.done').exists():
            old=json.loads((out/'summary.json').read_text())
            if json.loads((out/'.done').read_text())!={'summary_sha256':sha256_file(out/'summary.json')}:raise ValueError('Scale summary changed')
            for size in sizes:validate_case(out/str(size))
            return old
        results=[]
        for size in sizes:
            case=out/str(size);case.mkdir(exist_ok=True)
            if (case/'case.done').exists():results.append(validate_case(case));continue
            attempts=sorted(case.glob('attempt_*.json'));number=len(attempts)+1
            attempt=dict(number=number,started_utc=datetime.now(timezone.utc).isoformat(),max_case_seconds=max_case_seconds,
                max_rss_gib=max_rss_gib,max_case_gib=max_case_gib,min_free_gib=min_free_gib)
            atomic_json(case/f'attempt_{number:03}.json',attempt)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            with (case/f'console_{number:03}.log').open('w') as stream:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(case),'--size',str(size),'--sqlite-cache-mib',str(sqlite_cache_mib)],
                    env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic();reason=None;peak_rss=peak_total=peak_journal=0
            try:
                while proc.poll() is None:
                    elapsed=time.monotonic()-started;status_path=Path(f'/proc/{proc.pid}/status')
                    if status_path.exists():
                        try:
                            values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status_path.read_text().splitlines() if ':' in line}
                            peak_rss=max(peak_rss,int(values.get('VmHWM','0 kB').split()[0]))
                        except (FileNotFoundError,ProcessLookupError):pass
                    total,journal=case_usage(case)
                    peak_total=max(peak_total,total);peak_journal=max(peak_journal,journal)
                    if elapsed>=max_case_seconds:reason='time'
                    elif peak_rss>max_rss_gib*1024**2:reason='rss'
                    elif total>max_case_gib*1024**3:reason='case_bytes'
                    elif shutil.disk_usage(out).free<min_free_gib*1024**3:reason='free_space'
                    atomic_json(out/'state.json',dict(phase='running',case_documents=size,attempt=number,pid=proc.pid,
                        elapsed_seconds=elapsed,max_rss_kib=peak_rss,sampled_peak_case_bytes=peak_total,sampled_peak_journal_bytes=peak_journal))
                    if reason:stop(proc);break
                    time.sleep(.1)
            finally:
                if proc.poll() is None:stop(proc)
            # Fast workers may exit between polls: check their exact final HWM too.
            if proc.returncode==0:
                final=json.loads((case/'worker_result.json').read_text())
                peak_rss=max(peak_rss,final['max_rss_kib'])
                total,journal=case_usage(case);peak_total=max(peak_total,total);peak_journal=max(peak_journal,journal)
                if peak_rss>max_rss_gib*1024**2:reason=reason or 'rss_at_completion'
                elif total>max_case_gib*1024**3:reason=reason or 'case_bytes_at_completion'
                elif shutil.disk_usage(out).free<min_free_gib*1024**3:reason=reason or 'free_space_at_completion'
            attempt.update(elapsed_seconds=time.monotonic()-started,returncode=proc.returncode,pause_reason=reason,
                sampled_max_rss_kib=peak_rss,sampled_peak_case_bytes=peak_total,sampled_peak_journal_bytes=peak_journal)
            atomic_json(case/f'attempt_{number:03}.json',attempt)
            if reason:
                result=dict(status='paused',contract_id=identity(config),config=config,completed_cases=results,case_documents=size,pause_reason=reason,
                    scope='Resource pause; generated inputs/SQLite partial preserved. Same-contract resume allowed; not full-scale validation.')
                atomic_json(out/'summary.json',result);atomic_json(out/'state.json',dict(phase='paused',case_documents=size,reason=reason));return result
            if proc.returncode!=0:
                atomic_json(out/'state.json',dict(phase='error',case_documents=size,returncode=proc.returncode))
                raise RuntimeError('Scale worker failed; inspect '+str(case/f'console_{number:03}.log'))
            value=json.loads((case/'worker_result.json').read_text());history=[json.loads(p.read_text()) for p in sorted(case.glob('attempt_*.json'))]
            value.update(attempts=len(history),cumulative_attempt_seconds=sum(x['elapsed_seconds'] for x in history),
                sampled_peak_case_bytes=max(x['sampled_peak_case_bytes'] for x in history),
                sampled_peak_journal_bytes=max(x['sampled_peak_journal_bytes'] for x in history))
            atomic_json(case/'result.json',value);atomic_json(case/'case.done',dict(result_sha256=sha256_file(case/'result.json'),artifacts=artifact_hashes(case)))
            results.append(value)
        result=dict(status='bounded_synthetic_verified',contract_id=identity(config),config=config,cases=results,
            scope='Explicit synthetic case sizes only. Resumed attempts and sampled peaks are labelled; no corpus-throughput/storage extrapolation.')
        atomic_json(out/'summary.json',result);atomic_json(out/'.done',dict(summary_sha256=sha256_file(out/'summary.json')))
        atomic_json(out/'state.json',dict(phase='complete',cases=len(results)));return result
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--worker');ap.add_argument('--size',type=int)
    ap.add_argument('--output');ap.add_argument('--sizes',nargs='+',type=int,default=[10000,100000])
    ap.add_argument('--sqlite-cache-mib',type=int,default=64);ap.add_argument('--max-case-seconds',type=float,default=1800)
    ap.add_argument('--max-rss-gib',type=float,default=4);ap.add_argument('--max-case-gib',type=float,default=8);ap.add_argument('--min-free-gib',type=float,default=200);a=ap.parse_args()
    if a.worker:worker(a.worker,a.size,a.sqlite_cache_mib)
    else:
        if not a.output:ap.error('--output required')
        r=run(a.output,a.sizes,max_case_seconds=a.max_case_seconds,max_rss_gib=a.max_rss_gib,max_case_gib=a.max_case_gib,
              min_free_gib=a.min_free_gib,sqlite_cache_mib=a.sqlite_cache_mib)
        print(json.dumps(dict(status=r['status'],cases=r.get('cases',r.get('completed_cases')),pause_reason=r.get('pause_reason'))))
        if r['status']=='paused':sys.exit(2)
