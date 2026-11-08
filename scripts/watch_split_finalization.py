"""Read-only30s observer through publication. No model calls or producer restarts.
The publisher performs all full DB checks; this observer checks its pinned receipts,
activity and terminal state, records exceptions, and updates STATUS on termination.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,sha256_file,identity,atomic_json,artifact_path


def read_small(path):
    p=Path(path)
    if p.stat().st_size>4*1024**2:raise ValueError('Observer metadata exceeds4MiB')
    return json.loads(p.read_text())


def verify_completion(run,expected_contract_sha,total):
    """Receipt consistency; does not repeat the publisher's full SQL/DB-hash scans."""
    run=Path(run);proof=read_small(run/'.done');contract=read_small(run/'contract.json');report=read_small(run/'report.json')
    if (sha256_file(run/'contract.json')!=expected_contract_sha or proof['contract_sha256']!=expected_contract_sha or
            proof['report_sha256']!=sha256_file(run/'report.json')):raise ValueError('Run completion proof mismatch')
    if report['status']!='passed':raise ValueError('Publisher validation did not pass')
    out=resolve(contract['output']).resolve();manifest=out/'manifest.json';m=read_small(manifest);marker=read_small(out/'.done')
    if resolve(report['manifest']).resolve()!=manifest or report['manifest_sha256']!=sha256_file(manifest):raise ValueError('Published manifest differs')
    if read_small(out/'publication.json')!=contract:raise ValueError('Publication ownership differs')
    if (m['status']!='validated' or identity(m['contract'])!=m['contract_id'] or
            marker!=dict(contract_id=m['contract_id'],manifest_sha256=sha256_file(manifest)) or
            report['split_contract_id']!=m['contract_id']):raise ValueError('Split completion proof mismatch')
    for key in ['seed','eval_fraction']:
        if m['contract'][key]!=contract[key]:raise ValueError('Published split policy differs')
    if m['contract']['source_snapshot']!=contract['source_snapshot']:raise ValueError('Published input snapshot differs')
    if report['counts']!=m['counts'] or set(report['counts'])-{'train','eval'}:raise ValueError('Invalid train/eval counts')
    if any(type(n) is not int or n<0 for n in report['counts'].values()):raise ValueError('Invalid counts')
    if sum(report['counts'].values())!=total or sum(f['rows'] for f in m['contract']['files'])!=total:raise ValueError('Input row coverage mismatch')
    perfile={label:sum(v.get(label,0) for v in m['per_file'].values()) for label in report['counts']}
    if perfile!=report['counts'] or len(m['per_file'])!=len(m['contract']['files']):raise ValueError('Per-file totals differ')
    db=resolve(m['database']['path']).resolve()
    if db!=out/'split.sqlite' or not db.is_file() or db.stat().st_size<=0 or report['database_sha256']!=m['database']['sha256']:
        raise ValueError('Published database binding differs')
    if sha256_file(resolve(contract['source_manifest']))!=contract['source_sha256']:raise ValueError('Source manifest changed')
    for file,digest in contract['code'].items():
        if sha256_file(resolve(file))!=digest:raise ValueError('Pinned producer changed: '+file)
    return dict(status='passed',source_run=str(run),counts=report['counts'],total_documents=total,
                manifest=str(manifest),manifest_sha256=report['manifest_sha256'],database_sha256=report['database_sha256'],database_bytes=db.stat().st_size,
                publisher_verification=report['verification'],receipt_verification='Pinned run/manifest/publication/source/code/count bindings verified. Publisher performed full DB SHA, integrity, cross-split and row coverage checks.',
                next='Review source distributions and final train-only tokenizer inputs. No next-stage job launched by observer.')


def activity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
        if s[0]=='Z':return None
        io=dict(line.split(': ') for line in (p/'io').read_text().splitlines())
        return [int(s[11])+int(s[12])]+[int(io[k]) for k in ['rchar','wchar','read_bytes','write_bytes']]
    except (OSError,ValueError,IndexError):return None


def progress_phase(builder):
    if builder is None:return None
    for name in ['split.sqlite.building','split.sqlite']:
        path=builder.parent/name
        if not path.exists():continue
        db=None
        try:
            db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.1)
            row=db.execute("SELECT value FROM progress WHERE key='phase'").fetchone()
            return json.loads(row[0]) if row else None
        except (OSError,sqlite3.Error):pass
        finally:
            if db:db.close()
    return None


def phase(worker,dbphase,previous):
    if worker in ('ingest','ingest_pending','assign','local_build','finalize','finalize_pending'):
        if dbphase=='complete':return 'validate_local'
        if dbphase:return dbphase
        return previous or worker
    return worker or previous or 'waiting'


def emit(event,**fields):
    item=dict(utc=datetime.now(timezone.utc).isoformat(),stage='f4_finalization_observer',event=event,**fields)
    with resolve('logs/pipeline_events.jsonl').open('a') as out:out.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(json.dumps(item,ensure_ascii=False),flush=True)


def status_note(heading,body):
    path=resolve('docs/STATUS.md')
    with resolve('logs/status_document.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);old=path.read_text();i=old.index('\n\n')+2
        stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        text=old[:i]+f'## {heading} ({stamp}, 로컬 관측기 기록)\n\n'+body+'\n\n'+old[i:]
        tmp=path.with_suffix('.observer.tmp');tmp.write_text(text);tmp.replace(path)


def watch(a):
    from scripts.hardware_profile import inspect
    source=resolve(a.source_run);root=resolve(a.output);root.mkdir(parents=True,exist_ok=False)
    hw=inspect();initial=read_small(source/'contract.json');expected=sha256_file(source/'contract.json')
    plan=read_small(source/'plan.json');builder=next(Path(plan['scratch']).glob('split-*/build.json'),None)
    if builder is None:raise ValueError('Missing source build contract')
    total=sum(f['rows'] for f in read_small(builder)['contract']['files'])
    own=dict(source_run=str(source),source_contract_sha256=expected,total_documents=total,interval_seconds=a.interval,
             stall_seconds=1200,hardware=hw,code={'scripts/watch_split_finalization.py':sha256_file(Path(__file__))},
             scope='Read-only activity and completion-receipt observer; no full DB scan or producer modification, no model invocation or external push.')
    atomic_json(root/'contract.json',own);previous=None;last_activity=None;last_move=time.monotonic();stall_reported=False;oom_before=None
    try:
        while True:
            state=read_small(source/'state.json');kind=state['phase'];worker=state.get('worker',{})
            if kind in ('paused','error'):raise RuntimeError('Source '+kind+': '+str(state.get('reason')))
            if kind=='complete':
                result=verify_completion(source,expected,total)
                if sha256_file(Path(__file__))!=own['code']['scripts/watch_split_finalization.py']:raise ValueError('Observer changed')
                atomic_json(root/'report.json',result)
                status_note('F4 최종 저장·검증 완료',f"전체{total:,}문서 분할·영구발행 완료. train={result['counts'].get('train',0):,}, eval={result['counts'].get('eval',0):,}. 실행기가 전체DB SHA·무결성/내용·group/repo교차0·row coverage 검증을 완료했고 관측기는 고정코드/입력/발행/완료증명을 대조했다. 근거 {artifact_path(root)}/report.json(.done). 다음은source별분포 검토→최종train-only tokenizer 준비(권장Sol / High). 실제분포/최종tokenizer/본학습은후속이며관측기가자동기동하지않았다.")
                emit('publication_validation_complete',counts=result['counts'],manifest=result['manifest'])
                atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
                atomic_json(root/'state.json',dict(phase='complete'));return
            pid=worker.get('pid')
            if pid is None and (source/'worker_state.json').exists():worker=read_small(source/'worker_state.json');pid=worker.get('pid')
            current_activity=activity(pid) if pid else None
            if current_activity is not None and current_activity!=last_activity:last_move=time.monotonic();stall_reported=False
            last_activity=current_activity
            current=phase(worker.get('phase'),progress_phase(builder),previous)
            if current!=previous:emit('phase_changed',previous=previous,phase=current);previous=current
            idle=time.monotonic()-last_move
            if idle>1200 and not stall_reported:
                emit('activity_stalled',phase=current,idle_seconds=idle);stall_reported=True
                status_note('F4 활동 정체 예외 관측',f'{current}에서CPU/IO활동이20분간진전하지않음. 원본/producer는보존하고로그진단필요. 관측기가생산작업을중지하거나재시작하지않음. 근거 {root}/state.json. 권장Sol / High.')
            events=dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
            oom=int(events['oom_kill'])
            if oom_before is not None and oom>oom_before:emit('container_oom_observed',previous=oom_before,count=oom)
            oom_before=oom
            controller_pid=int(resolve(a.pid_file).read_text())
            if activity(controller_pid) is None and kind=='running':
                if read_small(source/'state.json')['phase']=='complete':continue
                raise RuntimeError('Main controller exited without terminal proof')
            atomic_json(root/'state.json',dict(phase='running',case=current,worker_pid=pid,activity_idle_seconds=idle,
                        activity_stalled=stall_reported,source_phase=kind,oom_kill=oom,unix=time.time()))
            time.sleep(a.interval)
    except BaseException as exc:
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)))
        emit('observer_exception',error=str(exc))
        status_note('F4 최종 검증 감시 예외',f'관측예외: {str(exc)}. 실제완료로표시하지않음. 원본/진행산출물보존,실행기state와로그진단필요. 근거 {root}/state.json. 권장Sol / High.')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',default='runs/f4_full_v2');p.add_argument('--output',required=True)
    p.add_argument('--pid-file',default='logs/f4_full_v2.pid');p.add_argument('--interval',type=float,default=30);a=p.parse_args()
    if a.interval<1:p.error('Interval must be positive')
    watch(a)
