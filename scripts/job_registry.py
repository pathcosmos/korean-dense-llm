"""Small independent-job observations; completion and validation are distinct."""
import json,time
from pathlib import Path
from common.data_contracts import resolve,sha256_file,atomic_json
TERMINAL={'complete','measurement_complete','needs_review','failed'}
def validate(spec):
    if not isinstance(spec,dict) or not isinstance(spec.get('run_id'),str) or not spec['run_id'] or '/' in spec['run_id']:raise ValueError('Invalid run_id')
    for key in ['state_file','pid_file','log_file','certificate','report_file']:
        if not isinstance(spec.get(key),str) or not spec[key]:raise ValueError('Missing '+key)
    if not spec.get('proofs'):raise ValueError('Require completion content proofs')
    if not any(p.get('path')==spec['report_file'] for p in spec['proofs']):raise ValueError('Report must be content bound')
    for p in spec['proofs']:
        if not isinstance(p.get('path'),str) or not isinstance(p.get('field'),list) or not p['field'] or any(not isinstance(k,str) for k in p['field']):raise ValueError('Invalid proof')
    if type(spec.get('stall_seconds',1200)) not in (int,float) or not 30<=spec.get('stall_seconds',1200)<=86400:raise ValueError('Invalid stall budget')
    return spec

def statuses(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('status','readiness','production_adoption') and isinstance(v,str):yield v
            elif isinstance(v,(list,dict)):yield from statuses(v)
    elif isinstance(value,list):
        for v in value:yield from statuses(v)

def observe(spec,alive,tail):
    try:
        validate(spec);sp=resolve(spec['state_file']);pp=resolve(spec['pid_file']);lp=resolve(spec['log_file']);marker=resolve(spec['certificate']);rp=resolve(spec['report_file'])
        state=json.loads(sp.read_text()) if sp.exists() else {};pid=int(pp.read_text()) if pp.exists() else 0
        phase=state.get('status',state.get('phase',state.get('event')));live=bool(pid and alive(pid))
        if live and spec.get('pid_cmd_contains'):
            try:live=spec['pid_cmd_contains'].encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
            except OSError:live=False
        verdict=None;reason=None
        if marker.exists():
            proof=json.loads(marker.read_text())
            for binding in spec['proofs']:
                expected=proof
                for key in binding['field']:expected=expected[key]
                p=resolve(binding['path'])
                if p.stat().st_size>16*1024**2:raise ValueError('Monitor artifact exceeds 16MiB')
                if sha256_file(p)!=expected:raise ValueError('Completion content mismatch')
            observed=set(statuses(json.loads(rp.read_text())))
            if observed & {'failed','error'}:kind='failed'
            elif spec.get('requires_review') or observed & {'needs_review','pending','diagnosis_complete_adoption_pending','measurement_complete_optimization_candidates'}:kind='needs_review'
            elif observed and observed<= {'passed','passed_bounded_pair'}:kind='complete'
            else:kind='measurement_complete'
            verdict=sorted(observed)
        elif phase in ('error','failed','paused'):kind=phase
        elif phase in ('complete','completed','diagnostic_complete'):kind='error';reason='Terminal state without certificate'
        elif any(t in tail(lp) for t in ['Traceback (most recent call last)','CUDA out of memory']):kind='error'
        elif not live:kind='unexpected_exit'
        else:
            heartbeat=max((p.stat().st_mtime for p in [sp,lp] if p.exists()),default=0)
            kind='stalled' if time.time()-heartbeat>spec.get('stall_seconds',1200) else 'running'
        return dict(state=kind,done=state.get('done',0),total=state.get('total',0),pid=pid,phase=phase,run_id=spec['run_id'],case=state.get('case'),verdict=verdict,reason=reason)
    except (ValueError,TypeError,KeyError,OSError) as exc:
        return dict(state='error',done=0,total=0,run_id=spec.get('run_id','invalid'),reason=type(exc).__name__)

def load_jobs(registry,alive,tail):
    if not Path(registry).exists():return {}
    try:
        d=json.loads(Path(registry).read_text())
        if d.get('schema_version')!=1 or not isinstance(d.get('jobs'),list):raise ValueError('Invalid registry')
        result={}
        for spec in d['jobs']:
            name='job:'+spec.get('run_id','invalid')
            if name in result:raise ValueError('Duplicate job')
            result[name]=observe(spec,alive,tail)
        return result
    except (ValueError,TypeError,AttributeError,OSError):return {'job:registry':dict(state='invalid_registration',done=0,total=0)}


def ensure_watcher(root):
    import fcntl,subprocess,sys
    root=Path(root);log=root/'logs'
    with (log/'pipeline_watch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
        fcntl.flock(lock,fcntl.LOCK_UN)
    with (log/'pipeline_watch.log').open('a') as stream:
        proc=subprocess.Popen([sys.executable,'-u','scripts/watch_data_pipeline.py','--interval','30'],cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (log/'pipeline_watch.pid').write_text(str(proc.pid)+'\n')
    return proc.pid
