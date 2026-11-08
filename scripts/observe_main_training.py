"""30-second read-only main training progress observer; no LLM, restart or pruning."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file, identity
from scripts.audit_main_training import finite, integer


def rows_from_trace(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.stat().st_size > 128*1024**2:
        raise ValueError('Observer trace budget exceeded')
    raw = path.read_bytes()
    # Ignore only a non-newline-terminated append in flight, not malformed rows.
    lines = raw.splitlines()
    if raw and not raw.endswith(b'\n'):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def summarize(rows, config):
    last = None
    rates = []
    evaluations = {}
    train_rows = regressions = 0
    for row in rows:
        step = integer(row['step'], 'observed step')
        if step > config['max_steps']:
            raise ValueError('Observed step exceeds target')
        if 'loss' in row:
            for key in ['loss','gnorm','lr','tok_s','tokens_b']:
                finite(row[key], key)
            if (step % config['log_every'] or row['tok_s'] <= 0 or
                    not math.isclose(row['tokens_b']*1e9, step*config['global_tokens'], rel_tol=0, abs_tol=1)):
                raise ValueError('Observed token/log geometry differs')
            if last:
                if step <= last['step']:
                    regressions += 1
                    rates = []
                if (step == last['step']+config['log_every'] and
                        last['step'] % config['eval_every'] and last['step'] % config['ckpt_every']):
                    rates.append(row['tok_s'])
            last = row
            train_rows += 1
        elif 'by_source' in row:
            finite(row['holdout_loss'], 'holdout loss')
            finite(row['nll_sum'], 'holdout nll')
            integer(row['valid_tokens'], 'holdout targets')
            if not math.isclose(row['holdout_loss'],row['nll_sum']/row['valid_tokens'],rel_tol=1e-12):
                raise ValueError('Observed weighted holdout differs')
            for group in ['by_source','by_category']:
                for value in row[group].values():
                    finite(value['holdout_loss'], 'source/category holdout loss')
            evaluations[str(step)] = dict(step=step, holdout_loss=row['holdout_loss'],
                by_category={k:v['holdout_loss'] for k,v in row['by_category'].items()},
                valid_tokens=row['valid_tokens'], generation_present='fixed_generations' in row)
        else:
            raise ValueError('Unknown observed trace row')
    rate = statistics.median(rates[-10:]) if len(rates) >= 3 else None
    return dict(last_train=last, logged_step=last['step'] if last else 0, train_records=train_rows,
                log_step_regressions=regressions, steady_tokens_s=rate, evaluations=evaluations,
                remaining_compute_seconds=(config['max_steps']-last['step'])*config['global_tokens']/rate if rate else None,
                rate_scope='Last up to10 steady log intervals; first/after-eval/after-save/discontinuous intervals excluded. ETA excludes all future overhead.')


def progress_state(summary, history, now, *, supervisor_live, worker_live, phase, worker_pid, stall_seconds=1200):
    fingerprint=[summary['logged_step'],summary['train_records'],sorted(summary['evaluations']),worker_pid]
    history=dict(history)
    if history.get('fingerprint') != fingerprint:
        history.update(fingerprint=fingerprint,last_progress_unix=now)
    if phase in ('error','failed','paused'):
        state=phase
    elif phase=='complete':
        state='completed_waiting_proof'
    elif not supervisor_live or not worker_live:
        state='unexpected_exit'
    elif now-history['last_progress_unix'] >= stall_seconds:
        state='suspected_stall'
    else:
        state='running'
    return state,history


def live(pid, token, source):
    try:
        p=Path('/proc')/str(pid)
        state=(p/'stat').read_text().split(') ',1)[1].split()[0]
        cmd=(p/'cmdline').read_bytes()
        return state!='Z' and token.encode() in cmd and source.name.encode() in cmd
    except FileNotFoundError:
        return False


def observe(source, output):
    source,output=resolve(source),resolve(output)
    output.relative_to(resolve('runs'))
    if output.is_relative_to(source):
        raise ValueError('Observer output must be independent of managed training')
    c=json.loads((source/'contract.json').read_text());cfg=json.loads((source/'main.json').read_text())
    if c['purpose']!='main_training' or not c['main_training_authorized'] or cfg['max_steps']!=57000:
        raise ValueError('Expected authorized actual3B main run')
    output.mkdir(parents=True,exist_ok=True)
    binding=dict(source_run=artifact_path(source),source_contract_sha256=sha256_file(source/'contract.json'),
        source_config_sha256=sha256_file(source/'main.json'),interval=30,stall_seconds=1200,resource_interval=600,
        max_rss_mib=512,max_seconds=22*86400,code={p:sha256_file(resolve(p)) for p in
             ['scripts/observe_main_training.py','scripts/audit_main_training.py','scripts/report_resource_usage.py']},
        scope='Read-only progress and producer receipt observation; no training/restart/deletion/upload/LLM/push; no Goal completion claim.')
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cp=output/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=binding:
            raise ValueError('Observer binding changed; preserve and use a new observer run')
        atomic_json(cp,binding);(output/'pid').write_text(str(os.getpid())+'\n')
        cursor=output/'cursor.json'
        history=json.loads(cursor.read_text()) if cursor.exists() else dict(seen=[],certificate_hashes={})
        start=time.monotonic()
        def event(key,kind,**kwargs):
            if key in history['seen']:
                return
            record=dict(utc=datetime.now(timezone.utc).isoformat(),stage='main_training_observer',
                        event=kind,event_id=key,source_run=artifact_path(source),**kwargs)
            for path in [output/'events.jsonl',resolve('logs/pipeline_events.jsonl')]:
                with path.open('a') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            history['seen'].append(key)
        while True:
            poll_start=time.monotonic()
            try:
                rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
                if rss>binding['max_rss_mib']*2**20 or time.monotonic()-start>binding['max_seconds']:
                    raise RuntimeError('Observer resource budget')
                if sha256_file(source/'contract.json')!=binding['source_contract_sha256'] or sha256_file(source/'main.json')!=binding['source_config_sha256']:
                    raise ValueError('Live main contract/configuration changed')
                state=json.loads((source/'state.json').read_text())
                pid=int((source/'pid').read_text());worker=state.get('worker_pid',0)
                summary=summarize(rows_from_trace(resolve(cfg['out_dir'])/'log.jsonl'),cfg)
                kind,history=progress_state(summary,history,time.time(),supervisor_live=live(pid,'supervise_main_training.py',source),
                    worker_live=live(worker,'train.monitored',source),phase=state.get('phase'),worker_pid=worker,
                    stall_seconds=binding['stall_seconds'])
                for path in sorted((source/'checkpoint_certificates').glob('*.json')):
                    digest=sha256_file(path)
                    prior=history['certificate_hashes'].get(path.name)
                    if prior and prior!=digest:
                        raise ValueError('Published producer certificate changed')
                    if not prior:
                        proof=json.loads(path.read_text())
                        if proof['status']!='passed_main_checkpoint_storage' or proof['step']!=int(path.stem.split('_')[1]):
                            raise ValueError('Unexpected producer certificate')
                        history['certificate_hashes'][path.name]=digest
                        event('certificate:'+path.name,'checkpoint_certified',step=proof['step'],bytes=proof['bytes'],
                              certificate=artifact_path(path),scope='Producer receipt; not a fresh payload rehash or current residency claim.')
                for step,evaluation in summary['evaluations'].items():
                    event('evaluation:'+step,'source_evaluation_observed',**evaluation)
                if history.get('source_state')!=kind:
                    event('state:'+str(time.time_ns()),'training_observation_state_changed',previous=history.get('source_state'),state=kind,step=summary['logged_step'])
                history['source_state']=kind
                if summary['logged_step']:
                    event('milestone:'+str(summary['logged_step']//500),'training_milestone_observed',step=summary['logged_step'])
                if time.time()-history.get('last_resource_unix',0)>=binding['resource_interval']:
                    from scripts.report_resource_usage import snapshot
                    resources=snapshot();atomic_json(output/'resources.json',resources)
                    event('resources:'+resources['utc'],'resource_sample',snapshot=artifact_path(output/'resources.json'),resources=resources)
                    history['last_resource_unix']=time.time()
                retention=json.loads((source/'retention_state.json').read_text()) if (source/'retention_state.json').exists() else None
                out=dict(phase='running',case=kind,done=summary['logged_step'],total=cfg['max_steps'],
                         supervisor_pid=pid,worker_pid=worker,last_progress_unix=history['last_progress_unix'],
                         certificates_ever_observed=len(history['certificate_hashes']),summary=summary,
                         last_retention_report=retention,retention_scope='Last manager report, not an independent current storage audit.')
                atomic_json(output/'state.json',out);atomic_json(cursor,history)
                if (source/'.done').exists():
                    marker=json.loads((source/'.done').read_text())
                    if (marker.get('report_sha256')!=sha256_file(source/'report.json') or
                            marker.get('contract_sha256')!=binding['source_contract_sha256']):
                        raise ValueError('Source completion proof changed')
                    report=dict(status='observation_complete',source_report_sha256=marker['report_sha256'],observation=out,scope=binding['scope'])
                    atomic_json(output/'report.json',report)
                    atomic_json(output/'.done',dict(report_sha256=sha256_file(output/'report.json'),contract_sha256=sha256_file(cp)))
                    atomic_json(output/'state.json',dict(phase='complete',done=summary['logged_step'],total=cfg['max_steps']))
                    event('observation_complete','main_target_report_observed_final_review_pending',step=summary['logged_step'])
                    atomic_json(cursor,history)
                    break
            except Exception as exc:
                detail=type(exc).__name__+': '+str(exc)
                event('error:'+identity(detail),'observer_check_error',reason=detail)
                atomic_json(output/'state.json',dict(phase='running',case='observation_error',reason=detail))
                atomic_json(cursor,history)
                if isinstance(exc,RuntimeError) and str(exc)=='Observer resource budget':
                    raise
            time.sleep(max(0,binding['interval']-(time.monotonic()-poll_start)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True)
    p.add_argument('--run-dir',required=True)
    a=p.parse_args();observe(a.source_run,a.run_dir)
