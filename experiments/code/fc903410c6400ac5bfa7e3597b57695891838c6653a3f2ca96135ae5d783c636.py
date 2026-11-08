"""Read-only30s rehearsal milestones/stall monitor. No LLM or push calls.

Uses producer checkpoint certificates, not another full payload SHA pass.
Observer completion means observation finished, not model/readiness acceptance.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file, identity


def read_rows(path):
    lines=Path(path).read_text().splitlines(); rows=[]
    for i,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1: raise
    return rows


def summarize(rows):
    train={}; evaluation={}
    for row in rows:
        step=row['step']
        if type(step) is not int or not 1<=step<=5000:
            raise ValueError('Observed step outside actual rehearsal')
        if 'loss' in row:
            for k in ['loss','gnorm','lr','tok_s','tokens_b']:
                if type(row.get(k)) not in (int,float) or not math.isfinite(row[k]):
                    raise ValueError('Nonfinite/incomplete observed train record')
            if row['tok_s']<=0 or row['gnorm']<0 or row['lr']<0 or not math.isclose(row['tokens_b']*1e9,step*2097152,abs_tol=1):
                raise ValueError('Observed train token/rate geometry differs')
            train[step]=row
        if 'by_source' in row:
            if len(row['by_source'])!=6 or set(row['by_category'])!={'ko','en','math','code'}:
                raise ValueError('Observed source/category coverage differs')
            for group in ['by_source','by_category']:
                if sum(r['valid_tokens'] for r in row[group].values())!=row['valid_tokens']:
                    raise ValueError('Observed evaluation target sum differs')
                if not math.isclose(sum(r['nll_sum'] for r in row[group].values()),row['nll_sum'],rel_tol=1e-12):
                    raise ValueError('Observed evaluation NLL sum differs')
                for r in row[group].values():
                    if r['valid_tokens']<1 or not math.isfinite(r['nll_sum']) or not math.isclose(r['holdout_loss'],r['nll_sum']/r['valid_tokens'],rel_tol=1e-12):
                        raise ValueError('Observed evaluation weighted NLL differs')
            if row['valid_tokens']!=1572864 or not math.isclose(row['holdout_loss'],row['nll_sum']/row['valid_tokens'],rel_tol=1e-12):
                raise ValueError('Observed full probe geometry differs')
            evaluation[step]=row
    # First resumed log at10 divides six actual updates by configured10 and is
    # inflated. The interval ending260 includes evaluation250 (and at510 save500).
    steady=[r['tok_s'] for step,r in sorted(train.items()) if step>=20 and step%250!=10]
    return dict(last_train=train[max(train)] if train else None,
                stable_median_tokens_s=statistics.median(steady[-10:]) if steady else None,
                train_points=len(train),evaluations=evaluation)


def producer_certificate(source, path, training_id):
    proof=json.loads(path.read_text()); step=int(path.stem.split('_')[1])
    checkpoint=source/'training'/path.stem
    if (proof['status']!='passed_checkpoint_storage' or proof['step']!=step or
            resolve(proof['checkpoint'])!=checkpoint or proof['training_contract_id']!=training_id or
            not (checkpoint/'.complete').is_file()):
        raise ValueError('Observed producer checkpoint certificate differs')
    records={r['path']:r for r in proof['files']}
    if len(records)!=len(proof['files']) or proof['bytes']!=sum(r['bytes'] for r in records.values()):
        raise ValueError('Observed checkpoint file list differs')
    for required in ['meta.json','.complete','.metadata',*[f'rng_rank{r}.pt' for r in range(8)]]:
        if required not in records:raise ValueError('Observed checkpoint metadata/RNG missing')
    meta=checkpoint/'meta.json'
    if sha256_file(meta)!=records['meta.json']['sha256']:
        raise ValueError('Observed checkpoint metadata changed')
    metadata=json.loads(meta.read_text())
    if metadata['step']!=step or metadata['training_contract']['id']!=training_id:
        raise ValueError('Observed checkpoint training identity differs')
    return dict(step=step,bytes=proof['bytes'],certificate=artifact_path(path),sha256=sha256_file(path),
                scope='Producer fresh payload SHA/fsync receipt observed; observer rehashes metadata only.')


def progress_state(snapshot, history, now, *, source_phase, live, stall_seconds=1200):
    last=snapshot['last_train']; step=last['step'] if last else 0
    fingerprint=[step,sorted(map(int,snapshot['evaluations'])),sorted(snapshot['checkpoints'])]
    if history.get('fingerprint')!=fingerprint:
        history=dict(history,fingerprint=fingerprint,last_progress_unix=now)
    if source_phase in ('error','failed','paused'): state=source_phase
    elif source_phase=='complete': state='completed_waiting_proof'
    elif not live: state='unexpected_exit'
    elif now-history['last_progress_unix']>=stall_seconds: state='suspected_stall'
    else: state='running'
    return state,history


def main(a):
    root,source=resolve(a.run_dir),resolve(a.source_run)
    root.relative_to(resolve('runs'));root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(source_run=artifact_path(source),interval=30,stall_seconds=1200,
                    code={'scripts/observe_rehearsal.py':sha256_file(Path(__file__))},
                    max_rss_mib=512,max_seconds=172800,
                    scope='Read-only local milestones and suspected stalls; no training/restart/pruning/upload/LLM/push.')
        cp=root/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Observer contract changed; use a new path')
        atomic_json(cp,config);(root/'pid').write_text(str(os.getpid())+'\n')
        cursor=root/'cursor.json';history=json.loads(cursor.read_text()) if cursor.exists() else dict(seen=[],certificates={})
        start=time.monotonic()
        def event(key,kind,**kw):
            if key in history['seen']:return
            record=dict(utc=datetime.now(timezone.utc).isoformat(),stage='rehearsal_observer',event=kind,event_id=key,source_run=artifact_path(source),**kw)
            for path in [root/'events.jsonl',resolve('logs/pipeline_events.jsonl')]:
                with path.open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            history['seen'].append(key)
        while True:
            try:
                rss=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
                if time.monotonic()-start>config['max_seconds'] or rss>config['max_rss_mib']*2**20:
                    raise RuntimeError('Observer resource budget')
                state=json.loads((source/'state.json').read_text())
                pid=int((source/'pid').read_text())
                try:
                    proc=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
                    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                    live=proc!='Z' and b'scripts/run_rehearsal.py' in cmd and source.name.encode() in cmd
                except FileNotFoundError:live=False
                summary=summarize(read_rows(source/'training/log.jsonl'))
                tid=json.loads((source/'training/training_contract.json').read_text())['id']
                checkpoints={}
                for path in sorted((source/'checkpoint_certificates').glob('*.json')):
                    receipt=producer_certificate(source,path,tid)
                    before=history['certificates'].setdefault(path.name,receipt['sha256'])
                    if before!=receipt['sha256']:raise ValueError('Previously observed certificate changed')
                    checkpoints[path.name]=receipt
                    event('checkpoint:'+path.stem,'checkpoint_certified',step=receipt['step'],bytes=receipt['bytes'],certificate=receipt['certificate'])
                summary['checkpoints']=checkpoints
                for step,row in sorted(summary['evaluations'].items()):
                    event('evaluation:'+str(step),'source_evaluation_completed',step=step,valid_tokens=row['valid_tokens'],
                          by_category={k:v['holdout_loss'] for k,v in row['by_category'].items()},probe_eval_seconds=row.get('probe_eval_seconds'),
                          scope=row['probe_scope'])
                last=summary['last_train'];step=last['step'] if last else 0
                if step:
                    event('milestone:'+str(step//500*500),'training_milestone_observed',step=step,trained_tokens=step*2097152)
                kind,history=progress_state(summary,history,time.time(),source_phase=state.get('phase'),live=live)
                previous=history.get('source_state')
                if previous!=kind:
                    event('state:'+str(time.time_ns()),'training_observation_state_changed',previous=previous,state=kind,step=step)
                history['source_state']=kind
                free=shutil.disk_usage(source).free
                if free<200*2**30:event('low_disk:'+str(step//500),'disk_reserve_below_threshold',step=step,free_bytes=free)
                out=dict(phase='running',case=kind,source_pid=pid,last_step=step,total_steps=5000,done=step,total=5000,
                         source_live=live,stable_median_tokens_s=summary['stable_median_tokens_s'],last_train=last,
                         evaluations_completed=len(summary['evaluations']),checkpoints_certified=len(checkpoints),
                         disk_free_bytes=free,last_progress_unix=history['last_progress_unix'])
                atomic_json(root/'observation.json',summary)
                atomic_json(root/'state.json',out);atomic_json(cursor,history)
                if (source/'.done').exists():
                    marker=json.loads((source/'.done').read_text())
                    if marker!=dict(report_sha256=sha256_file(source/'report.json'),contract_sha256=sha256_file(source/'contract.json')):
                        raise ValueError('Source completion proof changed')
                    source_report=json.loads((source/'report.json').read_text())
                    report=dict(status='observation_complete',source_status=source_report['status'],observation=out,
                                source_report_sha256=sha256_file(source/'report.json'),scope=config['scope'])
                    atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(cp)))
                    atomic_json(root/'state.json',dict(phase='complete',done=step,total=5000));break
            except Exception as exc:
                detail=type(exc).__name__+': '+str(exc)
                event('error:'+identity(detail),'observer_check_error',state='error',reason=detail)
                atomic_json(root/'state.json',dict(phase='running',case='observation_error',reason=detail))
                atomic_json(cursor,history)
                if isinstance(exc,RuntimeError) and str(exc)=='Observer resource budget':raise
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--source-run',default='runs/1b_rehearsal_final_v3');main(p.parse_args())
