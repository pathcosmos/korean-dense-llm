"""Local30s dependency coordinator: actual F6 mix review -> gated1B rehearsal.

No LLM calls, corpus deletion, external upload, or3B main-training command.
The training handoff still requires final trained-model evaluation/readiness review.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,artifact_path,atomic_json,sha256_file
from scripts.mix_review_policy import POLICY,choose_mix
from scripts.run_native_evaluation import certified


def event(root,name,**fields):
    from datetime import datetime,timezone
    value=dict(utc=datetime.now(timezone.utc).isoformat(),stage='pretraining_continuation',event=name,run_id=root.name,**fields)
    with resolve('logs/pipeline_events.jsonl').open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def review_f6(root,f6,contract):
    from scripts.run_final_tokens import verify_mix_review
    cp=f6/'contract.json';c=json.loads(cp.read_text())
    if sha256_file(cp)!=contract['f6_contract_sha256']:raise ValueError('Active F6 identity changed')
    for p,h in c['code'].items():
        if sha256_file(resolve(p))!=h:raise ValueError('Active F6 implementation changed')
    counts_path=f6/'counts.json';counts=json.loads(counts_path.read_text());out=resolve(c['settings']['output'])
    proof=json.loads((out/'.done').read_text());native=json.loads((out/'report.json').read_text())
    if proof!=dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')):
        raise ValueError('Actual whole token completion proof differs')
    if native['status']!='complete' or native['shards']!=native['planned_shards'] or counts['full_report_sha256']!=sha256_file(out/'report.json'):
        raise ValueError('Mix review requires actual complete source coverage')
    native_contract=json.loads((out/'contract.json').read_text())
    if native_contract['experimental']:raise ValueError('Experimental tokens cannot feed rehearsal')
    f5=resolve(c['settings']['tokenizer_run']);trained,tc=certified(f5,'passed')
    if counts['tokenizer_sha256']!=trained['tokenizer_sha256'] or sum(v['train_docs']+v['holdout_docs'] for v in counts['source_counts'].values())!=tc['total_documents']:
        raise ValueError('Full actual document/tokenizer coverage differs')
    selected=choose_mix(counts,tc['categories'])
    review=dict(decision='proceed',counts_sha256=sha256_file(counts_path),contract_sha256=sha256_file(cp),
                policy_sha256=sha256_file(root/'mix_policy.json'),coordinator_contract_sha256=sha256_file(root/'contract.json'),**selected)
    path=f6/'mix_review.json'
    if path.exists():
        existing=json.loads(path.read_text())
        for key in ['decision','counts_sha256','contract_sha256','policy_sha256','weights','expected_exposures','actual_train_tokens','reason']:
            if existing.get(key)!=review[key]:raise ValueError('Existing mix review differs; preserve and review instead of overwriting')
        review=existing
    else:atomic_json(path,review)
    verify_mix_review(f6,tc['categories']);atomic_json(root/'mix_decision.json',review)
    event(root,'actual_mix_review_applied',weights=review['weights'],expected_exposures=review['expected_exposures'])


def live(pid,program):
    try:return program.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:return False


def main(a):
    from scripts.resource_admission import inspect
    root=resolve(a.run_dir);root.resolve().relative_to(resolve('runs').resolve())
    f6=resolve(a.token_run);native=resolve(a.native_eval_run);target=resolve(a.rehearsal_run);hooks=resolve(a.hooks_run)
    target.resolve().relative_to(resolve('runs').resolve())
    settings=dict(token_run=artifact_path(f6),native_eval_run=artifact_path(native),rehearsal_run=artifact_path(target),hooks_run=artifact_path(hooks),interval=30)
    if not root.exists():
        root.mkdir(parents=True);atomic_json(root/'mix_policy.json',POLICY)
        deps=['scripts/continue_pretraining.py','scripts/mix_review_policy.py','scripts/run_final_tokens.py','scripts/run_rehearsal.py',
              'train/rehearsal.py','train/parallel_contract.py','train/deterministic_train.py','train/probe.py','scripts/run_native_evaluation.py','scripts/hardware_profile.py','scripts/resource_admission.py']
        atomic_json(root/'contract.json',dict(version=1,settings=settings,code={p:sha256_file(resolve(p)) for p in deps},
                    f6_contract_sha256=sha256_file(f6/'contract.json'),policy_sha256=sha256_file(root/'mix_policy.json'),max_seconds=432000,
                    scope='Local actual mix review and1B rehearsal dependency handoff only; final trained native evaluation/readiness report remain required. No3B main training.'))
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    contract=json.loads((root/'contract.json').read_text())
    if contract['settings']!=settings:raise ValueError('Coordinator settings changed')
    (root/'pid').write_text(str(os.getpid()));start=time.monotonic();child=None;launched=False
    try:
        while True:
            for p,h in contract['code'].items():
                if sha256_file(resolve(p))!=h:raise ValueError('Coordinator dependency changed:'+p)
            if sha256_file(root/'mix_policy.json')!=contract['policy_sha256']:raise ValueError('Declared mix policy changed')
            if time.monotonic()-start>contract['max_seconds']:raise RuntimeError('Coordinator time budget')
            if (root/'HOLD').exists():raise RuntimeError('Coordinator HOLD present; dependent launch stopped, active outputs preserved')
            fs=json.loads((f6/'state.json').read_text())
            if fs['phase'] in ('error','failed','paused'):raise RuntimeError('F6 requires recovery:'+str(fs))
            if (hooks/'state.json').exists() and json.loads((hooks/'state.json').read_text())['phase'] in ('error','failed'):
                raise RuntimeError('Actual rehearsal hook integration requires recovery')
            if fs.get('case')=='await_mix_review' and not (root/'mix_decision.json').exists():review_f6(root,f6,contract)
            phase='await_actual_f6_and_native_completion'
            if (f6/'.done').exists() and (native/'.done').exists() and (hooks/'.done').exists() and fs['phase']=='complete' and json.loads((native/'state.json').read_text())['phase']=='complete':
                certified(f6,'passed');certified(native,'passed_actual_native_evaluation');certified(hooks,'passed_bounded_actual_hooks')
                if (target/'.done').exists():
                    report,_=certified(target,'passed_actual_rehearsal_training')
                    handoff=dict(status='training_handoff_complete',rehearsal_report_sha256=sha256_file(target/'report.json'),
                                 trained_tokens=report['counts']['committed_tokens'],scope=contract['scope'],
                                 pending=['Final trained native ko/en PPL','Source/generation trajectory review','3B readiness report and separate main-training confirmation'])
                    atomic_json(root/'report.json',handoff);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
                    atomic_json(root/'state.json',dict(phase='complete'));event(root,'rehearsal_training_handoff_complete');return
                pid_path=root/'rehearsal_controller.pid';pid=int(pid_path.read_text()) if pid_path.exists() else 0
                if pid and live(pid,'run_rehearsal.py'):
                    phase='rehearsal_running';launched=True
                elif launched or pid or target.exists():raise RuntimeError('Rehearsal controller stopped/partial output exists; inspect and resume explicitly, data preserved')
                else:
                    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                    hw=inspect()
                    if occupied or hw['memory']['cache_aware_admission']['headroom_gib']<384:phase='await_fresh_gpu_or_memory_budget'
                    else:
                        atomic_json(root/'rehearsal_launch_hardware.json',hw)
                        env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR='/tmp',PYTHONUNBUFFERED='1')
                        with (root/'rehearsal_controller.log').open('a') as log:
                            child=subprocess.Popen([sys.executable,'-u','scripts/run_rehearsal.py','--run-dir',str(target),'--token-run',str(f6),'--native-eval-run',str(native)],
                                                   stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                        pid_path.write_text(str(child.pid));launched=True;phase='rehearsal_starting'
                        from scripts.job_registry import validate
                        spec=dict(run_id=target.name,state_file=artifact_path(target/'state.json'),pid_file=artifact_path(pid_path),pid_cmd_contains='run_rehearsal.py',
                                  log_file=artifact_path(root/'rehearsal_controller.log'),certificate=artifact_path(target/'.done'),report_file=artifact_path(target/'report.json'),
                                  proofs=[dict(path=artifact_path(target/'report.json'),field=['report_sha256']),dict(path=artifact_path(target/'contract.json'),field=['contract_sha256'])],stall_seconds=1800)
                        atomic_json(root/'rehearsal_job.json',validate(spec))
                        subprocess.run([sys.executable,'scripts/register_pipeline_job.py','--spec',str(root/'rehearsal_job.json')],check=True)
                        event(root,'actual1b_rehearsal_started',pid=child.pid,model_recommendation='GPT-6 Astra / High',main3b_started=False)
            atomic_json(root/'state.json',dict(phase='running',case=phase,unix=time.time(),elapsed_seconds=time.monotonic()-start))
            time.sleep(30)
    except BaseException as exc:
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)));event(root,'coordinator_error',reason=str(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--token-run',default='runs/f6_tokens_v1');p.add_argument('--native-eval-run',default='runs/native_eval_validation_v1')
    p.add_argument('--rehearsal-run',default='runs/1b_rehearsal_final_v1');p.add_argument('--hooks-run',default='runs/rehearsal_hooks_actual_v2');main(p.parse_args())
