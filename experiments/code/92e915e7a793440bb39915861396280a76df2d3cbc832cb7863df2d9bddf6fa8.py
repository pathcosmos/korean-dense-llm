"""Assemble closed actual evidence for final review; never starts main training.

Wait mode is a30s local process. Report/.done attest assembly only, not final
quality acceptance or goal completion. A reviewer must inspect remaining items.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.run_native_evaluation import certified


def final_checks(rehearsal, native, monitored, validation):
    if rehearsal['counts']['committed_tokens'] != 10485760000:
        raise ValueError('Full1B rehearsal target not reached')
    if rehearsal['curves']['trained_tokens'] != 10485760000:
        raise ValueError('Full1B trajectory token coverage differs')
    if monitored['counts']['committed_tokens'] != 8388608:
        raise ValueError('Actual monitored3B four-update evidence missing')
    if native['rehearsal_status'] != rehearsal['status']:
        raise ValueError('Final native evaluation rehearsal lineage differs')
    if set(native['languages']) != {'ko','en'}:
        raise ValueError('Both final native languages required')
    for k, targets in [('ko',663905),('en',705634)]:
        r=native['languages'][k]
        if r['valid_tokens']!=targets or not math.isfinite(r['mean_nll']):
            raise ValueError('Final native raw target coverage differs')
    rows={r['model']:r for r in validation['results']}
    if set(rows)!={'1b','3b'} or any(not r['checkpoint_comparison']['exact'] or r['checkpoint_comparison']['max_abs']!=0 or r['trajectory']['loss_max_abs']!=0 for r in rows.values()):
        raise ValueError('Actual1B/3B model/optimizer resume proof missing')
    concerns=[]
    if rehearsal['status']!='passed_actual_rehearsal_training' or rehearsal['curves']['loss_decreased'] is not True:
        concerns.append('Long rehearsal training loss did not meet its recorded decrease check')
    if any(v.get('perplexity_overflow') for v in native['languages'].values()):
        concerns.append('Final native perplexity overflow requires quality investigation')
    return concerns


def gate_paths(a):
    return [(a.rehearsal_run,{'passed_actual_rehearsal_training','needs_review'}),
            (a.final_native_run,{'passed_actual_trained_native_evaluation'}),
            (a.monitored_run,{'passed_actual_monitored_3b'})]


def pending_gates(a):
    missing=[]
    for name,statuses in gate_paths(a):
        root=resolve(name)
        if (root/'.done').exists():
            status=json.loads((root/'report.json').read_text())['status']
            if status not in statuses:raise ValueError('Required actual stage failed: '+name)
            certified(name,status)
        else:
            if (root/'state.json').exists() and json.loads((root/'state.json').read_text()).get('phase') in ('error','failed','paused'):
                raise RuntimeError('Actual prerequisite requires recovery: '+name)
            missing.append(name)
    return missing


def plot_rehearsal(source, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from scripts.observe_rehearsal import summarize,read_rows
    rows=read_rows(source/'training/log.jsonl');s=summarize(rows)
    train={r['step']:r for r in rows if 'loss' in r};evaluation=s['evaluations']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    steps=sorted(train);axes[0,0].plot(steps,[train[i]['loss'] for i in steps]);axes[0,0].set_ylabel('Training loss')
    for category in ['ko','en','math','code']:
        ev=sorted(evaluation);axes[0,1].plot(ev,[evaluation[i]['by_category'][category]['holdout_loss'] for i in ev],label=category)
    axes[0,1].legend();axes[0,1].set_ylabel('Fixed-prefix bf16 probe loss')
    axes[1,0].plot(steps,[train[i]['lr'] for i in steps]);axes[1,0].set_ylabel('Learning rate')
    stable=[i for i in steps if i>=20 and i%250!=10]
    axes[1,1].plot(stable,[train[i]['tok_s']/1000 for i in stable]);axes[1,1].set_ylabel('Steady interval k tokens/s')
    for ax in axes.flat:ax.set_xlabel('Optimizer step');ax.grid(alpha=.2)
    fig.suptitle('Actual1B rehearsal: fixed probes are separate from native full-target PPL')
    fig.savefig(output,dpi=160);plt.close(fig)


def assemble(a, root):
    from scripts.evaluate_final_rehearsal import completed_rehearsal,verify_final_result
    from scripts.run_rehearsal import trace_review
    from scripts.plan_main_training import make_plan
    from train.probe import verify_probe
    from train.rehearsal import aggregate_ledgers
    r=resolve(a.rehearsal_run)
    rehearsal,checkpoint,proof,meta=completed_rehearsal(r)
    native,nc=certified(a.final_native_run,'passed_actual_trained_native_evaluation')
    monitored,mc=certified(a.monitored_run,'passed_actual_monitored_3b')
    f4,c4=certified('runs/f4_full_v2','passed')
    f5,c5=certified('runs/f5_tokenizer_v1','passed')
    f6,c6=certified(a.token_run,'passed')
    validation,vc=certified(a.validation_run,'passed_bounded_actual_training')
    concerns=final_checks(rehearsal,native,monitored,validation)
    if native['rehearsal_report_sha256']!=sha256_file(r/'report.json'):
        raise ValueError('Final trained native report is bound to another rehearsal')
    nr=resolve(a.final_native_run); result=json.loads((nr/'1b_final.json').read_text())
    if native['result_sha256']!=sha256_file(nr/'1b_final.json'):
        raise ValueError('Native detail report content differs')
    marker=json.loads((nr/'1b_final.json.done').read_text())
    if marker!=dict(contract_id=result['contract_id'],sha256=sha256_file(nr/'1b_final.json')):
        raise ValueError('Native detail completion proof differs')
    reference=json.loads(resolve(nc['reference_report']).read_text())
    verify_final_result(result,proof,nc['tokenizer_sha256'],nc['expected_targets'],reference)
    config=json.loads((r/'main.json').read_text()); probe=verify_probe(r/'probe')
    curves=trace_review(r/'training/log.jsonl',config,probe)
    mix=json.loads(resolve(config['mix']).read_text())
    counts=aggregate_ledgers(list((r/'ledgers').glob('*.json')),world=8,steps=5000,seq_len=4096,global_tokens=2097152,sources=mix['sources'])
    if curves!=rehearsal['curves'] or counts!=rehearsal['counts']:
        raise ValueError('Independent final trace/source reconstruction differs')
    for path,digest in monitored['evidence'].items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Actual monitored3B evidence changed')
    plan=make_plan(a)
    plan['remaining']=['Final review of actual generated samples and source/category curves',
                       'Freeze supervised main execution bundle and recovery command',
                       'Choose a feasible local retention policy; archive-based alternatives need separate setup/verification/approval',
                       'Separate user confirmation before main training or checkpoint pruning; no start by this assembler']
    plan['actual_completed'].append('Actual3B source-aware2step-save/resume4 monitoring integration')
    atomic_json(root/'main_proposal.json',plan)
    plot_rehearsal(r,root/'rehearsal_curves.png')
    reports=['runs/f4_full_v2','runs/f5_tokenizer_v1',a.token_run,a.batch_run,a.validation_run,a.native_run,
             a.rehearsal_run,a.final_native_run,a.monitored_run]
    evidence={artifact_path(resolve(run)/name):sha256_file(resolve(run)/name) for run in reports for name in ['report.json','contract.json','.done']}
    evidence.update({artifact_path(root/'main_proposal.json'):sha256_file(root/'main_proposal.json'),
                     artifact_path(root/'rehearsal_curves.png'):sha256_file(root/'rehearsal_curves.png')})
    report=dict(status='assembled_for_final_review',goal_complete=False,main_training_authorized=False,
                requirements=dict(full_split=dict(status='actual_passed',report='runs/f4_full_v2/report.json'),
                    final_tokenizer=dict(status='actual_passed',tokenizer=f5['tokenizer'],sha256=f5['tokenizer_sha256']),
                    full_tokens_mix=dict(status='actual_passed',train_tokens=sum(v['tokens'] for v in f6['source_counts'].values()),
                                         eval_tokens=sum(v['holdout_tokens'] for v in f6['source_counts'].values())),
                    actual_model_validation=dict(status='actual_passed',core='1B/3B finite/eval/model+optimizer/RNG exact',monitored='3B2save/resume4 source/generation/checkpoint integration'),
                    long_rehearsal=dict(status='actual_completed_review_pending',counts=counts,curves=curves,
                                        native_languages=native['languages'],fixed_generations=result['generations']),
                    main_readiness=dict(status='final_review_pending',proposal='main_proposal.json')),
                measured_training=dict(rehearsal_seconds=rehearsal['elapsed_seconds'],peak_host_rss_bytes=rehearsal['peak_host_rss_bytes'],
                                      bounded_monitored3b_seconds=monitored['elapsed_seconds'],
                                      note='Rehearsal wall includes startup/save/eval. Bounded3B wall includes prerequisite wait; not steady compute throughput.'),
                concerns=concerns,remaining_review=plan['remaining'],evidence=evidence,
                scope='Closed actual artifacts and independently reconstructed source/trajectory data assembled. This is not goal completion or a main-training approval.')
    atomic_json(root/'report.json',report)
    atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
    return report


def main(a):
    import fcntl
    root=resolve(a.run_dir);root.relative_to(resolve('runs'));root.mkdir(parents=True,exist_ok=True)
    if a.worker:
        contract=json.loads((root/'contract.json').read_text())
        for path,digest in contract['code'].items():
            if sha256_file(resolve(path))!=digest:raise ValueError('Assembly worker dependency changed')
        if pending_gates(a):raise RuntimeError('Actual completion gates missing')
        return assemble(a,root)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract=dict(sources=dict(rehearsal=a.rehearsal_run,native=a.final_native_run,monitored=a.monitored_run),
                      code={p:sha256_file(resolve(p)) for p in ['scripts/assemble_pretrain_review.py','scripts/plan_main_training.py',
                          'scripts/evaluate_final_rehearsal.py','scripts/run_rehearsal.py','train/probe.py','train/rehearsal.py','scripts/observe_rehearsal.py',
                          'scripts/run_local_split.py','scripts/resource_admission.py','scripts/hardware_profile.py']},
                      scope='Assembly only;final review still required;no training/deletion/upload',max_seconds=172800)
        cp=root/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Assembly contract changed;use a new path')
        atomic_json(cp,contract);(root/'pid').write_text(str(os.getpid())+'\n')
        if (root/'.done').exists():return certified(root,'assembled_for_final_review')
        start=time.monotonic();proc=None
        try:
            while True:
                missing=pending_gates(a)
                if not missing:break
                atomic_json(root/'state.json',dict(phase='running',case='await_actual_completion',pending=missing))
                if not a.wait:raise RuntimeError('Actual rehearsal/native/monitored evidence incomplete')
                if time.monotonic()-start>contract['max_seconds']:raise RuntimeError('Assembly wait budget')
                time.sleep(30)
            for path,digest in contract['code'].items():
                if sha256_file(resolve(path))!=digest:raise ValueError('Assembler dependency changed')
            atomic_json(root/'state.json',dict(phase='running',case='assemble_actual_review'))
            from scripts.resource_admission import inspect
            from scripts.run_local_split import process_tree_rss,size_tree,stop
            hw=inspect();atomic_json(root/'assembly_hardware.json',hw)
            if hw['memory']['cache_aware_admission']['headroom_gib']<8:raise RuntimeError('Assembly headroom unavailable')
            with (root/'assembly.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            begun=time.monotonic();peak=0
            while proc.poll() is None:
                peak=max(peak,process_tree_rss(proc.pid))
                atomic_json(root/'state.json',dict(phase='running',case='assemble_actual_review',worker_pid=proc.pid,peak_rss_bytes=peak))
                if peak>4*2**30 or time.monotonic()-begun>600 or size_tree(root)>2**30 or shutil.disk_usage(root).free<200*2**30:
                    raise RuntimeError('Assembly time/RSS/disk budget')
                time.sleep(2)
            if proc.returncode:raise RuntimeError('Assembly worker failed; preserve output and inspect log')
            certified(root,'assembled_for_final_review');atomic_json(root/'state.json',dict(phase='complete',peak_rss_bytes=peak))
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                from scripts.run_local_split import stop
                stop(proc)
            atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--wait',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--rehearsal-run',default='runs/1b_rehearsal_final_v3');p.add_argument('--final-native-run',default='runs/rehearsal_final_native_v1')
    p.add_argument('--monitored-run',default='runs/monitored_3b_validation_v1');p.add_argument('--token-run',default='runs/f6_tokens_v1')
    p.add_argument('--batch-run',default='runs/actual_deterministic_batch_v1');p.add_argument('--validation-run',default='runs/actual_train_validation_v1')
    p.add_argument('--native-run',default='runs/native_eval_validation_v1');main(p.parse_args())
