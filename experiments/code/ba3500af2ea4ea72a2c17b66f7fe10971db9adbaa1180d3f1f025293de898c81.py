"""Prepare verified raw holdout during F6/F7, then evaluate actual validated checkpoints.

One fresh live split context covers preparation and both checkpoint evaluations.
The GPU lock is acquired only after the actual training validation certificate exists.
No model training, data pruning, conversion, or external upload is performed here.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,artifact_path,atomic_json,sha256_file


def certified(run,expected_status):
    run=resolve(run);report=json.loads((run/'report.json').read_text());contract=json.loads((run/'contract.json').read_text())
    if json.loads((run/'.done').read_text())!=dict(report_sha256=sha256_file(run/'report.json'),contract_sha256=sha256_file(run/'contract.json')):
        raise ValueError('Completion content proof differs')
    if report['status']!=expected_status:raise ValueError('Required actual stage did not pass')
    for file,digest in contract['code'].items():
        if sha256_file(resolve(file))!=digest:raise ValueError('Certified stage implementation changed:'+file)
    return report,contract


def checkpoint_inputs(run,tokenizer_sha256):
    report,contract=certified(run,'passed_bounded_actual_training')
    if contract['world']!=8 or contract['sequence']!=4096 or contract['steps']!=4 or contract['compile']:
        raise ValueError('Actual model validation geometry differs')
    checkpoints={}
    for r in report['results']:
        model=r['model'];published=r['published_checkpoint'];path=resolve(published['path'])
        if model not in ('1b','3b') or model in checkpoints:raise ValueError('Actual checkpoint model coverage differs')
        if path!=resolve(run)/(model+'_validated_checkpoint'):raise ValueError('Unexpected published checkpoint path')
        publication=path/'publication.json'
        if sha256_file(publication)!=published['publication_sha256']:raise ValueError('Checkpoint publication proof differs')
        meta=json.loads((path/'meta.json').read_text())
        if not (path/'.complete').is_file() or meta['step']!=4 or meta['config']['model']!=model:
            raise ValueError('Actual checkpoint metadata differs')
        if meta['training_contract']['mix']['tokenizer_sha256']!=tokenizer_sha256:raise ValueError('Actual checkpoint tokenizer differs')
        if any(meta['config'][k]!=v for k,v in contract['batch_geometry'][model].items()):raise ValueError('Checkpoint selected batch/LR differs')
        checkpoints[model]=path
    if set(checkpoints)!={'1b','3b'}:raise ValueError('Both actual model checkpoints are required')
    return checkpoints


def verify_evaluation(report,checkpoint,tokenizer_sha256):
    publication=json.loads((checkpoint/'publication.json').read_text())
    expected={r['path']:r['sha256'] for r in publication['files']}
    # run() already freshly hashed the checkpoint. Reuse that content proof rather
    # than reading tens of GB a second time solely to check the publication copy.
    actual=report['contract']['checkpoint']
    if actual!={**expected,'publication.json':sha256_file(checkpoint/'publication.json')}:
        raise ValueError('Evaluated checkpoint differs from certified publication')
    if (report['status']!='passed' or report['contract']['tokenizer_sha256']!=tokenizer_sha256 or
            report['contract']['input_provenance']['level']!='verified-eval-split' or report['checkpoint_step']!=4):
        raise ValueError('Native evaluation lacks actual verified provenance')
    if set(report['languages'])!={'ko','en'} or any(v['valid_tokens']<1 for v in report['languages'].values()):
        raise ValueError('Native evaluation requires both languages and valid targets')
    prompts=report['generations']
    if len(prompts)!=6 or len({r['id'] for r in prompts})!=6 or {r['category'] for r in prompts}!={'ko','en'}:
        raise ValueError('Native fixed generation coverage differs')


def worker(root):
    root=resolve(root);contract=json.loads((root/'contract.json').read_text());plan=contract['evaluation']
    def progress(phase,**kw):atomic_json(root/'worker_state.json',dict(phase=phase,unix=time.time(),pid=os.getpid(),**kw))
    from scripts.run_final_tokens import verify_tokenizer_run
    trained,_=verify_tokenizer_run(resolve(contract['tokenizer_run']),contract['sources_sha256'],contract['split_sha256'])
    if trained['tokenizer_sha256']!=contract['tokenizer_sha256']:raise ValueError('Final tokenizer proof changed')
    from eval.prepare_holdout import local_verified_inputs
    from data.split_inputs import prepare
    from eval.run import run
    progress('verify_local_eval_context')
    with local_verified_inputs(plan['source_manifest'],plan['canonical_split_manifest'],plan['reader_manifest'],hash_workers=16) as inputs:
        progress('prepare_raw_eval')
        prepared=prepare(plan['source_manifest'],plan['canonical_split_manifest'],root/'holdout',plan['byte_budgets'],
                         mode='eval',seed=plan['seed'],categories=plan['categories'],verified_inputs=inputs)
        atomic_json(root/'prepared_inputs.json',dict(stats=prepared['stats'],manifest_sha256=sha256_file(root/'holdout/manifest.json')))
        validation=resolve(contract['validation_run'])
        while not (validation/'.done').exists():
            progress('await_actual_training_validation');time.sleep(30)
        checkpoints=checkpoint_inputs(validation,trained['tokenizer_sha256'])
        lock=resolve('runs/.gpu_benchmark.lock').open('a')
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:progress('await_gpu_lock');time.sleep(30);continue
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not occupied:break
            fcntl.flock(lock,fcntl.LOCK_UN);progress('await_idle_gpu');time.sleep(30)
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        results={}
        for model,checkpoint in checkpoints.items():
            progress('native_evaluation',model=model);start=time.monotonic()
            result=run(checkpoint,resolve(trained['tokenizer']),root/'holdout/samples.jsonl',resolve(plan['prompts']),root/(model+'.json'),
                       seq_len=plan['context'],stride=plan['stride'],max_new_tokens=plan['max_new_tokens'],seed=plan['seed'],temperature=0,
                       device='cuda',holdout_manifest=root/'holdout/manifest.json',verified_inputs=inputs)
            verify_evaluation(result,checkpoint,trained['tokenizer_sha256'])
            results[model]=dict(report=artifact_path(root/(model+'.json')),report_sha256=sha256_file(root/(model+'.json')),
                                elapsed_seconds=time.monotonic()-start,languages=result['languages'])
        checkpoint_inputs(validation,trained['tokenizer_sha256'])
    for f,h in contract['code'].items():
        if sha256_file(resolve(f))!=h:raise ValueError('Native evaluation implementation changed')
    report=dict(status='passed_actual_native_evaluation',results=results,tokenizer_sha256=trained['tokenizer_sha256'],
                validation_report_sha256=sha256_file(validation/'report.json'),
                scope='Actual four-step1B/3B native raw-document ko/en full-target PPL and six base prompts; not official benchmark or trained-model quality proof.')
    atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
    progress('complete')


def main(a):
    if a.worker:return worker(a.run_dir)
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    root=resolve(a.run_dir);root.resolve().relative_to(resolve('runs').resolve())
    if (root/'.done').exists():return certified(root,'passed_actual_native_evaluation')
    hw=inspect()
    if len(hw['gpus'])!=8 or hw['memory']['conservative_headroom_gib']<384:raise RuntimeError('Native eval CPU preparation headroom unavailable')
    plan=json.loads(resolve(a.plan).read_text())['native_evaluation']
    if plan['context']!=4096 or plan['stride']!=2048 or plan['max_new_tokens']!=64 or sha256_file(resolve(plan['prompts']))!=plan['prompts_sha256']:
        raise ValueError('Actual native evaluation policy differs')
    root.mkdir(parents=True,exist_ok=False);scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(parents=True,exist_ok=False)
    deps=['scripts/run_native_evaluation.py','scripts/run_final_tokens.py','scripts/validate_actual_training.py','eval/run.py','eval/prepare_holdout.py',
          'eval/perplexity.py','eval/generation.py','model/model.py','data/split_inputs.py','data/verified_split_context.py','data/local_split_reader.py',
          'data/split_manifest.py','common/source_inventory.py','common/data_contracts.py']
    contract=dict(version=1,evaluation=plan,tokenizer_run=a.tokenizer_run,validation_run=a.validation_run,
                  tokenizer_sha256=json.loads((resolve(a.tokenizer_run)/'report.json').read_text())['tokenizer_sha256'],
                  sources_sha256=sha256_file(resolve(plan['source_manifest'])),split_sha256=sha256_file(resolve(plan['canonical_split_manifest'])),
                  code={f:sha256_file(resolve(f)) for f in deps},hardware=hw,host_rss_gib=224,local_budget_gib=8,durable_budget_gib=8,reserve_gib=200,max_seconds=21600)
    atomic_json(root/'contract.json',contract);(root/'pid').write_text(str(os.getpid()))
    cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity'];env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',TMPDIR=str(scratch),PYTHONUNBUFFERED='1')
    start=time.monotonic();peak=0;proc=None
    try:
        cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),'--run-dir',str(root),'--worker']
        with (root/'console.log').open('w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            peak=max(peak,process_tree_rss(proc.pid));elapsed=time.monotonic()-start
            w=json.loads((root/'worker_state.json').read_text()) if (root/'worker_state.json').exists() else {}
            atomic_json(root/'state.json',dict(phase='running',case=w.get('phase','starting'),worker_pid=proc.pid,elapsed_seconds=elapsed,peak_rss_bytes=peak))
            if peak>224*2**30 or elapsed>21600:raise RuntimeError('Native evaluation time/RSS budget')
            for path in [root,scratch]:
                if size_tree(path)>8*2**30 or shutil.disk_usage(path).free<200*2**30:raise RuntimeError('Native evaluation disk budget')
            time.sleep(2)
        if proc.returncode:raise RuntimeError('Native evaluation worker failed; preserve output and inspect console')
        certified(root,'passed_actual_native_evaluation');atomic_json(root/'state.json',dict(phase='complete'))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:stop(proc)
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--plan',default='runs/f7_preflight_v1/plan.json');p.add_argument('--tokenizer-run',default='runs/f5_tokenizer_v1');p.add_argument('--validation-run',default='runs/actual_train_validation_v1')
    main(p.parse_args())
