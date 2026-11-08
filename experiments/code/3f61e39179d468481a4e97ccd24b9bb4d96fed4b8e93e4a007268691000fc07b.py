"""Bounded read-only-source checkpoint copy/CPU restore experiment, no pruning.

Two immutable1B copies compare4/8 workers. A full actual3B checkpoint is then
copied and deserialized on CPU. Originals and experimental copies are retained.
No GPU training, external upload, or production retirement is performed.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,artifact_path,sha256_file
from scripts.tiered_checkpoint_retention import copy_payload,inventory,atomic_json,fsync_dir,GIB


def cpu_state(path):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    metadata=dcp.FileSystemReader(path).read_metadata()
    state={k:(torch.empty(v.size,dtype=v.properties.dtype) if isinstance(v,TensorStorageMetadata) else None)
           for k,v in metadata.state_dict_metadata.items()}
    dcp.load(state,checkpoint_id=path)
    return state


def run(root):
    import torch
    import torch.distributed.checkpoint as dcp
    from scripts.resource_admission import inspect
    from scripts.tiered_checkpoint_retention import size_tree
    import shutil
    hw=inspect();atomic_json(root/'hardware.json',hw)
    scratch=Path(hw['storage']['scratch_dir'])/'tiered_retention_copy_validation_v1'
    scratch.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'inputs.json',dict(local_output=str(scratch),source_write_allowed=False,local_budget_gib=96,
                max_cpu_pss_gib=128,copy_workers_candidates=[4,8],hash_workers=8,cpu_threads=8))
    if (hw['memory']['cache_aware_admission']['headroom_gib']<160 or
            shutil.disk_usage(scratch).free<(96+1024+200)*GIB):
        raise RuntimeError('Copy/CPU restore budget unavailable')
    start=time.monotonic()
    def state(case,**kw):
        if time.monotonic()-start>1200 or size_tree(scratch)>96*GIB:
            raise RuntimeError('Bounded checkpoint copy budget exceeded')
        atomic_json(root/'state.json',dict(phase='running',case=case,elapsed_seconds=time.monotonic()-start,**kw))
    source1=resolve('runs/1b_rehearsal_final_v3/training/ckpt_0001000')
    certpath=resolve('runs/1b_rehearsal_final_v3/checkpoint_certificates/ckpt_0001000.json')
    proof1=json.loads(certpath.read_text());inventory(source1,proof1)
    results=[]
    for workers in [4,8]:
        state('copy_worker_comparison',workers=workers)
        stage=scratch/f'1b_workers{workers}.staging';final=scratch/f'1b_workers{workers}'
        metrics=copy_payload(source1,stage,proof1,copy_workers=workers,hash_workers=8)
        stage.rename(final);fsync_dir(scratch)
        results.append(dict(workers=workers,**metrics,output=str(final)))
    # One observation/candidate is not a robust optimality claim. Prefer4 if
    # the difference is below5%, otherwise choose the measured faster candidate.
    selected=min(results,key=lambda x:x['total_seconds'])['workers']
    if abs(results[0]['total_seconds']-results[1]['total_seconds'])/max(r['total_seconds'] for r in results)<.05:
        selected=4
    source=resolve('runs/actual_train_validation_v1/3b_validated_checkpoint')
    pub=source/'publication.json';publication=json.loads(pub.read_text());meta=json.loads((source/'meta.json').read_text())
    report=json.loads(resolve('runs/actual_train_validation_v1/report.json').read_text())
    row=next(r for r in report['results'] if r['model']=='3b')
    if row['published_checkpoint']['publication_sha256']!=sha256_file(pub):
        raise ValueError('Original3B publication differs')
    files=publication['files']+[dict(path='publication.json',bytes=pub.stat().st_size,sha256=sha256_file(pub))]
    proof=dict(step=meta['step'],training_contract_id=meta['training_contract']['id'],files=files,bytes=sum(r['bytes'] for r in files))
    atomic_json(root/'source3b_manifest.json',proof)
    state('actual_full3b_copy',workers=selected)
    stage=scratch/'3b.staging';final=scratch/'3b'
    copy=copy_payload(source,stage,proof,copy_workers=selected,hash_workers=8)
    stage.rename(final);fsync_dir(scratch)
    state('actual_full3b_cpu_deserialization')
    torch.set_num_threads(8)
    restore_start=time.monotonic();left=cpu_state(source);right=cpu_state(final)
    if left.keys()!=right.keys():raise ValueError('Actual3B restored state keys differ')
    tensors=0;elements=0
    for name,a in left.items():
        b=right[name]
        if isinstance(a,torch.Tensor):
            if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
                raise ValueError('Actual3B restored tensor differs: '+name)
            tensors+=1;elements+=a.numel()
        elif a!=b:raise ValueError('Actual3B optimizer non-tensor state differs')
    for rank in range(8):
        a=torch.load(source/f'rng_rank{rank}.pt',map_location='cpu',weights_only=True)
        b=torch.load(final/f'rng_rank{rank}.pt',map_location='cpu',weights_only=True)
        if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
            raise ValueError('Actual rank RNG differs after local copy')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss>128*GIB:raise RuntimeError('CPU deserialization memory budget exceeded')
    result=dict(status='passed_actual_checkpoint_copy_and_cpu_state_restore',worker_comparison=results,
                selected_copy_workers=selected,worker_selection_scope='One actual1B copy per candidate, not a universal optimum',
                actual3b_copy=copy,actual3b_output=str(final),cpu_restored_tensors=tensors,cpu_restored_elements=elements,
                model_tensors=sum(k.startswith('model.') and isinstance(v,torch.Tensor) for k,v in left.items()),
                optimizer_tensors=sum(k.startswith('optim.') and isinstance(v,torch.Tensor) for k,v in left.items()),
                cpu_state_exact=True,all8_original_cpu_cuda_rng_exact=True,cpu_restore_seconds=time.monotonic()-restore_start,
                actual_peak_rss_bytes=rss,local_output_bytes=size_tree(scratch),elapsed_seconds=time.monotonic()-start,
                evidence={artifact_path(certpath):sha256_file(certpath),artifact_path(pub):sha256_file(pub)},
                scope='Full actual3B CPU state equality and all original rank RNG; not a new8GPU optimizer-update/resume validation or25 full-size checkpoint endurance test. Source deletion0, external upload0, main launch0. Experimental copies preserved.')
    atomic_json(root/'report.json',result);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json')))
    atomic_json(root/'state.json',dict(phase='complete'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args();root=resolve(a.run_dir)
    root.mkdir(parents=True,exist_ok=False);(root/'pid').write_text(str(os.getpid())+'\n')
    atomic_json(root/'code.json',{name:sha256_file(resolve(name)) for name in ['scripts/validate_tiered_retention_copy.py','scripts/tiered_checkpoint_retention.py']})
    try:run(root)
    except BaseException as exc:
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise
