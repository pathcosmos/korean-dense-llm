"""Validate an actual3B staged copy; preserve both source and reusable stage."""
import json
import resource
from pathlib import Path
import time
import torch
from common.data_contracts import resolve, artifact_path, sha256_file
from scripts.tiered_checkpoint_retention import TieredRetention, copy_payload, atomic_json
from scripts.validate_tiered_retention_copy import cpu_state


def run():
    from scripts.resource_admission import inspect
    root=resolve('runs/tiered_retention_copy_v2');root.mkdir(exist_ok=False)
    hw=inspect();atomic_json(root/'hardware.json',hw)
    if hw['memory']['cache_aware_admission']['headroom_gib']<512:
        raise RuntimeError('Training plus128GiB copy/CPU validation headroom unavailable')
    main=resolve('runs/3b_full_v1');policy=json.loads((main/'contract.json').read_text())['tiered_retention']
    manager=TieredRetention(main,policy,authorized=True)
    source=main/'training/ckpt_0000500';cert=main/'checkpoint_certificates/ckpt_0000500.json'
    proof=json.loads(cert.read_text());stage=manager.local/'staging/ckpt_0000500';started=time.monotonic()
    atomic_json(root/'code.json',{p:sha256_file(resolve(p)) for p in
                ['scripts/validate_retention3_copy.py','scripts/tiered_checkpoint_retention.py']})
    # Existing manager is between saves. The exact shared lock prevents any
    # concurrent mutation, while the trainer itself continues unchanged.
    with manager.locked():
        manager.budgets(incoming_bytes=proof['bytes'])
        metrics=copy_payload(source,stage,proof,copy_workers=8,hash_workers=8)
        torch.set_num_threads(8);left=cpu_state(source);right=cpu_state(stage)
        if left.keys()!=right.keys():raise ValueError('State key set differs')
        tensors=0
        for name,a in left.items():
            b=right[name]
            if isinstance(a,torch.Tensor):
                if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):
                    raise ValueError('Restored tensor differs: '+name)
                tensors+=1
            elif a!=b:raise ValueError('Optimizer metadata differs: '+name)
        for rank in range(8):
            a=torch.load(source/f'rng_rank{rank}.pt',weights_only=True,map_location='cpu')
            b=torch.load(stage/f'rng_rank{rank}.pt',weights_only=True,map_location='cpu')
            if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('RNG differs')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if rss>128*2**30 or time.monotonic()-started>600:raise RuntimeError('Copy/restore budget exceeded')
        report=dict(status='passed_actual_checkpoint_copy_and_cpu_state_restore',cpu_state_exact=True,
                    all8_original_cpu_cuda_rng_exact=True,cpu_restored_tensors=tensors,selected_copy_workers=8,
                    actual3b_copy=metrics,actual3b_output=str(stage),elapsed_seconds=time.monotonic()-started,
                    actual_peak_rss_bytes=rss,source_deletions=0,stage_is_reusable_not_published=True,
                    evidence={artifact_path(cert):sha256_file(cert)},actual_gpu_restore=False)
        atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json')))
        print(json.dumps(report),flush=True)


if __name__=='__main__':run()
