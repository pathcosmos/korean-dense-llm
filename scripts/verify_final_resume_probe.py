"""Content certification and independent comparison for the final resume probe.

No source mutation, pruning, GPU launch or authorization. Tests may exercise
small CPU states; only an actual8rank3B run can satisfy final acceptance.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, atomic_json, artifact_path, identity, sha256_file
from train.final_resume_probe import file_stats, stat_record, metadata_for, MODES, OUTPUTS, diagnostic_lr


def verify_inventory(path, proof, *, hashes=False):
    path = Path(path)
    rows = proof['files']
    expected = {r['path']: r for r in rows}
    current = file_stats(path)
    if (len(expected) != len(rows) or set(current) != set(expected) or
            sum(r['bytes'] for r in rows) != proof['bytes'] or proof['stats'] != current):
        raise ValueError('Diagnostic inventory/stat snapshot differs')
    if json.loads((path/'meta.json').read_text()) != proof['metadata']:
        raise ValueError('Diagnostic metadata differs')
    for name, row in expected.items():
        if current[name]['st_size'] != row['bytes']:
            raise ValueError('Diagnostic file size differs')
        if hashes and sha256_file(path/name) != row['sha256']:
            raise ValueError('Diagnostic SHA differs')
    if file_stats(path) != current:
        raise ValueError('Diagnostic changed while verifying')


def seal(path, expected_meta, output):
    import torch
    import torch.distributed.checkpoint as dcp
    from scripts.supervise_main_training import validate_optimizer_map
    path, output = Path(path), Path(output)
    stats = file_stats(path)
    world = expected_meta['world_size']
    base = {'.complete', '.metadata', 'meta.json'} | {f'rng_rank{r}.pt' for r in range(world)}
    payloads = {p for p in stats if p.startswith('__') and p.endswith('.distcp') and '/' not in p}
    if (set(stats) != base | payloads or len(payloads) != world or
            json.loads((path/'meta.json').read_text()) != expected_meta):
        raise ValueError('Diagnostic checkpoint completeness/identity differs')
    meta = dcp.FileSystemReader(path).read_metadata()
    validate_optimizer_map(meta)
    if not meta.storage_data:
        raise ValueError('Diagnostic DCP storage map missing')
    for value in meta.storage_data.values():
        if value.relative_path not in payloads or value.offset < 0 or value.length <= 0 or stats[value.relative_path]['st_size'] < value.offset+value.length:
            raise ValueError('Diagnostic DCP storage map invalid')
    for rank in range(world):
        rng = torch.load(path/f'rng_rank{rank}.pt', weights_only=True, map_location='cpu')
        required = {'cpu', 'cuda'} if expected_meta.get('kind') == 'final_resume_diagnostic' else {'cpu'}
        if set(rng) != required or any(not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1 or v.numel() < 1 for v in rng.values()):
            raise ValueError('Diagnostic rank RNG missing/invalid')
    def record(name):
        target = path/name
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        digest = sha256_file(target)
        if stat_record(target) != stats[name]:
            raise ValueError('Diagnostic payload changed while certifying')
        return dict(path=name, bytes=stats[name]['st_size'], sha256=digest)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(record, sorted(stats)))
    from scripts.tiered_checkpoint_retention import fsync_dir
    fsync_dir(path)
    proof = dict(status='certified_diagnostic_checkpoint', path=artifact_path(path),
                 metadata=expected_meta, files=rows, bytes=sum(r['bytes'] for r in rows), stats=stats)
    verify_inventory(path, proof)
    if output.exists():
        if json.loads(output.read_text()) != proof:
            raise ValueError('Existing diagnostic certificate differs')
    else:
        atomic_json(output, proof)
    return proof


def compare_states(left_path, right_path, *, world):
    import torch
    from scripts.validate_tiered_retention_copy import cpu_state
    started = time.monotonic()
    left, right = cpu_state(left_path), cpu_state(right_path)
    if left.keys() != right.keys():
        raise ValueError('Restored model/optimizer key coverage differs')
    tensors = model = optimizer = 0
    for name, a in left.items():
        b = right[name]
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                raise ValueError('Restored state tensor differs: '+name)
            tensors += 1
            model += name.startswith('model.')
            optimizer += name.startswith('optim.')
        elif a != b:
            raise ValueError('Optimizer non-tensor state differs: '+name)
    for rank in range(world):
        a = torch.load(Path(left_path)/f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
        b = torch.load(Path(right_path)/f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError('Restored rank RNG differs')
    return dict(exact=True, tensors=tensors, model_tensors=model, optimizer_tensors=optimizer,
                rng_ranks=world, elapsed_seconds=time.monotonic()-started)


def ledger_review(rows, *, parent_step, world, config, source_order, plan_id):
    indexed = {}
    for row in rows:
        mode, rank = row['mode'], row['rank']
        if mode not in MODES or type(rank) is not int or not 0 <= rank < world or (mode, rank) in indexed:
            raise ValueError('Diagnostic mode/rank duplicate or invalid')
        if (row['success'] is not True or row['plan_id'] != plan_id or
                row['parent_step'] != parent_step or row['diagnostic_only'] is not True):
            raise ValueError('Diagnostic rank completion identity differs')
        begin, end = MODES[mode]
        expected = list(range(parent_step+begin+1, parent_step+end+1))
        if [s['step'] for s in row['steps']] != expected or [s['step'] for s in row['metrics']] != expected:
            raise ValueError('Diagnostic update coverage differs')
        for step, metric in zip(row['steps'], row['metrics']):
            counts = step['source_sequences']
            if (not counts or set(counts)-set(source_order) or
                    any(type(n) is not int or n < 1 for n in counts.values()) or
                    sum(counts.values())*config['seq_len']*world != config['global_tokens']):
                raise ValueError('Diagnostic committed source geometry differs')
            if (any(not math.isfinite(metric[k]) for k in ['loss', 'gnorm', 'lr']) or
                    metric['loss'] < 0 or metric['gnorm'] < 0 or metric['lr'] != diagnostic_lr(config)):
                raise ValueError('Diagnostic finite metric/LR differs')
        indexed[mode, rank] = row
    if set(indexed) != {(m, r) for m in MODES for r in range(world)}:
        raise ValueError('Incomplete diagnostic rank coverage')
    totals = {s: 0 for s in source_order}
    for rank in range(world):
        continuous = indexed['continuous', rank]
        split = [indexed['first', rank], indexed['resumed', rank]]
        if continuous['steps'] != [s for row in split for s in row['steps']]:
            raise ValueError('Restored stateless sampler choices differ')
        comparable = lambda metric: {k: metric[k] for k in ['step', 'loss', 'gnorm', 'lr']}
        if [comparable(m) for m in continuous['metrics']] != [comparable(m) for row in split for m in row['metrics']]:
            raise ValueError('Continuous/resumed trajectory differs')
        for step in continuous['steps']:
            for source, n in step['source_sequences'].items():
                totals[source] += n*config['seq_len']
    for offset in range(2):
        if len({indexed['continuous', r]['metrics'][offset]['loss'] for r in range(world)}) != 1:
            raise ValueError('Reduced loss differs across ranks')
    return dict(source_tokens=totals, unique_diagnostic_tokens=2*config['global_tokens'],
                actual_diagnostic_committed_tokens=4*config['global_tokens'],
                main_committed_tokens_added=0, compared_rank_pairs=world*2,
                scope='Two diagnostic updates repeated across continuous/split trajectories, outside main token target.')


def prepare(main, root, *, native_run=None):
    main, root = resolve(main), resolve(root)
    if root.exists() or root.is_relative_to(main) or main.is_relative_to(root) or not root.is_relative_to(resolve('runs')):
        raise ValueError('A new independent diagnostic run is required')
    config_path = main/'main.json'
    config = json.loads(config_path.read_text())
    mc = json.loads((main/'contract.json').read_text())
    parent_path = main/'training/training_contract.json'
    parent = json.loads(parent_path.read_text())
    if (not mc.get('main_training_authorized') or not mc.get('new_checkpoint_pruning_authorized') or
            mc['configurations']['main.json'] != sha256_file(config_path) or config['model'] != '3b' or config['max_steps'] != 57000 or
            config['compile'] or parent['world_size'] != 8 or parent['id'] != identity({k:v for k,v in parent.items() if k != 'id'})):
        raise ValueError('Reviewed parent3B8rank contract missing')
    for path, digest in {**mc['code'], **mc['evidence']}.items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Parent code/evidence changed: '+path)
    if native_run is None:
        raise ValueError('The concurrently reserved native evaluation run is required')
    native = resolve(native_run)
    native_contract = json.loads((native/'contract.json').read_text())
    if (native_contract['purpose'] != 'final_native_main_evaluation' or
            resolve(native_contract['main_run']) != main or native == root or
            (native_contract['host_rss_gib'], native_contract['durable_budget_gib'],
             native_contract['local_budget_gib']) != (256, 8, 8)):
        raise ValueError('Concurrent native run/target/resource reservation differs')
    scratch = Path(mc['tiered_retention']['temp_root'])/root.name
    if scratch.exists():
        raise ValueError('Preserve existing diagnostic scratch')
    if any(p.is_symlink() for path in [root, scratch] for p in [path, *path.parents]):
        raise ValueError('Diagnostic preparation path symlink rejected')
    files = [main/'contract.json', config_path, main/'authorization.json', parent_path,
             main/'expected_tokens.json', native/'contract.json']
    code = dict(mc['code'])
    for name in ['train/final_resume_probe.py', 'scripts/verify_final_resume_probe.py',
                 'scripts/audit_main_training.py', 'scripts/validate_tiered_retention_copy.py']:
        code[name] = sha256_file(resolve(name))
    plan = dict(version=2, purpose='isolated_final_resume_probe', main_run=artifact_path(main), run_dir=artifact_path(root),
                native_run=artifact_path(native),
                scratch=str(scratch), parent_contract=artifact_path(parent_path), parent_config=artifact_path(config_path),
                parent_training_contract_id=parent['id'], parent_step=57000, world_size=8, updates=2,
                main_target_modified=False, lr_policy='freeze_last_parent_update', diagnostic_lr=diagnostic_lr(config),
                code=code, inputs={artifact_path(p): sha256_file(p) for p in files}, framework=mc['framework'],
                max_seconds=3600, max_pss_gib=256, cpu_affinity=list(range(64,96)),
                local_budget_gib=192, durable_budget_gib=1, reserved_main_temp_gib=1024,
                estimated_checkpoint_count=4, expected_checkpoint_bytes=mc['expected_checkpoint_bytes'],
                required_gpu_gate='Audited actual57000 + full parent SHA + global idle8GPU lock',
                scope='Full-state roundtrip + continuous2 versus split1/resume1 in separate local scratch. No pruning or main mutation.')
    root.mkdir(parents=True)
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'prepared.json', dict(status='prepared_not_executed', plan_sha256=sha256_file(root/'plan.json'),
                plan_id=identity(plan), actual_final_resume_verified=False))
    return plan


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--native-run', required=True)
    args = parser.parse_args()
    prepare(args.source_run, args.run_dir, native_run=args.native_run)
