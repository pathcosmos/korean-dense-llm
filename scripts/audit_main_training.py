"""Independently audit persisted optimizer ledgers and the contracted probe schedule.

Read-only toward the producer. A passed prefix is not final storage/restore,
native evaluation, useful quality, or Goal completion. Never imports a trainer.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, identity, atomic_json, sha256_file


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer: '+label)
    return value


def finite(value, label, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError('Invalid finite value: '+label)
    return value


def verify_restart_metadata(meta, config, training_contract, through_step):
    # train.train.main resolves only out_dir before run/save_checkpoint. Keep
    # the approved on-disk config/hash intact and compare the exact saved form;
    # no other field or path is normalized or omitted.
    saved_config = dict(config, out_dir=str(resolve(config['out_dir'])))
    if (meta['step'] != through_step or meta['world_size'] != 8 or
            meta['tokens_per_step'] != config['global_tokens'] or
            meta['training_contract'] != training_contract or meta['config'] != saved_config):
        raise ValueError('Checkpoint restart metadata differs')


class Snapshot:
    """Bind bytes read once and reject replacement/mutation before publication."""
    def __init__(self):
        self.bindings = {}
        self.stats = {}

    @staticmethod
    def stamp(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def read(self, path, *, lines=False, require_newline=False):
        path = Path(path)
        if path.is_symlink():
            raise ValueError('Symlink audit input')
        before = self.stamp(path)
        if before[2] > 128*1024**2:
            raise ValueError('Audit metadata exceeds128MiB budget')
        raw = path.read_bytes()
        if before != self.stamp(path):
            raise ValueError('Audit input changed while reading')
        key = artifact_path(path)
        digest = hashlib.sha256(raw).hexdigest()
        if key in self.bindings and self.bindings[key] != digest:
            raise ValueError('Audit input changed between reads')
        self.bindings[key] = digest
        self.stats[path] = before
        if lines:
            if require_newline and raw and not raw.endswith(b'\n'):
                raise ValueError('Incomplete trace line; retry observation, never restart training')
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        return json.loads(raw)

    def unchanged(self):
        if any(self.stamp(p) != s for p, s in self.stats.items()):
            raise ValueError('Audit snapshot changed; retry observation')


def aggregate(records, *, world, through_step, target_steps, seq_len,
              global_tokens, sources, generation_steps, contract_id):
    for name, value in [('world', world), ('through', through_step),
                        ('target', target_steps), ('seq_len', seq_len), ('global', global_tokens)]:
        integer(value, name)
    if through_step > target_steps or global_tokens % (world*seq_len):
        raise ValueError('Invalid audited geometry')
    sources = tuple(sources)
    if not sources or len(set(sources)) != len(sources) or not contract_id:
        raise ValueError('Invalid source/contract identity')
    schedule = list(generation_steps)
    if sorted(set(schedule)) != schedule or any(type(s) is not int or not 1 <= s <= target_steps for s in schedule):
        raise ValueError('Invalid generation schedule')
    rows = {}
    generated = {}
    attempts = set()
    persisted = replayed = tail = initializations = 0
    completed_ranks = set()
    for attempt, record in records:
        rank = integer(record['rank'], 'rank', 0)
        if rank >= world or (attempt, rank) in attempts:
            raise ValueError('Duplicate attempt/rank or invalid rank')
        attempts.add((attempt, rank))
        initializations += integer(record['optimizer_initializations'], 'optimizer_initializations', 0)
        steps = record['steps']
        if not steps and not record['generations']:
            continue  # Failed before the first actual update; not coverage.
        if record['contract_id'] != contract_id or type(record['success']) is not bool:
            raise ValueError('Ledger contract/success differs')
        previous = None
        in_attempt = set()
        for row in steps:
            step = integer(row['step'], 'step')
            if step > target_steps or (previous is not None and step != previous+1):
                raise ValueError('Noncontiguous or duplicated attempt update')
            previous = step
            in_attempt.add(step)
            counts = row['source_sequences']
            if not counts or set(counts)-set(sources):
                raise ValueError('Unknown/missing sampled source')
            for count in counts.values():
                integer(count, 'source_sequences')
            if sum(counts.values())*seq_len*world != global_tokens:
                raise ValueError('Rank token geometry differs')
            tokens = sum(counts.values())*seq_len
            if step > through_step:
                tail += tokens
                continue
            persisted += tokens
            key = (rank, step)
            canonical = tuple(counts.get(source, 0) for source in sources)
            if key in rows:
                if rows[key] != canonical:
                    raise ValueError('Replayed source choices changed')
                replayed += tokens
            else:
                rows[key] = canonical
        seen_generations = set()
        for g in record['generations']:
            step = integer(g['step'], 'generation_step')
            digest = g['sha256']
            if (step not in schedule or step not in in_attempt or step in seen_generations or
                    not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
                raise ValueError('Invalid generation receipt')
            seen_generations.add(step)
            if step <= through_step:
                key = (step, rank)
                if key in generated and generated[key] != digest:
                    raise ValueError('Replayed generation changed')
                generated[key] = digest
        if record['success'] and previous == target_steps:
            completed_ranks.add(rank)
    if len(rows) != world*through_step:
        raise ValueError('Incomplete step/rank coverage')
    source_tokens = {s: 0 for s in sources}
    for counts in rows.values():
        for source, count in zip(sources, counts):
            source_tokens[source] += count*seq_len
    if sum(source_tokens.values()) != global_tokens*through_step or persisted-replayed != sum(source_tokens.values()):
        raise ValueError('Unique committed token total differs')
    generations = {}
    for step in schedule:
        if step > through_step:
            continue
        values = [generated.get((step, rank)) for rank in range(world)]
        if None in values or len(set(values)) != 1:
            raise ValueError('Generation rank coverage/agreement differs')
        generations[str(step)] = values[0]
    return dict(contract_id=contract_id, through_step=through_step,
                committed_step_rank_pairs=len(rows), committed_tokens=through_step*global_tokens,
                source_tokens=source_tokens, source_fractions={s:n/(through_step*global_tokens) for s,n in source_tokens.items()},
                persisted_attempt_tokens_through_prefix=persisted, replayed_tokens_through_prefix=replayed,
                persisted_attempt_tokens_beyond_prefix=tail, optimizer_initializations_excluded=initializations,
                successful_target_ranks=sorted(completed_ranks), generation_sha256_by_step=generations,
                scope='Persisted successful optimizer updates only; work lost before ledger persistence is unknown, not counted as zero.')


def audit_trace(rows, *, config, through_step, contract_id, probe, prompts, generation_hashes):
    train = {}
    evaluated = {}
    generated = {}
    replayed_logs = 0
    groups = {g['source']: g for g in probe['groups']}
    if len(groups) != len(probe['groups']):
        raise ValueError('Duplicate probe source')
    category_sources = {}
    for source, group in groups.items():
        category_sources.setdefault(group['category'], []).append(source)
    prompt_map = {p['id']: p for p in prompts}
    if len(prompt_map) != len(prompts):
        raise ValueError('Duplicate fixed prompt')
    for row in rows:
        step = integer(row['step'], 'trace_step')
        if step > config['max_steps']:
            raise ValueError('Trace outside training target')
        if step > through_step:
            continue
        if 'loss' in row:
            for key in ['loss', 'gnorm', 'lr', 'tok_s', 'tokens_b']:
                finite(row[key], key)
            if (step % config['log_every'] or row['tok_s'] <= 0 or
                    not math.isclose(row['tokens_b']*1e9, step*config['global_tokens'], abs_tol=1, rel_tol=0)):
                raise ValueError('Train trace geometry differs')
            exact = {k:row[k] for k in ['loss','gnorm','lr','tokens_b']}
            if step in train:
                if train[step] != exact:
                    raise ValueError('Deterministic replay trace differs')
                replayed_logs += 1
            train[step] = exact
        elif 'by_source' in row:
            if (step % config['eval_every'] or row['contract_id'] != contract_id or
                    set(row['by_source']) != set(groups) or set(row['by_category']) != set(category_sources)):
                raise ValueError('Evaluation identity/coverage differs')
            def weighted(value, targets, nll=None):
                if integer(value['valid_tokens'], 'valid_tokens') != targets:
                    raise ValueError('Evaluation target count differs')
                finite(value['nll_sum'], 'nll_sum');finite(value['holdout_loss'], 'holdout_loss')
                if not math.isclose(value['holdout_loss'], value['nll_sum']/targets, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Evaluation weighted NLL differs')
                if nll is not None and not math.isclose(value['nll_sum'], nll, rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('Evaluation NLL sum differs')
            for source, value in row['by_source'].items():
                weighted(value, groups[source]['count']*config['seq_len'])
            for category, sources in category_sources.items():
                weighted(row['by_category'][category],
                         sum(row['by_source'][s]['valid_tokens'] for s in sources),
                         sum(row['by_source'][s]['nll_sum'] for s in sources))
            weighted(row, probe['target_tokens'], sum(v['nll_sum'] for v in row['by_source'].values()))
            values = {s:{k:v[k] for k in ['nll_sum','valid_tokens','holdout_loss']} for s,v in row['by_source'].items()}
            if step in evaluated and evaluated[step] != values:
                raise ValueError('Deterministic evaluation replay differs')
            evaluated[step] = values
            if 'fixed_generations' in row:
                items = row['fixed_generations']
                if len(items) != len(prompts) or {g['id'] for g in items} != set(prompt_map):
                    raise ValueError('Fixed generation prompt coverage differs')
                for g in items:
                    p = prompt_map[g['id']]
                    if g['prompt'] != p['text'] or g['category'] != p['category']:
                        raise ValueError('Fixed generation prompt changed')
                    if not 1 <= len(g['generated_ids']) <= 64 or any(type(x) is not int or not 0 <= x < config['vocab_size'] for x in g['generated_ids']):
                        raise ValueError('Fixed generation token IDs invalid')
                digest = identity(items)
                if generation_hashes.get(str(step)) != digest:
                    raise ValueError('Generation trace differs from8rank receipts')
                generated[str(step)] = digest
        else:
            raise ValueError('Unknown trace record')
    if set(train) != set(range(config['log_every'], through_step+1, config['log_every'])):
        raise ValueError('Missing train log coverage')
    if set(evaluated) != set(range(config['eval_every'], through_step+1, config['eval_every'])):
        raise ValueError('Missing evaluation coverage')
    if generated != generation_hashes:
        raise ValueError('Missing fixed generation trace')
    return dict(train_points=len(train), evaluation_points=len(evaluated), repeated_train_records=replayed_logs,
                first_train=train[min(train)] if train else None, last_train=train[max(train)] if train else None,
                generation_sha256_by_step=generated)


def audit_run(source, through_step, *, validation_source=False):
    root = resolve(source)
    snapshot = Snapshot()
    c = snapshot.read(root/'contract.json')
    config = snapshot.read(root/'main.json')
    if c['configurations']['main.json'] != snapshot.bindings[artifact_path(root/'main.json')]:
        raise ValueError('Configuration binding changed')
    if validation_source:
        if c['purpose'] != 'bounded_validation' or config['max_steps'] != 4:
            raise ValueError('Validation audit restricted to actual4step fixture')
    elif (c['purpose'] != 'main_training' or c['main_training_authorized'] is not True or
            config['max_steps'] != 57000):
        raise ValueError('Expected authorized57000step main source')
    integer(through_step, 'through_step')
    if through_step > config['max_steps']:
        raise ValueError('Audit prefix exceeds target')
    for p, digest in {**c['code'], **c['evidence']}.items():
        if sha256_file(resolve(p)) != digest:
            raise ValueError('Producer code/evidence changed: '+p)
    training = resolve(config['out_dir'])
    t = snapshot.read(training/'training_contract.json')
    value = dict(t); tid = value.pop('id')
    if identity(value) != tid or t['world_size'] != 8 or t['config']['max_steps'] != config['max_steps']:
        raise ValueError('Training contract identity differs')
    if t['monitored_evaluation']['generation_steps'] != c['generation_steps']:
        raise ValueError('Generation schedule differs from training identity')
    probe = snapshot.read(root/'probe/probe.json')
    prompts = snapshot.read(resolve(c['prompts']), lines=True)
    mix = snapshot.read(resolve(config['mix']))
    if (sha256_file(root/'probe/probe.json') != t['monitored_evaluation']['probe_sha256'] or
            snapshot.bindings[c['prompts']] != t['monitored_evaluation']['prompts_sha256'] or
            set(mix['sources']) != set(t['source_order']) or config['global_tokens'] != 2097152 or config['seq_len'] != 4096):
        raise ValueError('Audited probe/source/geometry differs')
    paths = sorted((root/'ledgers').glob('*.json'))
    inodes = set()
    def records():
        for path in paths:
            match = re.fullmatch(r'(.+)-rank(\d+)\.json', path.name)
            if not match or path.stat().st_ino in inodes:
                raise ValueError('Duplicate ledger inode or invalid attempt filename')
            inodes.add(path.stat().st_ino)
            record = snapshot.read(path)
            if record['rank'] != int(match[2]):
                raise ValueError('Ledger rank differs from filename')
            yield match[1], record
    counts = aggregate(records(), world=8, through_step=through_step, target_steps=config['max_steps'],
                       seq_len=config['seq_len'], global_tokens=config['global_tokens'], sources=mix['sources'],
                       generation_steps=c['generation_steps'], contract_id=tid)
    trace = audit_trace(snapshot.read(training/'log.jsonl', lines=True, require_newline=True), config=config, through_step=through_step,
                        contract_id=tid, probe=probe, prompts=prompts, generation_hashes=counts['generation_sha256_by_step'])
    checkpoint = training/f'ckpt_{through_step:07d}'
    certificate_path = root/'checkpoint_certificates'/(checkpoint.name+'.json')
    certificate = snapshot.read(certificate_path)
    expected_status = 'passed_actual_checkpoint_storage' if validation_source else 'passed_main_checkpoint_storage'
    if (certificate['status'] != expected_status or certificate['step'] != through_step or
            certificate['training_contract_id'] != tid or resolve(certificate['checkpoint']) != checkpoint):
        raise ValueError('Audited NFS checkpoint certificate differs')
    files = {r['path']:r for r in certificate['files']}
    actual = list(checkpoint.rglob('*'))
    if (len(files) != len(certificate['files']) or any(p.is_symlink() for p in actual) or
            set(files) != {str(p.relative_to(checkpoint)) for p in actual if p.is_file()} or
            sum(r['bytes'] for r in files.values()) != certificate['bytes']):
        raise ValueError('Checkpoint file inventory differs')
    for name, record in files.items():
        if (checkpoint/name).stat().st_size != record['bytes']:
            raise ValueError('Checkpoint file size differs')
    for name in ['.complete', '.metadata', 'meta.json', *[f'rng_rank{rank}.pt' for rank in range(8)]]:
        if name not in files or sha256_file(checkpoint/name) != files[name]['sha256']:
            raise ValueError('Checkpoint metadata/RNG hash differs')
    meta = snapshot.read(checkpoint/'meta.json')
    verify_restart_metadata(meta, config, t, through_step)
    if through_step == config['max_steps']:
        marker = snapshot.read(root/'.done')
        report = snapshot.read(root/'report.json')
        if marker['report_sha256'] != snapshot.bindings[artifact_path(root/'report.json')] or marker['contract_sha256'] != snapshot.bindings[artifact_path(root/'contract.json')]:
            raise ValueError('Source completion proof differs')
        if counts['successful_target_ranks'] != list(range(8)):
            raise ValueError('Target requires successful completion from all8ranks')
        if not validation_source and (report['status'] != 'main_target_completed_review_pending' or resolve(report['final_checkpoint']) != checkpoint):
            raise ValueError('Main target completion differs')
    if paths != sorted((root/'ledgers').glob('*.json')):
        raise ValueError('Ledger set changed during audit')
    snapshot.unchanged()
    categories = Counter()
    for g in probe['groups']:
        categories[g['category']] += counts['source_tokens'][g['source']]
    return dict(status='passed_committed_prefix_audit', utc=datetime.now(timezone.utc).isoformat(),
                source_run=artifact_path(root), validation_source=validation_source, through_step=through_step,
                target_steps=config['max_steps'], goal_complete=False, counts=counts, category_tokens=dict(categories),
                trace=trace, checkpoint_certificate=artifact_path(certificate_path), evidence=snapshot.bindings,
                code={'scripts/audit_main_training.py':sha256_file(Path(__file__))},
                limitations=['Payload SHA/fsync producer certificate reused; metadata/RNG freshly hashed, sizes/inventory checked.',
                             'Actual final GPU restore/resume, native evaluation and retention audit are separate requirements.',
                             'Source ledgers may lose unsaved attempted work on a hard process/host failure.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--through-step', required=True, type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--validation-source', action='store_true')
    a = p.parse_args()
    output = resolve(a.output)
    output.relative_to(resolve('runs'))
    if output.exists() or output.is_relative_to(resolve(a.source_run)):
        raise ValueError('Use a new independent audit output; never mutate the training run')
    result = audit_run(a.source_run, a.through_step, validation_source=a.validation_source)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'report.json', result)
    atomic_json(output/'.done', dict(report_sha256=sha256_file(output/'report.json'), scope='Prefix audit only; not Goal completion'))
    print(json.dumps(dict(status=result['status'], through_step=a.through_step, committed_tokens=result['counts']['committed_tokens'])))
