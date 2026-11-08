import json, math, hashlib, datetime
from pathlib import Path
from common.data_contracts import identity, atomic_json
step, previous = 1250, 1000
run = Path('runs/3b_full_v1')
out = Path('runs/main_step1250_review_v1')
def digest(b): return hashlib.sha256(b).hexdigest()
binding_paths = [run/'contract.json', run/'main.json', run/'training/training_contract.json', run/'probe/probe.json']
bindings = {str(p): p.read_bytes() for p in binding_paths}
c = json.loads(bindings[str(run/'contract.json')])
cfg = json.loads(bindings[str(run/'main.json')])
t = json.loads(bindings[str(run/'training/training_contract.json')])
probe = json.loads(bindings[str(run/'probe/probe.json')])
assert digest(bindings[str(run/'contract.json')]) == 'de2b3f4ee2df402aa922c2bb29756242ea5333e4dec4d74bad5db6f86c08bfbe'
assert digest(bindings[str(run/'main.json')]) == c['configurations']['main.json'] == '8b2f1d21902b2bc2c7d233b91f0cb607b2ae10599d262bd17eeec58ed5c4a426'
value = dict(t); tid = value.pop('id')
assert identity(value) == tid == '9e88b26167e223d149c437a292842cd0d458580ca4c836c099728d5dc6e1274c'
assert t['world_size'] == 8 and cfg['max_steps'] == 57000 and cfg['global_tokens'] == 2097152
assert cfg['eval_every'] == 250 and cfg['ckpt_every'] == 500 and cfg['stop_after'] is None
assert digest(bindings[str(run/'probe/probe.json')]) == t['monitored_evaluation']['probe_sha256']
assert t['monitored_evaluation']['generation_steps'] == c['generation_steps'] and step not in c['generation_steps']
assert cfg['holdout'] == probe['holdout'] and cfg['seq_len'] == probe['seq_len'] == 4096
assert len(c['code']) == 49
for p, expected in c['code'].items():
    content = Path(p).read_bytes()
    assert digest(content) == expected, ('producer changed', p)
    bindings[p] = content
trace = run/'training/log.jsonl'
raw = trace.read_bytes()
complete = raw.splitlines(keepends=True)
if complete and not complete[-1].endswith(b'\n'): complete.pop()
cut, found = 0, 0
for line in complete:
    cut += len(line); row = json.loads(line)
    if row.get('step') == step and 'holdout_loss' in row:
        found += 1
        break
assert found == 1, 'Actual1250 evaluation not yet present'
prefix = raw[:cut]
rows = [json.loads(line) for line in prefix.splitlines()]
evaluations = [r for r in rows if 'holdout_loss' in r]
assert [r['step'] for r in evaluations] == [250,500,750,1000,1250]
assert len(probe['groups']) == 6 and probe['total_windows'] == 384 and probe['target_tokens'] == 1572864
categories = {g['source']: g['category'] for g in probe['groups']}
expected_counts = {g['source']: g['count'] for g in probe['groups']}
assert len(categories) == 6 and set(categories.values()) == {'ko','en','code','math'}
def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7), (a,b)
for row in evaluations:
    assert row['contract_id'] == tid
    assert (row['valid_tokens'],row['windows'],row['forwards_per_rank']) == (1572864,384,48)
    assert set(row['by_source']) == set(categories)
    assert set(row['by_category']) == set(categories.values())
    total_nll, total_tokens, total_forwards, groups = 0.0, 0, 0, {}
    for name,r in row['by_source'].items():
        assert (r['valid_tokens'],r['windows'],r['forwards_per_rank']) == (expected_counts[name]*4096,expected_counts[name],expected_counts[name]//8)
        assert r['nll_sum'] >= 0 and r['elapsed_seconds'] > 0 and math.isfinite(r['elapsed_seconds'])
        close(r['holdout_loss'],r['nll_sum']/r['valid_tokens'])
        total_nll += r['nll_sum']; total_tokens += r['valid_tokens']; total_forwards += r['forwards_per_rank']
        cat = groups.setdefault(categories[name],{'nll_sum':0.0,'valid_tokens':0})
        cat['nll_sum'] += r['nll_sum']; cat['valid_tokens'] += r['valid_tokens']
    close(row['nll_sum'],total_nll)
    assert row['valid_tokens'] == total_tokens and row['forwards_per_rank'] == total_forwards
    close(row['holdout_loss'],total_nll/total_tokens)
    for name,cat in groups.items():
        assert row['by_category'][name]['valid_tokens'] == cat['valid_tokens']
        close(row['by_category'][name]['nll_sum'],cat['nll_sum'])
        close(row['by_category'][name]['holdout_loss'],cat['nll_sum']/cat['valid_tokens'])
    assert math.isfinite(row['probe_eval_seconds']) and row['probe_eval_seconds'] > 0
    assert bool(row.get('fixed_generations')) == (row['step'] in c['generation_steps'])
before = next(r for r in evaluations if r['step'] == previous); after = evaluations[-1]
comparisons = {}
for level in ['by_source','by_category']:
    comparisons[level] = {k:{'step1000':v['holdout_loss'],'step1250':after[level][k]['holdout_loss'],'delta':after[level][k]['holdout_loss']-v['holdout_loss']} for k,v in before[level].items()}
for p, content in bindings.items(): assert Path(p).read_bytes() == content, ('binding changed during review',p)
with trace.open('rb') as f: assert f.read(cut) == prefix, 'Earlier log prefix changed'
assert not out.exists(), 'Preserve previous review'
out.mkdir()
(out/'trace_through_eval1250.jsonl').write_bytes(prefix)
assert (out/'trace_through_eval1250.jsonl').read_bytes() == prefix
atomic_json(out/'evaluations.json',evaluations)
report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='passed_fixed_probe_log_aggregate_review',step=step,previous_step=previous,source_run=str(run),
    training_contract_id=tid,config_sha256=c['configurations']['main.json'],
    frozen_code_files_verified=len(c['code']),valid_tokens=after['valid_tokens'],windows=after['windows'],
    holdout_loss_previous=before['holdout_loss'],holdout_loss_current=after['holdout_loss'],
    holdout_loss_delta=after['holdout_loss']-before['holdout_loss'],comparisons=comparisons,
    all_six_source_losses_lower=all(x['delta']<0 for x in comparisons['by_source'].values()),
    eval_seconds=after['probe_eval_seconds'],generation_performed=bool(after.get('fixed_generations')),
    generation_schedule_matches_contract=True,trace_prefix_sha256=digest(prefix),trace_prefix_bytes=len(prefix),
    binding_sha256={p:digest(b) for p,b in bindings.items()},
    checks=['Approved main and training contract identities unchanged','All49 producer code hashes unchanged',
        'Fixed probe definition SHA matches training contract','Same six source labels and fixed target/window/forward counts',
        'All five evaluation rows have finite, consistent source/category/overall NLL',
        'Evaluation schedule and generation presence match contract','Copied log prefix reread identical'],
    limits=['Probe payload file SHA not rerun in this log review',
        'Probe target weighting ko50/en16.67/math16.67/code16.67 differs from training50/30/10/10',
        'Fixed384windows are not full native holdout or official benchmarks',
        'No checkpoint at1250; independently verified committed prefix remains1000',
        'No new generation quality conclusion; repetition/arithmetic/code risks unresolved'],
    actual_final_verified=False,goal_complete=False)
atomic_json(out/'report.json',report)
for p, content in bindings.items(): assert Path(p).read_bytes() == content
with trace.open('rb') as f: assert f.read(cut) == prefix
assert json.loads((out/'report.json').read_text()) == report
print(json.dumps({k:report[k] for k in ['utc','status','step','holdout_loss_previous','holdout_loss_current','comparisons','eval_seconds','all_six_source_losses_lower','goal_complete']},ensure_ascii=False,indent=2))
