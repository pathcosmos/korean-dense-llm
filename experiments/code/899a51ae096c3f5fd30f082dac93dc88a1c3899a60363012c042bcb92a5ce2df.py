"""One-shot planning calculation. No network, deletion, or GPU work."""
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import PROJECT_ROOT

root = Path(PROJECT_ROOT)
inputs = {
    'proposal': 'runs/pretrain_readiness_v1/main_training_proposal_v1.json',
    'batch': 'runs/actual_deterministic_batch_v1/report.json',
    'publication': 'runs/actual_train_validation_v1/3b_validated_checkpoint/publication.json',
    'archive': 'runs/pretrain_readiness_v1/storage_cost_and_retention_proposal_v1.json',
    'hardware': 'logs/hardware_profile_cost_review_admission_20261107.json',
    'plan': 'docs/TRAINING_COST_EXPERIMENT.md',
}
evidence = {k: {'path': p, 'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest()}
            for k, p in inputs.items()}
d = {k: json.loads((root / p).read_text()) for k, p in inputs.items() if k != 'plan'}
f = d['proposal']['rolling2_forecast']
c = d['proposal']['candidate']
files = d['publication']['files']
size = sum(x['bytes'] for x in files)
base = (root / inputs['publication']).parent
assert size == f['measured_checkpoint_bytes']
assert len({x['path'] for x in files}) == len(files)
for entry in files:
    assert (base / entry['path']).is_file()
    assert (base / entry['path']).stat().st_size == entry['bytes'], entry['path']
rates = [x['aggregate_tok_s'] for x in d['batch']['results']
         if x['model'] == '3b' and x['micro_bsz'] == 4]
rate = statistics.median(rates)
tokens = c['max_steps'] * c['global_tokens']
assert math.isclose(rate, f['median_tokens_s']) and tokens == f['target_tokens']
assert c['max_steps'] % c['ckpt_every'] == 0
count = c['max_steps'] // c['ckpt_every']
assert count == f['checkpoints_written'] == 228
seconds = tokens / rate
interval = c['ckpt_every'] * c['global_tokens'] / rate
cold = d['archive']['savings']['candidate_logical_bytes']
g = size / 1e9
rolling = 3 * g
b2_rate, r2_rate, overage = .00695, .015, .01


def b2(s, download):
    return b2_rate * s + overage * max(0, download - 3 * s)


break_even = 3 + (r2_rate - b2_rate) / overage
assert math.isclose(b2(rolling, break_even * rolling), r2_rate * rolling)
assert b2(rolling, 3 * rolling) < r2_rate * rolling < b2(rolling, 4 * rolling)
scenarios = []
for name, avg, download in [
    ('checkpoint_three_one_restore', rolling, g),
    ('checkpoint_three_all_228_readbacks', rolling, count * g),
    ('cold_plus_checkpoint_three_one_restore', cold / 1e9 + rolling, g),
    ('cold_plus_checkpoint_three_all_228_readbacks', cold / 1e9 + rolling, count * g),
]:
    scenarios.append(dict(name=name, average_payload_gb=avg, download_gb=download,
                          b2_storage_and_download_usd=b2(avg, download),
                          r2_standard_storage_usd_before_api=avg * r2_rate))

h = d['hardware']
latest = json.loads((root / 'runs/1b_rehearsal_final_v3/training/log.jsonl')
                    .read_text().splitlines()[-1])
report = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'status': 'planning_and_arithmetic_complete_external_validation_pending',
    'user_cost_basis': {'server_incremental_cost_krw': 0,
                        'source': 'User: 지금 이 서버는 별도 비용 없다',
                        'server_invoice_independently_verified': False},
    'evidence': evidence,
    'calculation_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'checks': {'publication_file_count': len(files), 'current_file_sizes_match': True,
               'rate_and_token_count_match_prior_evidence': True,
               'break_even_crosscheck': True, 'checkpoint_rehashed_this_turn': False,
               'remote_transfer_or_restore_validated': False},
    'baseline': {'checkpoint_bytes': size, 'checkpoint_gb': g,
                 'checkpoint_gib': size / 2**30, 'checkpoints': count,
                 'all_checkpoint_transfer_tb': size * count / 1e12,
                 'latest_three_bytes': 3 * size,
                 'local_three_plus_staging_with_margin_gib': 4 * size * 1.1 / 2**30,
                 'target_tokens': tokens, 'compute_tokens_s': rate,
                 'compute_hours': seconds / 3600, 'interval_minutes': interval / 60,
                 'cold_candidate_payload_bytes': cold,
                 'potential_reclaim_allocated_bytes': d['archive']['savings']['potential_reclaimed_allocated_bytes'],
                 'checkpoint_plus_token_payload_bytes': size + 244804787626,
                 'scope': 'Prior short actual 3B pilot, excluding save/eval/recovery and sustained full-mix differences.'},
    'rate_sources': ['https://www.backblaze.com/cloud-storage/pricing',
                     'https://developers.cloudflare.com/r2/pricing/',
                     'https://www.hetzner.com/storage/storage-box/',
                     'https://wasabi.com/pricing'],
    'rates': {'b2_usd_per_gb_month_assumed': b2_rate,
              'r2_usd_per_gb_month': r2_rate, 'b2_overage_usd_per_gb': overage,
              'hetzner_previous_quote_eur_month_1tb': 3.20,
              'hetzner_previous_quote_eur_month_5tb': 10.90,
              'hetzner_numeric_price_reconfirmed_this_turn': False},
    'cost_scenarios': scenarios,
    'cost_scope': 'Same billing GB unit assumed. Specified average payload for a full month; ignores free allowances, rounding, API, transient replicas, unfinished uploads, taxes and FX. R2 daily peak averaging differs from this steady-state approximation. Not a quote or invoice.',
    'break_even': {'download_to_monthly_average_storage_ratio': break_even,
                   'checkpoint_only_download_gb': break_even * rolling},
    'remote_cadence_candidates': [
        {'steps': n, 'checkpoints': math.ceil(c['max_steps'] / n),
         'transfer_tb': math.ceil(c['max_steps'] / n) * size / 1e12,
         'compute_interval_minutes': n * c['global_tokens'] / rate / 60,
         'risk': 'Host-loss recovery point also includes upload/verification lag.'}
        for n in [250, 500, 1000]],
    'transfer_theoretical': {
        'minimum_payload_mbps_to_keep_up': size * 8 / interval / 1e6,
        'minimum_payload_mbps_for_30_minutes': size * 8 / 1800 / 1e6,
        'minutes_per_checkpoint_by_mbps': {str(m): size * 8 / (m * 1e6) / 60 for m in [100, 200, 500, 1000]},
        'scope': 'Payload lower bounds, not measured network speeds.'},
    'capacity': {
        'nfs_free_gib': h['storage']['project']['free_bytes'] / 2**30,
        'local_free_gib': h['storage']['scratch_parent']['free_bytes'] / 2**30,
        'nfs_headroom_after_prior_future_reservations_and_rolling3_gib':
            (h['storage']['project']['free_bytes'] - f['reserved_other_future_bytes']
             - f['reserve_bytes'] - 4 * size * 1.1) / 2**30,
        'scope': 'Prior future reservations reused; fresh admission required. Local scratch is ephemeral.'},
    'rehearsal_observation': latest,
    'proposed_trial': {'first_providers': ['B2', 'R2 Standard'],
                       'total_upload_gb': 200, 'total_download_gb': 200,
                       'remote_storage_gb': 200, 'duration_hours': 48,
                       'external_budget_usd': 10, 'full_roundtrips_per_provider': 2,
                       'gpu_validation_hours_cap': 1,
                       'status': 'not_authorized_for_concrete_external_trial'},
    'not_performed': ['external upload', 'paid resource creation', 'data deletion',
                      'checkpoint pruning', 'new GPU work', 'main training',
                      'remote recovery drill'],
}
output = Path(__file__).with_name('analysis.json')
if output.exists():
    old = json.loads(output.read_text())
    assert old['evidence'] == evidence and old['calculation_code_sha256'] == report['calculation_code_sha256']
    assert old['cost_scenarios'] == scenarios and old['baseline'] == report['baseline']
    print('Existing planning calculation verified; original observation preserved.')
else:
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
print(json.dumps({'report': str(output.relative_to(root)), 'checked_files': len(files),
                  'scenarios': scenarios, 'latest_observed_step': latest['step']},
                 ensure_ascii=False, indent=2))
