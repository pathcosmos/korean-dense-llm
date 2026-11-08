"""Read certified actual results and write a proposal; never starts training.

All time/storage figures are measured-input projections. A plan/.done is not
F8/F9 completion or authorization to train/prune. Existing drafts stay intact.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.run_native_evaluation import certified

GIB = 2**30


def geometry(config, world=8):
    for key in ('seq_len', 'micro_bsz', 'global_tokens', 'max_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Positive integer geometry required: '+key)
    if type(world) is not int or world < 1:
        raise ValueError('Positive integer world required')
    per_micro = config['micro_bsz']*config['seq_len']*world
    if config['global_tokens'] % per_micro:
        raise ValueError('Global batch must be exactly divisible; no rounded token budget')
    return dict(world=world, accumulation=config['global_tokens']//per_micro,
                tokens_per_micro_global=per_micro, tokens_per_step=config['global_tokens'],
                target_tokens=config['max_steps']*config['global_tokens'])


def forecast(config, rates, checkpoint_bytes, free_bytes, reserved_future_bytes=0, reserve_gib=200):
    g = geometry(config)
    if (not rates or any(type(r) not in (int,float) or not math.isfinite(r) or r <= 0 for r in rates) or
            type(checkpoint_bytes) is not int or checkpoint_bytes < 1 or
            type(free_bytes) is not int or free_bytes < 0 or
            type(reserved_future_bytes) is not int or reserved_future_bytes < 0 or
            type(reserve_gib) not in (int,float) or not math.isfinite(reserve_gib) or reserve_gib < 0):
        raise ValueError('Finite positive actual rates/bytes and nonnegative reserves required')
    for key in ('ckpt_every','eval_every','keep_ckpts'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Positive schedule required: '+key)
    steps = config['max_steps']
    checkpoint_count = steps//config['ckpt_every'] + int(bool(config['save_final']) and steps % config['ckpt_every'] != 0)
    resident = min(checkpoint_count, config['keep_ckpts'])
    # The next complete checkpoint coexists with retained checkpoints until pruning.
    peak_count = min(checkpoint_count, config['keep_ckpts']+1)
    peak = math.ceil(peak_count*checkpoint_bytes*1.1)
    required = peak+reserved_future_bytes+math.ceil(reserve_gib*GIB)
    rate = statistics.median(rates)
    return dict(**g, actual_rate_samples=rates, median_tokens_s=rate,
                compute_seconds=g['target_tokens']/rate,
                compute_days=g['target_tokens']/rate/86400,
                compute_days_range=[g['target_tokens']/max(rates)/86400,g['target_tokens']/min(rates)/86400],
                evaluations=steps//config['eval_every'], checkpoints_written=checkpoint_count,
                checkpoint_interval_compute_seconds=config['ckpt_every']*g['tokens_per_step']/rate,
                measured_checkpoint_bytes=checkpoint_bytes, checkpoint_capacity_margin=1.1,
                retained_checkpoint_count=resident, peak_checkpoint_count_including_staging=peak_count,
                checkpoint_peak_bytes_with_margin=peak, reserved_other_future_bytes=reserved_future_bytes,
                reserve_bytes=math.ceil(reserve_gib*GIB), free_bytes_at_measurement=free_bytes,
                proposed_new_peak_and_reserve_bytes=required, fits_with_other_future_peaks=required<=free_bytes,
                remaining_after_peak_and_reserve_bytes=free_bytes-required,
                timing_scope='Short actual3B deterministic pilot compute only. Add full-mix/eval/save/recovery costs; no ten-day guarantee.')


def make_plan(a):
    from scripts.resource_admission import inspect
    from scripts.run_local_split import size_tree
    from train.train import DEFAULT
    from train.probe import verify_probe
    batch, bc = certified(a.batch_run, 'candidate_selected')
    validation, vc = certified(a.validation_run, 'passed_bounded_actual_training')
    native, nc = certified(a.native_run, 'passed_actual_native_evaluation')
    tokens, tc = certified(a.token_run, 'passed')
    if (not bc['deterministic_algorithms'] or not vc['deterministic_algorithms'] or
            vc['batch_geometry'] != batch['selected_model_settings']):
        raise ValueError('Selected deterministic geometry is not actually validated')
    mix = resolve(tokens['mix'])
    if sha256_file(mix) != tokens['mix_sha256']:
        raise ValueError('Final mix content differs')
    rehearsal = resolve(a.rehearsal_run); rc = json.loads((rehearsal/'contract.json').read_text())
    for p, h in rc['code'].items():
        if sha256_file(resolve(p)) != h:
            raise ValueError('Active rehearsal producer changed: '+p)
    probe = verify_probe(rehearsal/'probe')
    selected = batch['selected_model_settings']['3b']
    if selected['micro_bsz'] != 4 or selected['global_tokens'] != 2097152:
        raise ValueError('Unexpected selected3B geometry; review new proposal explicitly')
    candidate = dict(DEFAULT, model='3b', vocab_size=65536, seq_len=4096, **selected,
                     max_steps=57000, warmup=1000, decay_frac=.15, min_lr_ratio=.1,
                     compile=False, mix=artifact_path(mix), holdout=probe['holdout'],
                     out_dir='runs/3b_full_v1', ckpt_every=250, eval_every=250,
                     log_every=10, keep_ckpts=2, save_final=True, sampler_version=2, stop_after=None)
    geometry(candidate)
    rates = [r['aggregate_tok_s'] for r in batch['results']
             if r['model']=='3b' and r['micro_bsz']==selected['micro_bsz'] and r['status']=='passed']
    if len(rates) != 2:
        raise ValueError('Selected3B repeated actual throughput evidence missing')
    row = next(r for r in validation['results'] if r['model']=='3b')
    publication = resolve(row['published_checkpoint']['path'])/'publication.json'
    if sha256_file(publication) != row['published_checkpoint']['publication_sha256']:
        raise ValueError('Measured3B checkpoint publication changed')
    if not row['checkpoint_comparison']['exact'] or row['trajectory']['loss_max_abs'] != 0:
        raise ValueError('Actual3B exact-resume evidence missing')
    hw = inspect()
    remaining = max(0, rc['forecast_checkpoint_bytes']-size_tree(rehearsal/'training'))
    # Late final native worker can still create up to8GiB. Existing files are
    # already included in measured free space, so do not charge them again.
    other_future = remaining+8*GIB
    projected = forecast(candidate, rates, row['published_checkpoint']['bytes'],
                         hw['storage']['project']['free_bytes'], other_future)
    retain_all = dict(candidate, keep_ckpts=candidate['max_steps']+1)
    all_projection = forecast(retain_all, rates, row['published_checkpoint']['bytes'],
                              hw['storage']['project']['free_bytes'], other_future)
    import torch
    from model.model import Transformer, get_config
    with torch.device('meta'):
        model = Transformer(get_config('3b', vocab_size=65536, max_seq_len=4096))
    parameter_count = model.num_params()
    deps = ['scripts/plan_main_training.py','scripts/storage_budget.py','scripts/resource_admission.py']
    files = [resolve(x)/name for x in [a.batch_run,a.validation_run,a.native_run,a.token_run]
             for name in ('report.json','contract.json','.done')]
    files += [mix, publication, rehearsal/'contract.json', rehearsal/'probe/probe.json']
    proposal = dict(status='draft_not_ready_not_authorized', candidate=candidate,
                    parameter_count=parameter_count, rolling2_forecast=projected,
                    preserve_all_forecast=all_projection, hardware=hw,
                    evidence={artifact_path(p):sha256_file(p) for p in files},
                    code={**rc['code'],**{p:sha256_file(resolve(p)) for p in deps}},
                    required_approvals=['Start3B main training after final review',
                      'Explicitly approve pruning only newly produced3B checkpoints to newest2,or provision extra permanent storage; existing diagnostic/rehearsal data remain preserved'],
                    actual_completed=['Final source/split/tokenizer/full tokens/mix evidence',
                      'Selected actual3B deterministic throughput repeat and exact model/optimizer/RNG resume',
                      'Diagnostic native evaluation coverage'],
                    remaining=['Actual1B5000-step full rehearsal completion and trained native evaluation',
                      'Review source/category curves, fixed generations, nonfinite events, actual long-run save/eval overhead',
                      'Freeze3B source-aware evaluation/generation and content-certification entry; validate actual3B integration if that entry changes',
                      'Recheck fresh hardware, free space, main output absence and checkpoint retention permission',
                      'Final readiness report and user review; current proposal cannot authorize training'],
                    recovery=dict(entry='train.deterministic_train (existing validated base; final monitored entry still to freeze)',
                      invocation_proposal=['torchrun','--standalone','--nproc_per_node=8','--module','train.deterministic_train','--config','<reviewed3Bconfig>'],
                      procedure=['Stop only failed main job process group; preserve complete checkpoints and logs',
                        'Check .complete and full DCP/RNG/meta content certificate,then select latest certified checkpoint',
                        'Restore same8GPU count,tokenizer/mix/source order,sampler version,seed,global/micro batch and deterministic flags',
                        'Restart identical frozen config/entry; same-step RNG/data/model/optimizer contract must match',
                        'Keep incomplete checkpoint paths for inspection; never rewrite metadata to accept a different contract']),
                    evaluation=dict(probe='Fixed six-source64windows/file,1,572,864 targets; prefix probe,not uniform full holdout PPL',
                      cadence_steps=250, fixed_generation='Freeze bilingual6prompt generation schedule after actual rehearsal review',
                      full_native='Full raw-document ko/en target coverage before final review and at selected saved checkpoints; GPU work must be scheduled explicitly',
                      official_benchmarks='User-managed separate work; not executed or marked passed'),
                    risks=['Short pilot rate may differ on full mix over days; timings exclude eval/save/recovery',
                      'The1B LR trajectory is not evidence for3B long-run convergence;3B peakLR3e-4 has bounded actual validation only',
                      'Rolling2 pruning is not authorized; keep_ckpts2 would actually delete older new-run checkpoints',
                      'Preserving all228 planned3B checkpoints exceeds current permanent storage',
                      'Legacy train/configs/3b.json is an old micro8/compile-on draft and not the current validated candidate',
                      'Source-aware monitored3B entry and final runtime watchdog must be frozen/reviewed before main execution'],
                    scope='Planning only. Does not launch main training,delete/overwrite corpus,prune checkpoints,upload externally or assert F9 readiness.')
    return proposal


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-run', default='runs/actual_deterministic_batch_v1')
    p.add_argument('--validation-run', default='runs/actual_train_validation_v1')
    p.add_argument('--native-run', default='runs/native_eval_validation_v1')
    p.add_argument('--token-run', default='runs/f6_tokens_v1')
    p.add_argument('--rehearsal-run', default='runs/1b_rehearsal_final_v3')
    args = p.parse_args(); out = resolve(args.output)
    if out.exists():
        raise ValueError('Preserve existing proposal; use a new output path')
    report = make_plan(args)
    atomic_json(out, report)
    atomic_json(Path(str(out)+'.done'), dict(sha256=sha256_file(out),status='proposal_only_not_stage_completion'))
    print(json.dumps({k:report[k] for k in ('status','parameter_count','rolling2_forecast','preserve_all_forecast')}))
