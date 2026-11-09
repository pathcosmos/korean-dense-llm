"""Evaluate filter and PII rules against a gold set and write report.json + report.md.

  python -m dataqa.run --gold dataqa/gold/seed_v1.jsonl --out runs/dataqa/seed_v1
  python -m dataqa.run --gold ... --out ... --policy nfd_text_is_valid_korean=reject

Refuses to report (exit 2) if a parity gate fails: a rule copy that disagrees with the
production implementation would make every number meaningless.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

from common.data_contracts import atomic_json, sha256_file
from data import filters as F
from dataqa import gold, ko_rules, metrics, pii_rules

SWEEP = {
    'min_chars': [100, 150, 200, 250, 300],
    'min_hangul_ratio': [0.20, 0.25, 0.30, 0.35, 0.40],
    'max_dup_line_frac': [0.20, 0.30, 0.40, 0.50],
    'max_ngram_rep': [0.20, 0.30, 0.40, 0.50],
    'min_alnum_ratio': [0.40, 0.50, 0.60],
    'pre_normalize': ['none', 'nfc'],
}
UNCERTAIN_CLAIMS = [
    '전화번호 접두 허용 목록(intl_prefix): 작성자의 일반 지식이며 공식 번호 체계로 검증하지 않았다 (불확실).',
    '주민등록번호 7번째 자리 5-8을 외국인 등록번호로 본 가정은 검증하지 않았다 (불확실).',
    '주민등록번호 검증자리(check digit)는 2020년 번호 체계 변경 이후 발급분에 적용되는지 불확실하여 구현하지 않았다.',
    '합성 gold에서는 후보 규칙과 케이스를 같은 작성자가 만들었으므로 후보의 높은 점수는 설계 의도 확인이지 일반화 성능이 아니다.',
]
SYNTHETIC_NOTE = ('합성 gold만으로 만든 수치다. 작성자가 케이스와 라벨을 모두 정했으므로 하네스가 동작하고 규칙이 '
                  '설계대로 반응한다는 확인일 뿐, 실제 코퍼스의 정밀도/재현율이나 임계값 권고가 아니다.')


def code_identity():
    files = sorted(Path(__file__).parent.glob('*.py')) + [Path(F.__file__)]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b'\0' + f.read_bytes() + b'\0')
    return dict(sha256=h.hexdigest(), files=[f.name for f in files])


def parity(records):
    texts = [r['text'] for r in records]
    return dict(ko=ko_rules.parity_mismatches(texts), pii=pii_rules.parity_mismatches(texts))


def decisions(records, params):
    return {r['id']: ko_rules.filter_reason(r['text'], params) is not None for r in records if not r.get('ambiguous')}


def sweep(records, base=ko_rules.DEFAULT):
    base_dec = decisions(records, base)
    rows = []
    for name, values in SWEEP.items():
        for v in values:
            params = type(base)(**{**base.as_dict(), name: v})
            dec = decisions(records, params)
            m = metrics.filter_metrics(records, params)
            rows.append(dict(
                param=name, value=v, is_default=getattr(base, name) == v,
                flips_vs_default=sum(dec[k] != base_dec[k] for k in dec),
                metrics={kind: dict(precision=d['reject']['precision'], recall=d['reject']['recall'],
                                    f1=d['reject']['f1'], fp=d['reject']['fp'], fn=d['reject']['fn'])
                         for kind, d in m.items()}))
    return rows


def evaluate(records, overrides=None, params=ko_rules.DEFAULT):
    if overrides:
        records = gold.apply_policy(records, overrides)
    return dict(
        filter=metrics.filter_metrics(records, params),
        filter_nfc=metrics.filter_metrics(records, ko_rules.KoParams(**{**params.as_dict(), 'pre_normalize': 'nfc'})),
        pii={name: dict(config=cfg.as_dict(), metrics=metrics.pii_metrics(records, cfg))
             for name, cfg in pii_rules.PRESETS.items()},
        sweep=sweep(records, params))


def build_report(gold_path, overrides=None, params=ko_rules.DEFAULT):
    records = gold.load(gold_path)
    gate = parity(records)
    if gate['ko'] or gate['pii']:
        raise SystemExit(f'parity gate failed: ko={gate["ko"][:3]} pii={gate["pii"][:3]}')
    kinds = sorted({r['source_kind'] for r in records})
    body = evaluate(records, overrides, params)
    return dict(
        schema_version=1,
        gold=dict(path=str(gold_path), sha256=sha256_file(gold_path), records=len(records),
                  by_kind={k: sum(r['source_kind'] == k for r in records) for k in kinds}),
        policy_overrides=overrides or {},
        policy_assumptions=sorted({r['policy_assumption'] for r in records if r.get('policy_assumption')}),
        params=params.as_dict(),
        parity=dict(ko_mismatches=0, pii_mismatches=0, checked_texts=len(records)),
        code=code_identity(), python=platform.python_version(),
        notes=[SYNTHETIC_NOTE] if kinds == ['synthetic'] else [],
        uncertain_claims=UNCERTAIN_CLAIMS,
        **body)


def pct(x):
    return 'n/a' if x is None else f'{x:.3f}'


def render_markdown(rep):
    L = ['# dataqa report', '',
         f"gold: `{rep['gold']['path']}` sha256 `{rep['gold']['sha256'][:16]}…` records {rep['gold']['records']} {rep['gold']['by_kind']}",
         f"code identity `{rep['code']['sha256'][:16]}…` · python {rep['python']} · parity gate: passed ({rep['parity']['checked_texts']} texts)",
         f"policy overrides: {rep['policy_overrides'] or 'none (labels as stored)'}"]
    for n in rep['notes']:
        L += ['', f'> {n}']
    L += ['', '## 정제 필터 (positive = reject)', '',
          '| kind | n | tp | fp | fn | precision | recall | F1 | reason match | ambiguous 제외 |', '|---|---|---|---|---|---|---|---|---|---|']
    for kind, d in rep['filter'].items():
        r = d['reject']
        L.append(f"| {kind} | {d['n']} | {r['tp']} | {r['fp']} | {r['fn']} | {pct(r['precision'])} ({r['precision_den']}) | "
                 f"{pct(r['recall'])} ({r['recall_den']}) | {pct(r['f1'])} | {pct(d['reason_match']['rate'])} ({d['reason_match']['den']}) | {d['ambiguous_excluded']} |")
    for kind, d in rep['filter'].items():
        sem = d['semantic_rejects']
        if sem['n']:
            L += ['', f"규칙으로 표현할 수 없는 의미적 거절 ({kind}): {sem['n']}건 중 규칙이 잡은 것 {sem['caught_by_rules']}건 "
                      '(위 precision/recall 분모에서 제외)']
    for kind, d in rep['filter'].items():
        if d['errors']:
            L += ['', f'오류 목록 ({kind}):', '']
            L += [f"- `{e['id']}` {e['type']} (want {e['want']}, got {e['got']}"
                  + (f", policy {e['policy']}" if e.get('policy') else '') + ')' for e in d['errors']]
    L += ['', '`pre_normalize=nfc` 후보를 적용하면:', '', '| kind | fp | fn | precision | recall |', '|---|---|---|---|---|']
    for kind, d in rep['filter_nfc'].items():
        r = d['reject']
        L.append(f"| {kind} | {r['fp']} | {r['fn']} | {pct(r['precision'])} | {pct(r['recall'])} |")
    L += ['', '## PII 규칙 (span 단위, exact 일치)', ''] + [f'> 불확실: {c}' for c in rep['uncertain_claims']] + ['',
          '| preset | kind | detections | precision exact | recall phone | recall rrn | recall alien_reg | recall email | hard-negative docs hit |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, item in rep['pii'].items():
        for kind, d in item['metrics'].items():
            t = d['per_type']
            cell = lambda k: (f"{pct(t[k]['recall_exact'])} ({t[k]['exact']}/{t[k]['n']})" if k in t else 'n/a')
            L.append(f"| {name} | {kind} | {d['detections']} | {pct(d['precision_exact'])} | {cell('phone')} | {cell('rrn')} | "
                     f"{cell('alien_reg')} | {cell('email')} | {d['hard_negative_docs_with_detection']}/{d['hard_negative_docs']} |")
    base = rep['pii']['baseline']['metrics']
    for kind, d in base.items():
        missing = {k: v for k, v in d['per_type'].items() if not v['implemented']}
        if missing:
            L += ['', f'미구현 PII 유형 ({kind}): ' + ', '.join(f"{k} {v['overlap']}/{v['n']}" for k, v in missing.items())
                  + ' — 어떤 규칙도 탐지하지 않으므로 recall 0이 아니라 "범위 밖"이다.']
    L += ['', '## 임계값 sweep (한 번에 한 파라미터)', '',
          '**하네스 시연이다. 합성 경계 케이스는 임계값 근처에 일부러 놓았으므로 값을 옮기면 판정이 뒤집힐 뿐 최적값 근거가 아니다.**',
          '', '| param | value | default | flips | ' + ' | '.join(f'{k} P / R' for k in rep['gold']['by_kind']) + ' |',
          '|---|---|---|---|' + '---|' * len(rep['gold']['by_kind'])]
    for row in rep['sweep']:
        cells = ' | '.join(f"{pct(row['metrics'][k]['precision'])} / {pct(row['metrics'][k]['recall'])}" for k in rep['gold']['by_kind'])
        L.append(f"| {row['param']} | {row['value']} | {'●' if row['is_default'] else ''} | {row['flips_vs_default']} | {cells} |")
    return '\n'.join(L) + '\n'


def parse_policy(items):
    out = {}
    for it in items or []:
        name, _, outcome = it.partition('=')
        if not outcome:
            raise SystemExit('--policy expects name=keep|reject')
        out[name] = outcome
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--out', required=True, help='new output directory')
    ap.add_argument('--policy', action='append', help='policy_assumption=keep|reject (repeatable)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'refusing to overwrite non-empty output directory: {out}')
    rep = build_report(a.gold, parse_policy(a.policy))
    atomic_json(out / 'report.json', rep)
    (out / 'report.md').write_text(render_markdown(rep), encoding='utf-8')
    print(f'wrote {out}/report.json and report.md')


if __name__ == '__main__':
    main()
