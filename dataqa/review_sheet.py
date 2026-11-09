"""Prioritised LOCAL review sheet for a human reviewer (contains raw web text: never commit or publish).

  python -m dataqa.review_sheet --gold dataqa/review100/real_gold100.jsonl --out dataqa/review100/review_for_jun.md

Order of attention, each document listed once with every reason it qualifies:
  1. filter vs label disagreement  - current rules reject a keep, or keep a rule-expressible reject
  2. removal decisions             - every reject: these are the labels that delete data
  3. unlabeled PII detections      - a rule (any preset) fires where the label has no span
  4. random control                - a seeded sample of the remaining keeps, to catch systematic labeller errors
The human decision to record is only: agree, or the corrected label. Corrected labels are applied by
editing the spec and rebuilding with dataqa.make_labels, with label_source set to human_review.
"""
import argparse
import random
import re
from pathlib import Path

from dataqa import gold, ko_rules, pii_rules


def mask_digits(s):
    return re.sub(r'\d', '#', s)


def reasons_for(rec):
    why = []
    got = ko_rules.filter_reason(rec['text'])
    if rec.get('ambiguous'):
        why.append('labeller marked ambiguous')
    pre = rec.get('prereview')
    if pre and pre['status'] != 'confirmed':
        votes = ','.join(pre['votes'])
        why.append(f"pre-review {pre['status']} (reviewers: {votes}; +{len(pre['added_spans'])}/-{len(pre['missing_spans'])} spans)")
    if rec['label'] == 'keep' and got is not None:
        why.append(f'rules reject a keep ({got})')
    if rec['label'] == 'reject' and rec['reject_reason'] in gold.RULE_REASONS and got is None:
        why.append(f"rules keep a reject ({rec['reject_reason']})")
    if rec['label'] == 'reject':
        why.append(f"removal decision ({rec['reject_reason']}" + (', rules also reject' if got else ', rules would keep') + ')')
    labelled = [(s['start'], s['end']) for s in rec.get('pii_spans', [])]
    extra = set()
    for cfg in pii_rules.PRESETS.values():
        for d in pii_rules.apply(rec['text'], cfg)[1]:
            if not any(d['start'] < b and a < d['end'] for a, b in labelled):
                extra.add((d['start'], d['end'], d['type']))
    if extra:
        why.append(f'{len(extra)} unlabeled PII-like detection(s)')
    return why, sorted(extra)


def build(records, controls=8, seed=1, head=700, tail=200):
    flagged, plain = [], []
    for r in records:
        why, extra = reasons_for(r)
        (flagged if why else plain).append((r, why, extra))
    rng = random.Random(seed)
    rng.shuffle(plain)
    control = [(r, ['random control'], e) for r, _, e in plain[:controls]]
    rank = lambda item: (0 if any('rules reject a keep' in w or 'rules keep a reject' in w for w in item[1]) else
                         1 if any(w.startswith('pre-review label_flagged') for w in item[1]) else
                         2 if any(w.startswith('removal') for w in item[1]) else
                         3 if any(w.startswith('pre-review') for w in item[1]) else 4)
    items = sorted(flagged, key=rank) + control
    out = ['# 검수 시트 (LOCAL ONLY — 실제 웹 텍스트 포함, 커밋·게시 금지)', '',
           f'문서 {len(items)}건 / 전체 {len(records)}건. 각 문서에서 확인할 것: **에이전트 라벨이 맞는가**. 틀리면 문서 id와 고칠 라벨만 알려 주세요.', '']
    for r, why, extra in items:
        t = r['text']
        label = r['label'] + (f" ({r['reject_reason']})" if r['label'] == 'reject' else '')
        out += [f"## {r['id']}  — 에이전트 라벨: **{label}**{' / ambiguous' if r.get('ambiguous') else ''}",
                f"- 검수 사유: {'; '.join(why)}",
                f"- 태그: {', '.join(r.get('tags', [])) or '-'} · {len(t)}자",
                f"- 라벨된 PII span: {[(s['type'], 'ambiguous' if s.get('ambiguous') else 'personal') for s in r.get('pii_spans', [])] or '없음'}"]
        if extra:
            out.append('- 라벨 없는 탐지(숫자 마스킹): ' + '; '.join(f"{ty} `{mask_digits(t[a:b])}`" for a, b, ty in extra[:4]))
        out += ['', '```', t[:head] + ('\n…[중략]…\n' + t[-tail:] if len(t) > head + tail else t[head:]), '```', '']
    return '\n'.join(out), len(items)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--controls', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(argv)
    text, n = build(gold.load(a.gold), a.controls, a.seed)
    Path(a.out).write_text(text, encoding='utf-8')
    print(f'{n} documents -> {a.out} (LOCAL; contains raw web text)')


if __name__ == '__main__':
    main()
