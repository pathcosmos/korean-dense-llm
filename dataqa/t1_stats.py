"""T1 statistics from every row of randomly drawn Parquet row groups (no labels, no text stored).

Cluster sampling: the unit drawn at random is a row group (~1000 consecutive documents of one shard),
so documents inside a group are correlated. Confidence intervals therefore bootstrap over GROUPS, not
documents. Shards are the first N of the sorted listing, the same assumption as dataqa.real_sample.

Measured per document (on the text as data/clean.py sees it, i.e. after norm_ws unless noted):
  not_nfc               raw text is not NFC-normalised
  conjoining_jamo       raw text contains conjoining Jamo (decomposed Hangul, U+1100-11FF)
  nfd_rejected          check_ko rejects with few_hangul but passes with pre_normalize=nfc
  special_literal_any   raw text contains a tokenizer special-token string (tokenizer/train_tokenizer.py SPECIAL)
  eos_literal           raw text contains '<|eos|>'
  email_placeholder     text contains an address at a reserved example.com/org/net domain (the dataset's own placeholder)
  pii_email             an email-like match whose domain is not a reserved placeholder domain
  pii_default_changed   mask_pii(default) changes the text
  pii_*                 per-kind detections under the default policy
  pii_opt_in_differs    the opt-in production-safe PiiPolicy (intl phone, date-checked RRN, no git remotes)
                        masks a document differently from the default (more OR fewer masks)
  ko_reject_<reason>    check_ko outcome under the default policy
"""
import argparse
import json
import math
import random
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from data import filters as F
from dataqa import real_sample

import re

# RFC 2606 reserved domains. FineWeb-2 replaces e-mail addresses with firstname.lastname@example.org and
# <word>@example.com; both are placeholders, not personal data.
RESERVED = re.compile(r'@example\.(?:com|org|net)\b', re.I)
SAFE = F.PiiPolicy(phone='intl', rrn='date_checked', email='no_git')
JAMO = range(0x1100, 0x1200)


def special_strings():
    from tokenizer.train_tokenizer import SPECIAL
    return list(SPECIAL)


def scan_group(texts, specials, code=False):
    """code=True: only encoding and special-token facts (Korean filter and PII metrics are meaningless for code)."""
    c = Counter()
    lengths = []
    nfc_policy = F.KoPolicy(pre_normalize='nfc')
    for raw in texts:
        c['docs'] += 1
        text = raw if code else F.norm_ws(raw)
        lengths.append(len(text))
        if not unicodedata.is_normalized('NFC', raw):
            c['not_nfc'] += 1
        if any(ord(ch) in JAMO for ch in raw):
            c['conjoining_jamo'] += 1
        if any(s in raw for s in specials):
            c['special_literal_any'] += 1
        if '<|eos|>' in raw:
            c['eos_literal'] += 1
        if code:
            continue
        if RESERVED.search(raw):
            c['email_placeholder'] += 1
        reason = F.check_ko(text)
        c['ko_reject_' + (reason or 'keep')] += 1
        if reason == 'few_hangul' and F.check_ko(text, policy=nfc_policy) is None:
            c['nfd_rejected'] += 1
        masked = F.mask_pii(text)
        if masked != text:
            c['pii_default_changed'] += 1
        for kind in ('email', 'rrn', 'phone'):
            spans = [t for t in F.find_pii(text, kind) if not RESERVED.search(text[t[0]:t[1]])]
            if spans:
                c['pii_' + kind] += 1
        if F.mask_pii(text, SAFE) != masked:
            c['pii_opt_in_differs'] += 1
    return c, lengths


def bootstrap(groups, key, boots=2000, seed=0):
    """Point rate and 95% percentile interval for sum(key)/sum(docs), resampling whole groups."""
    rng = random.Random(seed)
    total_k, total_n = sum(g.get(key, 0) for g in groups), sum(g['docs'] for g in groups)
    rates = []
    for _ in range(boots):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        n = sum(g['docs'] for g in draw)
        rates.append(sum(g.get(key, 0) for g in draw) / n)
    rates.sort()
    return dict(count=total_k, docs=total_n, rate=total_k / total_n, ci95=[rates[int(.025 * boots)], rates[int(.975 * boots) - 1]])


def summarize(groups, lengths, source, shards_considered):
    keys = sorted({k for g in groups for k in g if k != 'docs'})
    return dict(
        source=source, row_groups=len(groups), docs=sum(g['docs'] for g in groups), shards_considered=shards_considered,
        chars_median=statistics.median(lengths), chars_p90=sorted(lengths)[int(.9 * len(lengths))],
        rates={k: bootstrap(groups, k) for k in keys})


def run(strata, seed, out):
    specials = special_strings()
    result = dict(
        schema=1, seed=seed, unit='row group (cluster); CIs bootstrap over groups',
        caveats=['shards = first N of the sorted listing, N from the registry; equivalence to the original selection is an assumption (불확실)',
                 'document-level rates describe the sampled shards only, not the full corpus',
                 'text is never stored; only counts'],
        sources=[])
    for source, n_groups in strata:
        repo, revision = real_sample.pinned(source)
        reader = real_sample.Reader(repo, revision)
        shards = real_sample.list_shards(repo, revision, real_sample.SUBSETS[source], real_sample.DEFAULT_SHARDS[source],
                                         spread=source in real_sample.SPREAD)
        column = real_sample.TEXT_COLUMN.get(source, 'text')
        groups, lengths = [], []
        for item in real_sample.plan(shards, real_sample.LazyCounts(reader), n_groups, seed):
            table = reader.group(item['file'], item['row_group'], (column,))
            c, ls = scan_group(table.column(column).to_pylist(), specials, code=source in real_sample.TEXT_COLUMN)
            groups.append(c)
            lengths += ls
            reader.cache.clear()
        result['sources'].append(dict(summarize(groups, lengths, source, len(shards)), repo=repo, revision=revision))
        print(f"{source}: {len(groups)} groups, {sum(g['docs'] for g in groups)} docs", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print('wrote', out)


ROWS = [
    ('ko_reject_keep', '필터 통과 (기본 정책)'), ('ko_reject_short', 'short (<200자)'),
    ('ko_reject_dup_lines', 'dup_lines'), ('ko_reject_ngram_rep', 'ngram_rep'),
    ('ko_reject_low_hangul_ratio', 'low_hangul_ratio'), ('ko_reject_few_hangul', 'few_hangul'),
    ('ko_reject_low_alnum', 'low_alnum'), ('ko_reject_bad_encoding', 'bad_encoding'), ('ko_reject_long', 'long'),
    ('not_nfc', 'NFC가 아닌 원문'), ('conjoining_jamo', '분해형(conjoining) 자모 포함'),
    ('nfd_rejected', 'NFD 때문에 거절 (NFC면 통과)'), ('special_literal_any', '특수 토큰 문자열 포함'),
    ('eos_literal', '<|eos|> 문자열 포함'), ('email_placeholder', '예약 도메인(example.*) 이메일 플레이스홀더 포함'),
    ('pii_email', '예약 도메인이 아닌 이메일 형태'), ('pii_phone', '전화번호 형태 (기본 정규식)'),
    ('pii_rrn', '주민번호 형태 (기본 정규식)'), ('pii_default_changed', '기본 마스킹이 문서를 바꿈'),
    ('pii_opt_in_differs', 'opt-in 정책의 마스킹이 기본과 다름')]


def render_markdown(result):
    cols = result['sources']
    head = '| 항목 | ' + ' | '.join(f"{c['source']} ({c['docs']:,}문서/{c['row_groups']}그룹)" for c in cols) + ' |'
    out = [f"seed {result['seed']} · 단위: {result['unit']}", '', head, '|---|' + '---|' * len(cols)]
    for key, label in ROWS:
        cells = []
        for c in cols:
            r = c['rates'].get(key)
            cells.append('0' if r is None else f"{r['rate'] * 100:.2f}% [{r['ci95'][0] * 100:.2f}, {r['ci95'][1] * 100:.2f}] (n={r['count']})")
        out.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    out.append('| 문서 길이 중앙값 / p90 (자) | ' + ' | '.join(f"{c['chars_median']:.0f} / {c['chars_p90']}" for c in cols) + ' |')
    out += ['', '"0"은 표본에서 한 건도 관찰되지 않았다는 뜻이다(전체에 없다는 보증이 아니다).', ''] + [f'- {x}' for x in result['caveats']]
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strata', default='fineweb2_ko:60,hplt_ko:60,wiki_ko:40', help='source:row_groups,...')
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--out')
    ap.add_argument('--render', help='print a markdown table for an existing result JSON and exit')
    a = ap.parse_args(argv)
    if a.render:
        print(render_markdown(json.loads(Path(a.render).read_text(encoding='utf-8'))))
        return
    if not a.out:
        ap.error('--out is required unless --render is used')
    run(real_sample.parse_strata(a.strata, 'fineweb2_ko', 0), a.seed, a.out)


if __name__ == '__main__':
    main()
