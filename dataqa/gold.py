"""Gold-set schema, validation and (de)serialisation.

One JSON object per line:
  id               unique string
  text             document text; must already be stable under data.filters.norm_ws when
                   it carries pii_spans (offsets are into this exact string)
  label            'keep' | 'reject'   desired outcome of the filter stage
  reject_reason    required for 'reject': one of REASONS (the reason check_ko is expected to give)
  source_kind      'synthetic' | 'real_reviewed'   denominators are always reported separately
  label_source     free text, e.g. 'author_synthetic', 'agent_prelabel', 'human_review'
  tags             list of strings
  policy_assumption  None, or a policy name: the label encodes a policy choice that can be
                   flipped per run (see apply_policy) without relabeling anything else
  ambiguous        true when no defensible label exists; excluded from precision/recall
  pii_spans        list of {start, end, type[, ambiguous]}; offsets are Python str indices
"""
import json
from pathlib import Path

SCHEMA_VERSION = 1
LABELS = ('keep', 'reject')
SOURCE_KINDS = ('synthetic', 'real_reviewed')
RULE_REASONS = ('short', 'long', 'empty', 'few_hangul', 'low_hangul_ratio', 'dup_lines',
                'ngram_rep', 'low_alnum', 'bad_encoding')
# Rejections a human wants but no threshold rule can express (ads, shopping boilerplate, machine-translated
# spam). Reported separately: they must not enter the precision/recall of the rule-based filter.
SEMANTIC_REASONS = ('semantic_low_quality',)
REASONS = RULE_REASONS + SEMANTIC_REASONS
PII_TYPES = ('phone', 'rrn', 'alien_reg', 'email', 'card', 'account', 'ip')


def validate(rec):
    """Raise ValueError unless `rec` is a well-formed gold record."""
    if not isinstance(rec, dict):
        raise ValueError('record must be an object')
    for key in ('id', 'text', 'label', 'source_kind'):
        if key not in rec:
            raise ValueError(f'missing field: {key}')
    if not isinstance(rec['id'], str) or not rec['id']:
        raise ValueError('id must be a nonempty string')
    if not isinstance(rec['text'], str):
        raise ValueError(f"{rec['id']}: text must be a string")
    if rec['label'] not in LABELS:
        raise ValueError(f"{rec['id']}: label must be one of {LABELS}")
    if rec['source_kind'] not in SOURCE_KINDS:
        raise ValueError(f"{rec['id']}: source_kind must be one of {SOURCE_KINDS}")
    if rec['label'] == 'reject' and rec.get('reject_reason') not in REASONS:
        raise ValueError(f"{rec['id']}: reject needs reject_reason in {REASONS}")
    if rec['label'] == 'keep' and rec.get('reject_reason') is not None:
        raise ValueError(f"{rec['id']}: keep must not carry reject_reason")
    tags = rec.get('tags', [])
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        raise ValueError(f"{rec['id']}: tags must be a list of strings")
    for flag in ('ambiguous',):
        if flag in rec and type(rec[flag]) is not bool:
            raise ValueError(f"{rec['id']}: {flag} must be boolean")
    pa = rec.get('policy_assumption')
    if pa is not None and (not isinstance(pa, str) or not pa):
        raise ValueError(f"{rec['id']}: policy_assumption must be a nonempty string or null")
    spans = rec.get('pii_spans', [])
    if not isinstance(spans, list):
        raise ValueError(f"{rec['id']}: pii_spans must be a list")
    last_end = 0
    for s in sorted(spans, key=lambda s: (s.get('start', -1), s.get('end', -1))):
        if type(s.get('start')) is not int or type(s.get('end')) is not int:
            raise ValueError(f"{rec['id']}: span offsets must be integers")
        if not 0 <= s['start'] < s['end'] <= len(rec['text']):
            raise ValueError(f"{rec['id']}: span out of range")
        if s['start'] < last_end:
            raise ValueError(f"{rec['id']}: overlapping spans")
        if s.get('type') not in PII_TYPES:
            raise ValueError(f"{rec['id']}: span type must be one of {PII_TYPES}")
        if 'ambiguous' in s and type(s['ambiguous']) is not bool:
            raise ValueError(f"{rec['id']}: span ambiguous must be boolean")
        last_end = s['end']
    if spans:
        from data.filters import norm_ws
        if norm_ws(rec['text']) != rec['text']:
            raise ValueError(f"{rec['id']}: text with pii_spans must be stable under norm_ws")


def dumps(records):
    ids = set()
    lines = []
    for rec in records:
        validate(rec)
        if rec['id'] in ids:
            raise ValueError(f"duplicate id: {rec['id']}")
        ids.add(rec['id'])
        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))
    return '\n'.join(lines) + '\n'


def dump(records, path):
    Path(path).write_text(dumps(records), encoding='utf-8')


def loads(text):
    records, ids = [], set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f'line {n}: invalid JSON: {e}') from None
        validate(rec)
        if rec['id'] in ids:
            raise ValueError(f"duplicate id: {rec['id']}")
        ids.add(rec['id'])
        records.append(rec)
    return records


def load(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def apply_policy(records, overrides):
    """Return records with policy-dependent labels flipped per `overrides`.

    overrides maps policy name -> 'keep' | 'reject'. Records whose policy_assumption is not
    named keep their label. A flip to 'reject' needs the record's `policy_reject_reason`.
    Unknown policy names raise, so a typo cannot silently do nothing.
    """
    known = {r.get('policy_assumption') for r in records} - {None}
    for name, outcome in overrides.items():
        if name not in known:
            raise ValueError(f'unknown policy: {name} (known: {sorted(known)})')
        if outcome not in LABELS:
            raise ValueError(f'policy {name}: outcome must be keep or reject')
    out = []
    for r in records:
        name = r.get('policy_assumption')
        if name in overrides and overrides[name] != r['label']:
            r = dict(r)
            r['label'] = overrides[name]
            if overrides[name] == 'reject':
                reason = r.get('policy_reject_reason')
                if reason not in REASONS:
                    raise ValueError(f"{r['id']}: policy flip to reject needs policy_reject_reason")
                r['reject_reason'] = reason
            else:
                r['reject_reason'] = None
        out.append(r)
    return out
