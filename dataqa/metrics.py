"""Filter and PII metrics over a gold set.

Rules for every number produced here:
  * denominators are reported with the value; a zero denominator yields None, never 0.0
    (same discipline as data/probe_similarity_recall.py)
  * synthetic and real_reviewed records are never pooled
  * ambiguous records / spans are excluded from precision and recall and counted separately
"""
from collections import Counter, defaultdict

from dataqa import ko_rules, pii_rules
from dataqa.gold import PII_TYPES, SEMANTIC_REASONS


def ratio(num, den):
    return None if den == 0 else num / den


def prf(tp, fp, fn):
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1,
                precision_den=tp + fp, recall_den=tp + fn)


def by_kind(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['source_kind']].append(r)
    return dict(groups)


def filter_metrics(records, params=ko_rules.DEFAULT):
    """Positive class = 'reject'. Returns {source_kind: {...}}."""
    out = {}
    for kind, recs in sorted(by_kind(records).items()):
        tp = fp = fn = tn = 0
        ambiguous = 0
        reason_match = reason_den = 0
        semantic = Counter()
        per_reason = defaultdict(lambda: Counter())
        errors = []
        for r in recs:
            if r.get('ambiguous'):
                ambiguous += 1
                continue
            got = ko_rules.filter_reason(r['text'], params)
            rejected = got is not None
            if r['label'] == 'reject' and r['reject_reason'] in SEMANTIC_REASONS:
                semantic['n'] += 1
                semantic['caught_by_rules'] += rejected
                continue
            if r['label'] == 'reject':
                want = r['reject_reason']
                per_reason[want]['n'] += 1
                if rejected:
                    tp += 1
                    per_reason[want]['caught'] += 1
                    reason_den += 1
                    if got == want:
                        reason_match += 1
                        per_reason[want]['reason_match'] += 1
                else:
                    fn += 1
                    errors.append(dict(id=r['id'], type='missed_reject', want=want, got=None))
            else:
                if rejected:
                    fp += 1
                    errors.append(dict(id=r['id'], type='false_reject', want=None, got=got,
                                       policy=r.get('policy_assumption')))
                else:
                    tn += 1
        out[kind] = dict(
            n=len(recs), ambiguous_excluded=ambiguous, tn=tn,
            reject=prf(tp, fp, fn),
            reason_match=dict(match=reason_match, den=reason_den, rate=ratio(reason_match, reason_den)),
            semantic_rejects=dict(n=semantic['n'], caught_by_rules=semantic['caught_by_rules']),
            per_reason={k: dict(v) for k, v in sorted(per_reason.items())},
            errors=errors)
    return out


def _overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def pii_metrics(records, cfg=pii_rules.PiiConfig()):
    """Span-level PII metrics for one detector config. Returns {source_kind: {...}}."""
    out = {}
    for kind, recs in sorted(by_kind(records).items()):
        per_type = {t: Counter() for t in PII_TYPES}
        tp = partial = wrong_type = fp = ambiguous_hits = 0
        hard_negative_docs = hard_negative_fp_docs = 0
        errors = []
        for r in recs:
            text = r['text']
            spans = r.get('pii_spans', [])
            _, dets = pii_rules.apply(text, cfg)
            gold = [s for s in spans if not s.get('ambiguous')]
            amb = [s for s in spans if s.get('ambiguous')]
            if 'pii_hard_negative' in r.get('tags', []):
                hard_negative_docs += 1
                if [d for d in dets if not any(_overlap(d, a) for a in amb)]:
                    hard_negative_fp_docs += 1
            for g in gold:
                per_type[g['type']]['n'] += 1
                hit_exact = any(d['start'] == g['start'] and d['end'] == g['end']
                                and g['type'] in pii_rules.FAMILY[d['type']] for d in dets)
                hit_any = any(_overlap(d, g) and g['type'] in pii_rules.FAMILY[d['type']] for d in dets)
                per_type[g['type']]['exact'] += hit_exact
                per_type[g['type']]['overlap'] += hit_any
                if not hit_any:
                    errors.append(dict(id=r['id'], type='missed_span', span_type=g['type'],
                                       text=text[g['start']:g['end']]))
            for d in dets:
                if any(_overlap(d, a) for a in amb):
                    ambiguous_hits += 1
                    continue
                overl = [g for g in gold if _overlap(d, g)]
                if not overl:
                    fp += 1
                    errors.append(dict(id=r['id'], type='false_detection', det_type=d['type'],
                                       text=text[d['start']:d['end']]))
                elif any(g['type'] in pii_rules.FAMILY[d['type']] for g in overl):
                    if any(g['start'] == d['start'] and g['end'] == d['end']
                           and g['type'] in pii_rules.FAMILY[d['type']] for g in overl):
                        tp += 1
                    else:
                        partial += 1
                        errors.append(dict(id=r['id'], type='partial_span', det_type=d['type'],
                                           text=text[d['start']:d['end']]))
                else:
                    wrong_type += 1
                    errors.append(dict(id=r['id'], type='wrong_type', det_type=d['type'],
                                       text=text[d['start']:d['end']]))
        det_total = tp + partial + wrong_type + fp
        types = {}
        for t, c in per_type.items():
            if c['n'] == 0:
                continue
            types[t] = dict(n=c['n'], exact=c['exact'], overlap=c['overlap'],
                            recall_exact=ratio(c['exact'], c['n']),
                            recall_overlap=ratio(c['overlap'], c['n']),
                            implemented=t in pii_rules.IMPLEMENTED_TYPES)
        out[kind] = dict(
            detections=det_total, tp_exact=tp, partial=partial, wrong_type=wrong_type, false_positive=fp,
            precision_exact=ratio(tp, det_total), precision_overlap=ratio(tp + partial, det_total),
            ambiguous_hits_excluded=ambiguous_hits,
            hard_negative_docs=hard_negative_docs, hard_negative_docs_with_detection=hard_negative_fp_docs,
            per_type=types, errors=errors)
    return out
