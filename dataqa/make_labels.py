"""Turn agent/human label specs (with PII needles) into committable labels (offsets only) + agreement.

  python -m dataqa.make_labels --sheet dataqa/review100/sheet.jsonl --specs a.json b.json --check
  python -m dataqa.make_labels --sheet ... --specs ... --out dataqa/real/prelabel100_labels_v1.jsonl \
        --agent-out dataqa/real/prelabel100_agent_labels_v1.jsonl --agreement dataqa/real/prelabel100_agreement_v1.json

Specs hold real PII strings (needles) and must never be committed; outputs contain offsets and types only.
Documents labelled by several agents form the overlap set: the consensus label is the majority vote
(a tie is marked ambiguous) and Fleiss' kappa over the overlap measures how consistent the agents are
with each other, which has to be known before agent-vs-human agreement can be interpreted.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataqa import gold


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def spans_for(text, items):
    spans = []
    for it in items:
        needle = it['needle']
        if not needle:
            raise ValueError('empty needle')
        starts, pos = [], text.find(needle)
        while pos >= 0:
            starts.append(pos)
            if it.get('occurrence', 'all') == 'first':
                break
            pos = text.find(needle, pos + len(needle))
        if not starts:
            raise ValueError(f"needle of type {it['type']} not found in document")
        for s in starts:
            sp = dict(start=s, end=s + len(needle), type=it['type'])
            if it.get('ambiguous'):
                sp['ambiguous'] = True
            spans.append(sp)
    return sorted(spans, key=lambda s: (s['start'], s['end']))


def build_agent_record(rid, doc, spec):
    rec = dict(id=rid, label=spec['label'], reject_reason=spec.get('reject_reason'),
               pii_spans=spans_for(doc['text'], spec.get('pii', [])), tags=sorted(set(spec.get('tags', []))))
    if spec.get('ambiguous'):
        rec['ambiguous'] = True
    if spec.get('notes'):
        rec['notes'] = spec['notes']
    return rec


def load_specs(paths):
    per_agent = []
    for p in paths:
        spec = json.loads(Path(p).read_text(encoding='utf-8'))
        if not spec.get('agent') or not isinstance(spec.get('docs'), dict):
            raise ValueError(f'{p}: spec needs "agent" and "docs"')
        per_agent.append(spec)
    if len({s['agent'] for s in per_agent}) != len(per_agent):
        raise ValueError('duplicate agent names')
    return per_agent


def category(rec):
    return 'ambiguous' if rec.get('ambiguous') else rec['label']


def fleiss_kappa(rows, categories):
    """rows: list of Counters (category -> raters) with the same rater count. None if undefined."""
    if not rows:
        return None
    n = sum(rows[0].values())
    if n < 2 or any(sum(r.values()) != n for r in rows):
        return None
    N = len(rows)
    p_j = [sum(r.get(c, 0) for r in rows) / (N * n) for c in categories]
    P_i = [(sum(v * v for v in r.values()) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def consensus(rid, recs):
    if len(recs) == 1:
        r = dict(recs[0])
        r['label_source'] = f"agent_prelabel:{recs[0]['agent']}"
        return r
    votes = Counter(category(r) for r in recs)
    (top, k), *rest = votes.most_common()
    tie = bool(rest) and rest[0][1] == k
    label = 'keep' if top == 'ambiguous' else top
    out = dict(id=rid, label=label, reject_reason=None, tags=sorted({t for r in recs for t in r['tags']}),
               label_source=f"agent_prelabel:overlap_majority({len(recs)})")
    if tie or top == 'ambiguous':
        out['ambiguous'] = True
    if label == 'reject':
        reasons = Counter(r['reject_reason'] for r in recs if r['label'] == 'reject' and not r.get('ambiguous'))
        out['reject_reason'] = reasons.most_common(1)[0][0] if reasons else 'semantic_low_quality'
    need = (len(recs) + 1) // 2
    span_votes = Counter((s['start'], s['end'], s['type']) for r in recs for s in r['pii_spans'])
    amb_votes = Counter((s['start'], s['end'], s['type']) for r in recs for s in r['pii_spans'] if s.get('ambiguous'))
    out['pii_spans'] = [dict(start=a, end=b, type=t, **({'ambiguous': True} if amb_votes[(a, b, t)] * 2 >= span_votes[(a, b, t)] else {}))
                        for (a, b, t), v in sorted(span_votes.items()) if v >= need]
    return out


def build(sheet, spec_paths, check_only=False):
    docs = {d['id']: d for d in read_jsonl(sheet)}
    per_agent_records, by_doc = [], defaultdict(list)
    for spec in load_specs(spec_paths):
        for rid, s in spec['docs'].items():
            if rid not in docs:
                raise ValueError(f"unknown document id in spec {spec['agent']}: {rid}")
            rec = build_agent_record(rid, docs[rid], s)
            rec['agent'] = spec['agent']
            gold.validate(dict(rec, text=docs[rid]['text'], source_kind='real_reviewed'))
            per_agent_records.append(rec)
            by_doc[rid].append(rec)
    missing = sorted(set(docs) - set(by_doc))
    final = [consensus(rid, by_doc[rid]) for rid in docs if rid in by_doc]
    overlap = {rid: recs for rid, recs in by_doc.items() if len(recs) > 1}
    cats = ['keep', 'reject', 'ambiguous']
    rows = [Counter(category(r) for r in recs) for recs in overlap.values()]
    sizes = Counter(len(r) for r in overlap.values())
    agreement = dict(
        documents=len(docs), labelled=len(by_doc), unlabelled=len(missing), overlap_documents=len(overlap),
        raters_per_overlap_doc=dict(sizes),
        fleiss_kappa_label=fleiss_kappa(rows, cats) if len(sizes) == 1 else None,
        unanimous_overlap=sum(1 for r in rows if len(r) == 1), categories=cats,
        reject_reason_disagreements=sum(1 for recs in overlap.values()
                                         if len({r['reject_reason'] for r in recs if r['label'] == 'reject'}) > 1),
        note='kappa is undefined (null) when overlap docs have unequal rater counts or all raters always agree on one category')
    return final, per_agent_records, agreement, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sheet', required=True, help='local sheet JSONL with full texts')
    ap.add_argument('--specs', nargs='+', required=True)
    ap.add_argument('--check', action='store_true', help='only verify that every needle resolves')
    ap.add_argument('--out')
    ap.add_argument('--agent-out')
    ap.add_argument('--agreement')
    a = ap.parse_args(argv)
    final, per_agent, agreement, missing = build(a.sheet, a.specs)
    print(f"{len(final)} documents labelled, {len(missing)} without labels, overlap {agreement['overlap_documents']}, "
          f"kappa {agreement['fleiss_kappa_label']}")
    if a.check:
        return
    for path, rows in ((a.out, final), (a.agent_out, per_agent)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + '\n', encoding='utf-8')
    if a.agreement:
        Path(a.agreement).write_text(json.dumps(agreement, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
