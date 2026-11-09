"""Second-pass pre-review of labels that no human has looked at yet.

Independent reviewers (fresh contexts, blind to the existing label and to the filter) label the same documents
again. This module compares them with the existing label and annotates each document with a `prereview`
record. It NEVER changes `label`, `reject_reason`, `pii_spans` or `label_source`: a pre-review narrows down what a
human should look at, it does not replace the human. Reviewers share the original labellers' model family, so
agreement raises confidence only modestly and correlated mistakes remain possible.

Statuses per document:
  confirmed      every reviewer agrees on the category and on the PII spans
  label_flagged  at least one reviewer disagrees on keep / reject / ambiguous
  span_flagged   categories agree but PII spans differ (spans added or missing, or personal vs organisational)

  python -m dataqa.prereview --labels dataqa/real/prelabel100_labels_v2_human.jsonl \
      --reviewers dataqa/real/keepreview56_reviewer_labels_v1.jsonl \
      --out dataqa/real/prelabel100_labels_v3_prereview.jsonl --summary dataqa/real/keepreview56_summary_v1.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataqa.make_labels import category, fleiss_kappa, read_jsonl


def span_key(s):
    return (s['start'], s['end'], s['type'], bool(s.get('ambiguous')))


def classify(label, reviewer_recs):
    cats = [category(r) for r in reviewer_recs]
    base = category(label)
    original = {span_key(s) for s in label.get('pii_spans', [])}
    seen = [{span_key(s) for s in r.get('pii_spans', [])} for r in reviewer_recs]
    found = set().union(*seen) if seen else set()
    common = set.intersection(*seen) if seen else set()
    added = sorted(found - original)
    missing = sorted(original - common)           # in the original but not found by every reviewer
    if any(c != base for c in cats):
        status = 'label_flagged'
    elif added or missing or any(s != original for s in seen):
        status = 'span_flagged'
    else:
        status = 'confirmed'
    return dict(status=status, reviewers=[r['agent'] for r in reviewer_recs], votes=cats,
                reviewer_reasons=[r.get('reject_reason') for r in reviewer_recs],
                added_spans=[list(x) for x in added], missing_spans=[list(x) for x in missing])


def apply(labels, reviewer_records):
    """Return (annotated labels, summary). Labels without reviewers are passed through untouched."""
    by_doc = defaultdict(list)
    for r in reviewer_records:
        by_doc[r['id']].append(r)
    unknown = set(by_doc) - {l['id'] for l in labels}
    if unknown:
        raise ValueError(f'reviewer labels for unknown documents: {len(unknown)}')
    out, statuses, kinds = [], Counter(), []
    pair_agree = pair_total = orig_agree = 0
    for l in labels:
        recs = by_doc.get(l['id'])
        if not recs:
            out.append(l)
            continue
        if len({r['agent'] for r in recs}) != len(recs):
            raise ValueError(f"{l['id']}: duplicate reviewer")
        l = dict(l)
        l['prereview'] = classify(l, recs)
        out.append(l)
        statuses[l['prereview']['status']] += 1
        kinds.append([category(r) for r in recs])
        orig_agree += all(category(r) == category(l) for r in recs)
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                pair_total += 1
                pair_agree += category(recs[i]) == category(recs[j])
    sizes = Counter(len(k) for k in kinds)
    rows = [Counter(k) for k in kinds]
    summary = dict(
        reviewed_documents=len(kinds), reviewers_per_document=dict(sizes), statuses=dict(statuses),
        reviewer_pair_category_agreement=None if not pair_total else pair_agree / pair_total,
        fleiss_kappa_reviewers=fleiss_kappa(rows, ['keep', 'reject', 'ambiguous']) if len(sizes) == 1 else None,
        original_vs_reviewers_category_agreement=None if not kinds else orig_agree / len(kinds),
        note=('reviewers are fresh contexts of the same model family as the original labellers; agreement is weak '
              'evidence of correctness, disagreement is a useful flag. Labels were NOT changed.'))
    return out, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--reviewers', required=True, help='per-reviewer label records (make_labels --agent-out)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary', required=True)
    a = ap.parse_args(argv)
    labels, summary = apply(read_jsonl(a.labels), read_jsonl(a.reviewers))
    Path(a.out).write_text('\n'.join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in labels) + '\n', encoding='utf-8')
    Path(a.summary).write_text(json.dumps(summary, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
