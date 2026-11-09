import unittest

from dataqa import prereview, review_sheet, seed_gold
import random


def label(rid, lab='keep', reason=None, spans=(), ambiguous=False):
    r = dict(id=rid, label=lab, reject_reason=reason, pii_spans=list(spans), tags=[], label_source='agent_prelabel:A')
    if ambiguous:
        r['ambiguous'] = True
    return r


def rev(rid, agent, lab='keep', reason=None, spans=(), ambiguous=False):
    r = label(rid, lab, reason, spans, ambiguous)
    r['agent'] = agent
    return r


SP = dict(start=3, end=16, type='phone')


class PreReviewTest(unittest.TestCase):
    def test_statuses(self):
        labels = [label('a'), label('b'), label('c'), label('d', spans=[SP]), label('e', spans=[SP]), label('f')]
        reviewers = [
            rev('a', 'R1'), rev('a', 'R2'),                                                         # confirmed
            rev('b', 'R1'), rev('b', 'R2', 'reject', 'semantic_low_quality'),                       # one dissent -> label_flagged
            rev('c', 'R1', ambiguous=True), rev('c', 'R2'),                                         # ambiguous vote -> label_flagged
            rev('d', 'R1', spans=[SP]), rev('d', 'R2', spans=[SP]),                                 # spans agree -> confirmed
            rev('e', 'R1', spans=[dict(SP, ambiguous=True)]), rev('e', 'R2', spans=[SP]),           # personal vs org differs -> span_flagged
            rev('f', 'R1', spans=[SP]), rev('f', 'R2'),                                             # added span by one -> span_flagged
        ]
        out, summary = prereview.apply(labels, reviewers)
        status = {l['id']: l['prereview']['status'] for l in out}
        self.assertEqual(status, dict(a='confirmed', b='label_flagged', c='label_flagged', d='confirmed',
                                      e='span_flagged', f='span_flagged'))
        self.assertEqual(summary['statuses'], dict(confirmed=2, label_flagged=2, span_flagged=2))
        self.assertEqual(summary['reviewers_per_document'], {2: 6})
        self.assertAlmostEqual(summary['reviewer_pair_category_agreement'], 4 / 6)
        self.assertAlmostEqual(summary['original_vs_reviewers_category_agreement'], 4 / 6)
        added = {l['id']: l['prereview']['added_spans'] for l in out}
        self.assertEqual(added['f'], [[3, 16, 'phone', False]])

    def test_labels_are_never_changed_and_unreviewed_pass_through(self):
        labels = [label('a'), label('z')]
        out, _ = prereview.apply(labels, [rev('a', 'R1', 'reject', 'semantic_low_quality'), rev('a', 'R2', 'reject', 'semantic_low_quality')])
        self.assertEqual((out[0]['label'], out[0]['label_source'], out[0]['reject_reason']), ('keep', 'agent_prelabel:A', None))
        self.assertEqual(out[0]['prereview']['status'], 'label_flagged')
        self.assertNotIn('prereview', out[1])
        self.assertEqual(labels[0].get('prereview'), None)                                          # input not mutated

    def test_validation(self):
        with self.assertRaises(ValueError):
            prereview.apply([label('a')], [rev('zzz', 'R1')])
        with self.assertRaises(ValueError):
            prereview.apply([label('a')], [rev('a', 'R1'), rev('a', 'R1')])

    def test_review_sheet_surfaces_flags_first_among_non_rule_reasons(self):
        text = seed_gold.prose(random.Random(1), 300)
        flagged = dict(label('x'), text=text, source_kind='real_reviewed')
        flagged['prereview'] = dict(status='label_flagged', votes=['keep', 'reject'], added_spans=[], missing_spans=[], reviewers=['R1', 'R2'])
        ok = dict(label('y'), text=text, source_kind='real_reviewed')
        ok['prereview'] = dict(status='confirmed', votes=['keep', 'keep'], added_spans=[], missing_spans=[], reviewers=['R1', 'R2'])
        sheet, n = review_sheet.build([ok, flagged], controls=0)
        self.assertEqual(n, 1)
        self.assertIn('pre-review label_flagged', sheet)
        self.assertIn('## x', sheet)
        self.assertNotIn('## y', sheet)


if __name__ == '__main__':
    unittest.main()
