import json
import random
import tempfile
import unicodedata
import unittest
from pathlib import Path

from dataqa import make_labels, seed_gold, t1_stats


def doc(rid, text):
    return dict(id=rid, text=text)


def spec(agent, docs):
    return dict(agent=agent, docs=docs)


class NeedleTest(unittest.TestCase):
    def test_all_and_first_occurrence_offsets(self):
        text = '연락 010-0000-1234 그리고 010-0000-1234 끝'
        both = make_labels.spans_for(text, [dict(needle='010-0000-1234', type='phone')])
        self.assertEqual([(s['start'], s['end']) for s in both], [(3, 16), (21, 34)])
        first = make_labels.spans_for(text, [dict(needle='010-0000-1234', type='phone', occurrence='first', ambiguous=True)])
        self.assertEqual(first, [dict(start=3, end=16, type='phone', ambiguous=True)])

    def test_missing_or_empty_needle_raises_without_echoing_it(self):
        with self.assertRaises(ValueError) as cm:
            make_labels.spans_for('abc', [dict(needle='secret-value', type='phone')])
        self.assertNotIn('secret-value', str(cm.exception))
        with self.assertRaises(ValueError):
            make_labels.spans_for('abc', [dict(needle='', type='phone')])


class ConsensusAndKappaTest(unittest.TestCase):
    def build(self, specs, texts):
        with tempfile.TemporaryDirectory() as tmp:
            sheet = Path(tmp) / 's.jsonl'
            sheet.write_text('\n'.join(json.dumps(doc(k, v)) for k, v in texts.items()), encoding='utf-8')
            paths = []
            for i, sp in enumerate(specs):
                p = Path(tmp) / f'spec{i}.json'
                p.write_text(json.dumps(sp), encoding='utf-8')
                paths.append(p)
            return make_labels.build(sheet, paths)

    def test_single_labeller_passes_through_and_unlabelled_are_reported(self):
        text = seed_gold.prose(random.Random(1), 300)
        final, per_agent, agreement, missing = self.build(
            [spec('A', {'x': dict(label='keep', tags=['news', 'news'])})], {'x': text, 'y': text})
        self.assertEqual(final[0]['label_source'], 'agent_prelabel:A')
        self.assertEqual(final[0]['tags'], ['news'])
        self.assertEqual((missing, agreement['unlabelled'], agreement['overlap_documents']), (['y'], 1, 0))

    def test_majority_tie_and_unknown_ids(self):
        text = seed_gold.prose(random.Random(2), 300)
        specs = [spec(a, {'x': dict(label=l, reject_reason=r)}) for a, l, r in
                 (('A', 'keep', None), ('B', 'reject', 'semantic_low_quality'), ('C', 'reject', 'semantic_low_quality'))]
        final, *_ = self.build(specs, {'x': text})
        self.assertEqual((final[0]['label'], final[0]['reject_reason']), ('reject', 'semantic_low_quality'))
        self.assertNotIn('ambiguous', final[0])
        tie, *_ = self.build(specs[:2], {'x': text})
        self.assertTrue(tie[0]['ambiguous'])
        with self.assertRaises(ValueError):
            self.build([spec('A', {'zzz': dict(label='keep')})], {'x': text})

    def test_span_needs_half_of_the_votes(self):
        text = '문의 010-0000-1234 입니다. ' + seed_gold.prose(random.Random(3), 250)
        pii = [dict(needle='010-0000-1234', type='phone')]
        specs = [spec('A', {'x': dict(label='keep', pii=pii)}), spec('B', {'x': dict(label='keep', pii=pii)}),
                 spec('C', {'x': dict(label='keep')})]
        final, *_ = self.build(specs, {'x': text})
        self.assertEqual(len(final[0]['pii_spans']), 1)
        minority, *_ = self.build([specs[0], specs[2], spec('D', {'x': dict(label='keep')})], {'x': text})
        self.assertEqual(minority[0]['pii_spans'], [])

    def test_fleiss_kappa_known_values(self):
        from collections import Counter
        cats = ['keep', 'reject', 'ambiguous']
        perfect = [Counter(keep=3), Counter(reject=3), Counter(keep=3), Counter(reject=3)]
        self.assertAlmostEqual(make_labels.fleiss_kappa(perfect, cats), 1.0)
        # textbook example (Fleiss 1971 style, 2 categories, 3 raters): kappa is 0 when agreement equals chance
        chance = [Counter(keep=2, reject=1), Counter(keep=1, reject=2)] * 2
        self.assertAlmostEqual(make_labels.fleiss_kappa(chance, cats), -0.3333333333, places=6)
        self.assertIsNone(make_labels.fleiss_kappa([Counter(keep=3), Counter(keep=3)], cats))   # no variation: undefined
        self.assertIsNone(make_labels.fleiss_kappa([Counter(keep=3), Counter(keep=2)], cats))   # unequal rater counts


class T1StatsTest(unittest.TestCase):
    def test_scan_group_counts_documented_conditions(self):
        base = seed_gold.prose(random.Random(4), 400)
        texts = [
            base,
            unicodedata.normalize('NFD', base),
            base + ' <|eos|> ' + base[:50],
            base + ' contact firstname.lastname@example.org',
            base + ' contact real.person@example.com +82-10-0000-1234',
            'short',
        ]
        c, lengths = t1_stats.scan_group(texts, ['<|eos|>', '<|pad|>'])
        self.assertEqual(c['docs'], 6)
        self.assertEqual(c['not_nfc'], 1)
        self.assertEqual(c['conjoining_jamo'], 1)
        self.assertEqual(c['nfd_rejected'], 1)
        self.assertEqual((c['eos_literal'], c['special_literal_any']), (1, 1))
        self.assertEqual(c['email_placeholder'], 1)
        self.assertEqual(c['pii_email'], 1)              # the placeholder is not counted as a real address
        self.assertEqual(c['pii_opt_in_extra'], 1)       # +82 number only the opt-in policy masks
        self.assertEqual(c['ko_reject_short'], 1)
        self.assertEqual(len(lengths), 6)

    def test_bootstrap_resamples_groups_and_is_deterministic(self):
        groups = [dict(docs=100, a=10), dict(docs=100, a=10), dict(docs=100, a=90)]
        r = t1_stats.bootstrap(groups, 'a', boots=500, seed=1)
        self.assertEqual((r['count'], r['docs']), (110, 300))
        self.assertAlmostEqual(r['rate'], 110 / 300)
        self.assertEqual(r, t1_stats.bootstrap(groups, 'a', boots=500, seed=1))
        self.assertLess(r['ci95'][0], r['rate'] < r['ci95'][1] and r['rate'] + 1e-9)
        self.assertGreater(r['ci95'][1] - r['ci95'][0], 0.1)       # few groups -> wide interval
        same = t1_stats.bootstrap([dict(docs=100, a=7)] * 5, 'a', boots=200, seed=1)
        self.assertEqual(same['ci95'], [0.07, 0.07])


if __name__ == '__main__':
    unittest.main()
