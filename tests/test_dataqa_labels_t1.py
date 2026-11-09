import json
import random
import tempfile
import unicodedata
import unittest
from pathlib import Path

from dataqa import gold, make_labels, review_sheet, seed_gold, semantic_probe, t1_stats


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


class DerivedTagsAndReviewTest(unittest.TestCase):
    def test_ambiguous_spans_imply_tags_regardless_of_what_the_labeller_wrote(self):
        text = '문의 02-000-1234 또는 press@news.example.kr 그리고 firstname.lastname@example.org 와 word@example.com'
        pii = [dict(needle='02-000-1234', type='phone', ambiguous=True),
               dict(needle='press@news.example.kr', type='email', ambiguous=True),
               dict(needle='firstname.lastname@example.org', type='email', ambiguous=True),
               dict(needle='word@example.com', type='email', ambiguous=False)]
        rec = make_labels.build_agent_record('x', doc('x', text), dict(label='keep', pii=pii, tags=['news']))
        self.assertEqual(rec['tags'], ['news', 'org_contact_email', 'org_contact_phone', 'upstream_email_placeholder'])
        personal = make_labels.build_agent_record('y', doc('y', text), dict(label='keep', pii=pii[3:]))
        self.assertEqual(personal['tags'], [])

    LABELS = [dict(id=f'r{i}', label=l, reject_reason=r, pii_spans=[], tags=[], label_source='agent_prelabel:A')
              for i, (l, r) in enumerate([('keep', None), ('reject', 'semantic_low_quality'), ('keep', None),
                                          ('reject', 'short'), ('keep', None), ('keep', None)])]

    def test_apply_review_marks_sources_and_reports_agreement(self):
        review = dict(reviewer='jun', reviewed=['r0', 'r1', 'r2', 'r3'], random_controls=['r0', 'r2'],
                      corrections={'r1': dict(label='keep', reject_reason=None)})
        out, rep = make_labels.apply_review(self.LABELS, review)
        by = {r['id']: r for r in out}
        self.assertEqual(by['r1']['label'], 'keep')
        self.assertEqual(by['r1']['label_source'], 'human_review:jun:corrected')
        self.assertEqual(by['r0']['label_source'], 'human_review:jun:agreed')
        self.assertEqual(by['r5']['label_source'], 'agent_prelabel:A')            # unreviewed stays untouched
        self.assertEqual((rep['reviewed'], rep['corrected']), (4, 1))
        self.assertAlmostEqual(rep['agreement_all_reviewed'], 0.75)
        self.assertEqual(rep['agreement_random_controls'], 1.0)
        self.assertIn('biased low', rep['note'])
        self.assertEqual(self.LABELS[1]['label'], 'reject')                        # input not mutated

    def test_apply_review_validates_and_flips_ambiguous(self):
        with self.assertRaises(ValueError):
            make_labels.apply_review(self.LABELS, dict(reviewed=['nope']))
        with self.assertRaises(ValueError):
            make_labels.apply_review(self.LABELS, dict(reviewed=['r0'], corrections={'r1': dict(label='keep')}))
        with self.assertRaises(ValueError):
            make_labels.apply_review(self.LABELS, dict(reviewed=['r0'], random_controls=['r1']))
        with self.assertRaises(ValueError):                                         # reject needs a reason
            make_labels.apply_review(self.LABELS, dict(reviewed=['r0'], corrections={'r0': dict(label='reject')}))
        out, _ = make_labels.apply_review(self.LABELS, dict(reviewed=['r0'], corrections={'r0': dict(label='keep', ambiguous=True)}))
        self.assertTrue(out[0]['ambiguous'])

    def test_cohen_kappa(self):
        self.assertAlmostEqual(make_labels.cohen_kappa([('a', 'a'), ('b', 'b')]), 1.0)
        self.assertAlmostEqual(make_labels.cohen_kappa([('a', 'b'), ('b', 'a')]), -1.0)
        self.assertIsNone(make_labels.cohen_kappa([('a', 'a'), ('a', 'a')]))
        self.assertIsNone(make_labels.cohen_kappa([]))


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
            base + ' mailto:firstname.lastname@example.org',
            base + ' contact real.person@naver.com +82-10-0000-1234 and word@example.com',
            'short',
        ]
        c, lengths = t1_stats.scan_group(texts, ['<|eos|>', '<|pad|>'])
        self.assertEqual(c['docs'], 7)
        self.assertEqual(c['not_nfc'], 1)
        self.assertEqual(c['conjoining_jamo'], 1)
        self.assertEqual(c['nfd_rejected'], 1)
        self.assertEqual((c['eos_literal'], c['special_literal_any']), (1, 1))
        self.assertEqual(c['email_placeholder'], 3)   # org placeholder x2 + example.com
        self.assertEqual(c['pii_email'], 1)              # only the naver.com address; reserved domains are placeholders
        self.assertEqual(c['pii_opt_in_differs'], 1)      # +82 number only the opt-in policy masks
        self.assertEqual(c['ko_reject_short'], 1)
        self.assertEqual(len(lengths), 7)

    def test_render_markdown_has_zero_cells_and_caveats(self):
        c, ls = t1_stats.scan_group([seed_gold.prose(random.Random(5), 300)], [])
        summary = t1_stats.summarize([c, c], ls * 2, 'fineweb2_ko', 25)
        md = t1_stats.render_markdown(dict(seed=1, unit='row group', caveats=['caveat-one'], sources=[summary]))
        self.assertIn('| 필터 통과 (기본 정책) | 100.00%', md)
        self.assertIn('| NFD 때문에 거절 (NFC면 통과) | 0 |', md)
        self.assertIn('- caveat-one', md)

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


class ReviewSheetTest(unittest.TestCase):
    def test_priorities_and_controls(self):
        recs = [r for r in gold.loads(Path(__file__).resolve().parents[1].joinpath('dataqa/gold/seed_v1.jsonl').read_text(encoding='utf-8'))]
        text, n = review_sheet.build(recs, controls=3, seed=1)
        self.assertEqual(text.count('\n## '), n)
        first = text.split('\n## ')[1]
        self.assertIn('rules', first.split('\n')[1])            # rule-vs-label disagreements come first
        self.assertEqual(text.count('random control'), 3)
        ids = [line.split()[0] for line in text.split('\n## ')[1:]]
        self.assertEqual(len(ids), len(set(ids)))                  # each document listed once
        self.assertEqual(review_sheet.mask_digits('010-1234'), '###-####')


class SemanticProbeTest(unittest.TestCase):
    def test_auc_known_values(self):
        self.assertEqual(semantic_probe.auc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertEqual(semantic_probe.auc([4, 3, 2, 1], [0, 0, 1, 1]), 0.0)
        self.assertEqual(semantic_probe.auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)
        self.assertIsNone(semantic_probe.auc([1, 2], [1, 1]))

    def test_probe_detects_an_obvious_signal_and_excludes_rule_rejects(self):
        rng = random.Random(3)
        recs = []
        for i in range(14):
            recs.append(dict(id=f'k{i}', text=seed_gold.prose(rng, 400), label='keep', reject_reason=None, source_kind='synthetic'))
            junk = '\n'.join(f'product{j} {rng.randint(1000, 9999)}원 buy now' for j in range(30))
            recs.append(dict(id=f'j{i}', text=junk, label='reject', reject_reason='semantic_low_quality', source_kind='synthetic'))
        recs.append(dict(id='s', text='short', label='reject', reject_reason='short', source_kind='synthetic'))
        recs.append(dict(id='a', text='amb', label='keep', reject_reason=None, ambiguous=True, source_kind='synthetic'))
        out = semantic_probe.probe(recs, boots=40)
        self.assertEqual((out['n'], out['semantic_low_quality'], out['keep']), (28, 14, 14))     # short + ambiguous excluded
        self.assertGreater(out['per_feature'][out['best_single_feature']]['auc'], 0.95)
        self.assertGreater(out['loo_logistic']['l2=10.0']['auc'], 0.9)
        self.assertEqual(set(out['per_feature']), set(semantic_probe.NAMES))


if __name__ == '__main__':
    unittest.main()
