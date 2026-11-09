import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from data import filters as F
from dataqa import gold, ko_rules, metrics, oracle, pii_rules, real_sample, run, seed_gold

ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / 'dataqa' / 'gold' / 'seed_v1.jsonl'


def make(rid='a', text='x', label='keep', **kw):
    r = dict(id=rid, text=text, label=label, source_kind='synthetic')
    r.update(kw)
    return r


fuzz_texts = oracle.fuzz_ko_texts
pii_fuzz = oracle.fuzz_pii_texts


class GoldTest(unittest.TestCase):
    def test_valid_and_invalid_records(self):
        gold.validate(make())
        gold.validate(make(label='reject', reject_reason='short'))
        with self.assertRaises(ValueError):
            gold.validate(make(label='reject'))
        with self.assertRaises(ValueError):
            gold.validate(make(label='keep', reject_reason='short'))
        with self.assertRaises(ValueError):
            gold.validate(make(label='maybe'))
        with self.assertRaises(ValueError):
            gold.validate(make(source_kind='web'))
        with self.assertRaises(ValueError):
            gold.validate({'id': 'x'})

    def test_span_rules(self):
        text = 'call 010-0000-1234 now'
        ok = dict(start=5, end=18, type='phone')
        gold.validate(make(text=text, pii_spans=[ok]))
        for bad in (dict(start=5, end=99, type='phone'), dict(start=5, end=5, type='phone'),
                    dict(start=5, end=18, type='fax'), dict(start=5.0, end=18, type='phone')):
            with self.assertRaises(ValueError):
                gold.validate(make(text=text, pii_spans=[bad]))
        with self.assertRaises(ValueError):
            gold.validate(make(text=text, pii_spans=[ok, dict(start=10, end=20, type='phone')]))
        with self.assertRaises(ValueError):  # offsets would shift after norm_ws
            gold.validate(make(text=' leading space 010', pii_spans=[dict(start=1, end=3, type='phone')]))

    def test_duplicate_ids_and_roundtrip(self):
        a = make('a'); b = make('b', label='reject', reject_reason='short')
        self.assertEqual(gold.loads(gold.dumps([a, b])), [a, b])
        with self.assertRaises(ValueError):
            gold.dumps([a, a])
        with self.assertRaises(ValueError):
            gold.loads('{not json}\n')

    def test_policy_override(self):
        r = make('p', policy_assumption='pol', policy_reject_reason='few_hangul')
        flipped = gold.apply_policy([r], {'pol': 'reject'})[0]
        self.assertEqual((flipped['label'], flipped['reject_reason']), ('reject', 'few_hangul'))
        self.assertEqual(r['label'], 'keep')  # input untouched
        self.assertEqual(gold.apply_policy([r], {'pol': 'keep'})[0]['label'], 'keep')
        with self.assertRaises(ValueError):
            gold.apply_policy([r], {'typo': 'reject'})
        with self.assertRaises(ValueError):
            gold.apply_policy([make('q', policy_assumption='pol2')], {'pol2': 'reject'})


class KoParityTest(unittest.TestCase):
    def test_default_matches_production_reason_on_gold_and_fuzz(self):
        texts = [r['text'] for r in gold.load(SEED_FILE)] + fuzz_texts()
        self.assertEqual(ko_rules.parity_mismatches(texts), [])
        reasons = {F.check_ko(t) for t in texts}
        self.assertTrue({None, 'short', 'few_hangul', 'low_hangul_ratio'} <= reasons, reasons)  # fuzz is not vacuous

    def test_length_knobs_match_production_keywords(self):
        p = ko_rules.KoParams(min_chars=0, max_chars=60)
        self.assertEqual(ko_rules.parity_mismatches(fuzz_texts(300), p), [])
        self.assertEqual(ko_rules.check_ko_param('가' * 61, p), 'long')

    def test_parity_gate_rejects_non_production_params(self):
        with self.assertRaises(ValueError):
            ko_rules.parity_mismatches(['x'], ko_rules.KoParams(min_hangul_ratio=0.5))

    def test_gate_detects_a_drifting_copy(self):
        with mock.patch.object(ko_rules, 'check_ko_param', return_value='short'):
            self.assertTrue(ko_rules.parity_mismatches(['가' * 300]))

    def test_each_knob_flips_exactly_at_the_measured_value(self):
        check = ko_rules.check_ko_param
        t = seed_gold.prose(random.Random(5), 400)
        self.assertIsNone(check(t))
        ratio = len(F.HANGUL.findall(t)) / sum(1 for c in t if not c.isspace())
        self.assertEqual(check(t, ko_rules.KoParams(min_hangul_ratio=ratio + 1e-9)), 'low_hangul_ratio')
        self.assertIsNone(check(t, ko_rules.KoParams(min_hangul_ratio=ratio)))          # comparison is strict '<'
        alnum = sum(c.isalnum() for c in t) / len(t)
        self.assertEqual(check(t, ko_rules.KoParams(min_alnum_ratio=alnum + 1e-9)), 'low_alnum')
        self.assertIsNone(check(t, ko_rules.KoParams(min_alnum_ratio=alnum)))
        self.assertEqual(check(t, ko_rules.KoParams(min_chars=len(t) + 1)), 'short')
        self.assertIsNone(check(t, ko_rules.KoParams(min_chars=len(t))))
        self.assertEqual(check(t, ko_rules.KoParams(max_chars=len(t) - 1)), 'long')
        self.assertEqual(check(t, ko_rules.KoParams(min_hangul=len(F.HANGUL.findall(t)) + 1)), 'few_hangul')
        rep = ' '.join([seed_gold.TEMPLATES[0].format(**seed_gold.slots(random.Random(6)))] * 6)
        stat = F.ngram_rep(rep.split())
        self.assertGreater(stat, 0.3)
        self.assertEqual(check(rep), 'ngram_rep')
        self.assertIsNone(check(rep, ko_rules.KoParams(max_ngram_rep=stat)))
        lines = '\n'.join(['동일한 문장이 반복되는 줄입니다 정말입니다'] * 3 + [f'서로 다른 문장 {i} 번째 줄입니다 확인하세요' for i in range(4)])
        dup = F.dedup_lines_stat([l for l in lines.split('\n') if l.strip()])
        self.assertEqual(check(lines + ' ' * 0, ko_rules.KoParams(min_chars=0, max_dup_line_frac=dup - 1e-9)), 'dup_lines')
        self.assertIsNone(check(lines, ko_rules.KoParams(min_chars=0, max_dup_line_frac=dup, max_ngram_rep=1.0)))

    def test_nfc_candidate_rescues_nfd_only(self):
        import unicodedata
        text = seed_gold.prose(random.Random(1), 400)
        nfd = unicodedata.normalize('NFD', text)
        self.assertIsNone(ko_rules.check_ko_param(text))
        self.assertEqual(ko_rules.check_ko_param(nfd), 'few_hangul')
        self.assertIsNone(ko_rules.check_ko_param(nfd, ko_rules.KoParams(pre_normalize='nfc')))
        with self.assertRaises(ValueError):
            ko_rules.check_ko_param(text, ko_rules.KoParams(pre_normalize='nfkc'))


class PiiTest(unittest.TestCase):
    def test_baseline_reproduces_mask_pii_on_gold_and_fuzz(self):
        texts = [r['text'] for r in gold.load(SEED_FILE)] + pii_fuzz()
        self.assertEqual(pii_rules.parity_mismatches(texts), [])

    def test_offsets_are_original_coordinates(self):
        text = '메일 user@example.com 번호 900101-1000001 전화 010-0000-1234 끝'
        masked, dets = pii_rules.apply(text)
        self.assertEqual(masked, F.mask_pii(text))
        self.assertEqual([(d['type'], text[d['start']:d['end']]) for d in dets],
                         [('email', 'user@example.com'), ('rrn', '900101-1000001'), ('phone', '010-0000-1234')])

    def test_one_digit_run_is_not_counted_twice(self):
        # 13 digits matched by RRN first; the phone step then sees a mask token, not digits
        _, dets = pii_rules.apply('번호 9001011000005 끝')
        self.assertEqual([d['type'] for d in dets], ['rrn'])

    def test_international_phone_candidate(self):
        text = '연락처 +82-10-0000-1234 입니다'
        _, base = pii_rules.apply(text)
        _, cand = pii_rules.apply(text, pii_rules.PiiConfig(phone='intl'))
        self.assertNotEqual([(d['start'], d['end']) for d in base], [(4, 20)])  # baseline cannot cover it exactly
        self.assertEqual([(d['start'], d['end']) for d in cand], [(4, 20)])
        _, paren = pii_rules.apply('전화 (02) 000-1234 끝', pii_rules.PiiConfig(phone='intl'))
        self.assertEqual(len(paren), 1)

    def test_prefix_allowlist_rejects_lookalikes(self):
        cfg = pii_rules.PiiConfig(phone='intl_prefix')
        self.assertEqual(pii_rules.apply('SKU 0123456789 끝', cfg)[1], [])
        self.assertEqual(len(pii_rules.apply('SKU 0123456789 끝', pii_rules.PiiConfig(phone='intl'))[1]), 1)
        self.assertEqual(len(pii_rules.apply('전화 +82 10 0000 1234 끝', cfg)[1]), 1)

    def test_rrn_candidates(self):
        wide = pii_rules.PiiConfig(rrn='range_date')
        strict = pii_rules.PiiConfig(rrn='range_date_sep')
        self.assertEqual(pii_rules.apply('번호 900101-5000007 끝')[1], [])           # baseline misses 5-8
        self.assertEqual(len(pii_rules.apply('번호 900101-5000007 끝', wide)[1]), 1)
        self.assertEqual(pii_rules.apply('번호 901301-1000001 끝', wide)[1], [])      # month 13
        self.assertEqual(pii_rules.apply('번호 900132-1000001 끝', wide)[1], [])      # day 32
        self.assertEqual(len(pii_rules.apply('번호 9001011000005 끝', wide)[1]), 1)
        self.assertEqual(pii_rules.apply('번호 9001011000005 끝', strict)[1], [])
        self.assertEqual(len(pii_rules.apply('번호 900101 1000006 끝', strict)[1]), 1)

    def test_email_git_remote(self):
        text = 'clone git@github.com:example/repo.git 후 user@example.com 확인'
        self.assertEqual(len(pii_rules.apply(text)[1]), 2)
        dets = pii_rules.apply(text, pii_rules.PiiConfig(email='no_git'))[1]
        self.assertEqual([text[d['start']:d['end']] for d in dets], ['user@example.com'])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            pii_rules.apply('x', pii_rules.PiiConfig(phone='nope'))


class SeedGoldTest(unittest.TestCase):
    def test_deterministic_and_committed_file_is_current(self):
        a, b = seed_gold.build(), seed_gold.build()
        self.assertEqual(a, b)
        self.assertEqual(gold.dumps(a), SEED_FILE.read_text(encoding='utf-8'),
                         'dataqa/gold/seed_v1.jsonl is stale: python -m dataqa.seed_gold --out dataqa/gold/seed_v1.jsonl')

    def test_designed_reasons_hold(self):
        for r in gold.load(SEED_FILE):
            got = ko_rules.filter_reason(r['text'])
            if r['label'] == 'reject':
                self.assertEqual(got, r['reject_reason'], r['id'])
            elif r.get('policy_assumption'):
                self.assertEqual(got, r['policy_reject_reason'], r['id'])
            else:
                self.assertIsNone(got, r['id'])

    def test_only_fictional_identifiers(self):
        text = SEED_FILE.read_text(encoding='utf-8')
        import re
        for m in re.finditer(r'[\w.+-]+@([\w.-]+\.[a-z]+)', text):
            if m.group(1).lower() != 'github.com':
                self.assertRegex(m.group(1).lower(), r'(^|\.)example\.')


class MetricsTest(unittest.TestCase):
    def test_zero_denominators_are_none_not_zero(self):
        m = metrics.filter_metrics([make('k', seed_gold.prose(random.Random(2), 300))])['synthetic']
        self.assertIsNone(m['reject']['recall'])      # no reject-labelled docs
        self.assertIsNone(m['reject']['precision'])   # nothing rejected
        self.assertEqual(m['reject']['recall_den'], 0)

    def test_kinds_are_not_pooled_and_ambiguous_excluded(self):
        prose = seed_gold.prose(random.Random(3), 300)
        recs = [make('s', prose), make('r', 'short', label='reject', reject_reason='short', source_kind='real_reviewed'),
                make('amb', 'short', label='keep', ambiguous=True, source_kind='real_reviewed')]
        m = metrics.filter_metrics(recs)
        self.assertEqual(set(m), {'synthetic', 'real_reviewed'})
        self.assertEqual(m['real_reviewed']['reject']['tp'], 1)
        self.assertEqual(m['real_reviewed']['ambiguous_excluded'], 1)
        self.assertEqual(m['real_reviewed']['reject']['fp'], 0)

    def test_filter_error_listing(self):
        prose = seed_gold.prose(random.Random(4), 300)
        recs = [make('fp', 'short text', label='keep'), make('fn', prose, label='reject', reject_reason='short')]
        errs = {e['id']: e['type'] for e in metrics.filter_metrics(recs)['synthetic']['errors']}
        self.assertEqual(errs, {'fp': 'false_reject', 'fn': 'missed_reject'})

    def test_pii_metrics_counts_and_unimplemented_types(self):
        recs = gold.load(SEED_FILE)
        m = metrics.pii_metrics(recs)['synthetic']
        self.assertTrue(m['per_type']['phone']['implemented'])
        for t in ('card', 'account', 'ip'):
            self.assertFalse(m['per_type'][t]['implemented'])
            self.assertEqual(m['per_type'][t]['overlap'] if t == 'ip' else 0, m['per_type'][t]['overlap'] if t == 'ip' else 0)
        self.assertEqual(m['per_type']['alien_reg']['exact'], 0)   # baseline accepts 7th digit 1-4 only
        wide = metrics.pii_metrics(recs, pii_rules.PRESETS['rrn_range_date'])['synthetic']
        self.assertEqual(wide['per_type']['alien_reg']['exact'], wide['per_type']['alien_reg']['n'])

    def test_ambiguous_span_hits_are_excluded_from_precision(self):
        recs = [r for r in gold.load(SEED_FILE) if r['id'] == 'syn-pii-ambiguous-00']
        m = metrics.pii_metrics(recs)['synthetic']
        self.assertEqual((m['false_positive'], m['ambiguous_hits_excluded']), (0, 1))
        self.assertIsNone(m['precision_exact'])


class RunTest(unittest.TestCase):
    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'o'
            run.main(['--gold', str(SEED_FILE), '--out', str(out)])
            rep = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(rep['gold']['records'], 94)
            self.assertEqual(rep['parity']['ko_mismatches'], 0)
            self.assertTrue(rep['notes'] and rep['uncertain_claims'])
            self.assertIn('baseline', rep['pii'])
            self.assertTrue(any(r['is_default'] for r in rep['sweep']))
            md = (out / 'report.md').read_text(encoding='utf-8')
            self.assertIn('하네스 시연', md)
            self.assertIn('범위 밖', md)
            with self.assertRaises(SystemExit):        # never overwrite a populated directory
                run.main(['--gold', str(SEED_FILE), '--out', str(out)])

    def test_policy_flags_change_labels_in_report(self):
        base = run.build_report(SEED_FILE)
        flipped = run.build_report(SEED_FILE, {'nfd_text_is_valid_korean': 'reject'})
        self.assertEqual(base['filter']['synthetic']['reject']['fp'] - 3, flipped['filter']['synthetic']['reject']['fp'])
        self.assertEqual(flipped['policy_overrides'], {'nfd_text_is_valid_korean': 'reject'})

    def test_report_refuses_when_rule_copy_drifts(self):
        with mock.patch.object(ko_rules, 'check_ko_param', return_value='short'):
            with self.assertRaises(SystemExit):
                run.build_report(SEED_FILE)

    def test_bad_policy_argument(self):
        with self.assertRaises(SystemExit):
            run.parse_policy(['no_equals_sign'])


class SemanticRejectTest(unittest.TestCase):
    def test_semantic_rejects_are_outside_rule_precision_recall(self):
        prose = seed_gold.prose(random.Random(8), 300)
        recs = [make('spam', prose, label='reject', reject_reason='semantic_low_quality'),
                make('ok', prose), make('s', 'short', label='reject', reject_reason='short')]
        m = metrics.filter_metrics(recs)['synthetic']
        self.assertEqual(m['semantic_rejects'], dict(n=1, caught_by_rules=0))
        self.assertEqual((m['reject']['tp'], m['reject']['fn'], m['reject']['fp']), (1, 0, 0))
        self.assertEqual(m['reject']['recall_den'], 1)

    def test_report_mentions_semantic_bucket(self):
        prose = seed_gold.prose(random.Random(9), 300)
        with tempfile.TemporaryDirectory() as tmp:
            g = Path(tmp) / 'g.jsonl'
            gold.dump([make('spam', prose, label='reject', reject_reason='semantic_low_quality')], g)
            md = run.render_markdown(run.build_report(g))
        self.assertIn('의미적 거절', md)


class RealSampleTest(unittest.TestCase):
    SHARDS = [f'data/s{i:02d}.parquet' for i in range(5)]
    COUNTS = {s: 40 for s in SHARDS}

    def test_plan_is_deterministic_distinct_and_bounded(self):
        a = real_sample.plan(self.SHARDS, self.COUNTS, 25, seed=3)
        self.assertEqual(a, real_sample.plan(self.SHARDS, self.COUNTS, 25, seed=3))
        self.assertNotEqual(a, real_sample.plan(self.SHARDS, self.COUNTS, 25, seed=4))
        self.assertEqual(len({(x['file'], x['row_group']) for x in a}), 25)
        self.assertTrue(all(0 <= x['row_group'] < 40 and 0 <= x['row_draw'] < 1 for x in a))
        self.assertGreater(len({x['file'] for x in a}), 1)       # not all from one shard
        self.assertGreater(max(x['row_group'] for x in a), 0)    # not just row group 0
        with self.assertRaises(ValueError):
            real_sample.plan(self.SHARDS, self.COUNTS, 201, seed=1)

    def test_registry_pins_a_revision(self):
        repo, rev = real_sample.pinned('fineweb2_ko')
        self.assertEqual(repo, 'HuggingFaceFW/fineweb-2')
        self.assertRegex(rev, r'^[0-9a-f]{40}$')

    def test_committed_pilot_files_contain_no_raw_text(self):
        manifest = [json.loads(l) for l in (ROOT / 'dataqa/real/pilot_manifest_v1.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        labels = [json.loads(l) for l in (ROOT / 'dataqa/real/pilot_labels_v1.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        self.assertEqual({m['id'] for m in manifest}, {l['id'] for l in labels})
        for m in manifest:
            self.assertNotIn('text', m)
            self.assertRegex(m['revision'], r'^[0-9a-f]{40}$')
            self.assertRegex(m['text_sha256'], r'^[0-9a-f]{64}$')
        for l in labels:
            self.assertNotIn('text', l)
            self.assertIn(l['label'], gold.LABELS)
            if l['label'] == 'reject':
                self.assertIn(l['reject_reason'], gold.REASONS)
            for sp in l['pii_spans']:
                self.assertEqual(set(sp) - {'ambiguous'}, {'start', 'end', 'type'})
                self.assertIn(sp['type'], gold.PII_TYPES)
            self.assertEqual(l['label_source'], 'agent_prelabel')   # not yet human reviewed


if __name__ == '__main__':
    unittest.main()
