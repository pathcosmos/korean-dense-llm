"""Direct tests of data/filters.py (check_ko, mask_pii, norm_ws) and of its opt-in policies.

Defaults are pinned by tests/test_filters_oracle.py; these tests cover behaviour that file cannot
express: threshold semantics, policy options, input validation.
"""
import random
import unicodedata
import unittest

from data import filters as F
from dataqa import seed_gold


def prose(seed=5, n=400):
    return seed_gold.prose(random.Random(seed), n)


class NormWsTest(unittest.TestCase):
    def test_strips_lines_collapses_blank_runs_and_cr(self):
        self.assertEqual(F.norm_ws('  a  \r\nb\t\n\n\n\n\nc  '), 'a\nb\n\nc')
        self.assertEqual(F.norm_ws('\n\n x \n\n'), 'x')

    def test_interior_spaces_and_indentation_of_inner_lines(self):
        self.assertEqual(F.norm_ws('a    b\n    c'), 'a    b\nc')   # leading indentation is lost by design


class CheckKoThresholdTest(unittest.TestCase):
    def test_each_threshold_flips_exactly_at_the_measured_value(self):
        t = prose()
        self.assertIsNone(F.check_ko(t))
        ratio = len(F.HANGUL.findall(t)) / sum(1 for c in t if not c.isspace())
        alnum = sum(c.isalnum() for c in t) / len(t)
        P = F.KoPolicy
        self.assertEqual(F.check_ko(t, policy=P(min_hangul_ratio=ratio + 1e-9)), 'low_hangul_ratio')
        self.assertIsNone(F.check_ko(t, policy=P(min_hangul_ratio=ratio)))      # strict '<'
        self.assertEqual(F.check_ko(t, policy=P(min_alnum_ratio=alnum + 1e-9)), 'low_alnum')
        self.assertIsNone(F.check_ko(t, policy=P(min_alnum_ratio=alnum)))
        self.assertEqual(F.check_ko(t, policy=P(min_chars=len(t) + 1)), 'short')
        self.assertIsNone(F.check_ko(t, policy=P(min_chars=len(t))))
        self.assertEqual(F.check_ko(t, policy=P(max_chars=len(t) - 1)), 'long')
        self.assertEqual(F.check_ko(t, policy=P(min_hangul=len(F.HANGUL.findall(t)) + 1)), 'few_hangul')

    def test_legacy_length_keywords_still_work_and_override_policy(self):
        t = 'x' * 10
        self.assertEqual(F.check_ko(t), 'short')
        self.assertEqual(F.check_ko(t, 0, 5), 'long')
        self.assertEqual(F.check_ko(t, min_chars=0, max_chars=5), 'long')
        self.assertEqual(F.check_ko(t, max_chars=5, policy=F.KoPolicy(min_chars=0)), 'long')

    def test_empty_and_whitespace(self):
        self.assertEqual(F.check_ko('', min_chars=0), 'empty')
        self.assertEqual(F.check_ko(' \n ' * 100, min_chars=0), 'empty')

    def test_repetition_checks(self):
        sentence = seed_gold.TEMPLATES[0].format(**seed_gold.slots(random.Random(6)))
        rep = ' '.join([sentence] * 6)
        stat = F.ngram_rep(rep.split())
        self.assertEqual(F.check_ko(rep), 'ngram_rep')
        self.assertIsNone(F.check_ko(rep, policy=F.KoPolicy(max_ngram_rep=stat)))
        spam = '\n'.join(['오늘만 특가 할인 이벤트에 참여하세요 지금 바로 확인하세요 지금 바로 신청하세요'] * 8)
        self.assertEqual(F.check_ko(spam), 'dup_lines')
        self.assertIsNone(F.check_ko(spam, policy=F.KoPolicy(max_dup_line_frac=1.0, max_ngram_rep=1.0)))

    def test_bad_encoding(self):
        t = prose()
        t = t[:100] + '�' + t[100:]
        self.assertEqual(F.check_ko(t), 'bad_encoding')
        self.assertIsNone(F.check_ko(t, policy=F.KoPolicy(max_bad_encoding_ratio=1.0)))

    def test_nfd_rejected_by_default_and_rescued_by_opt_in_nfc(self):
        t = prose()
        nfd = unicodedata.normalize('NFD', t)
        self.assertIsNone(F.check_ko(t))
        self.assertEqual(F.check_ko(nfd), 'few_hangul')
        self.assertIsNone(F.check_ko(nfd, policy=F.KoPolicy(pre_normalize='nfc')))
        with self.assertRaises(ValueError):
            F.check_ko(t, policy=F.KoPolicy(pre_normalize='nfkc'))


class MaskPiiTest(unittest.TestCase):
    def test_default_masks_email_rrn_phone_in_that_order(self):
        self.assertEqual(F.mask_pii('a@example.com 900101-1000001 010-0000-1234'), '<email> <id> <phone>')

    def test_default_known_limits_are_documented_behaviour(self):
        for unchanged in ('+82-10-0000-1234', '900101-5000007'):
            self.assertEqual(F.mask_pii(unchanged), unchanged)
        for overmasked in ('0123456789', '2411221234567'):
            self.assertNotEqual(F.mask_pii(overmasked), overmasked)

    def test_default_policy_object_equals_no_policy(self):
        text = 'x git@github.com:o/r.git 02-000-1234 900101-1000001 y'
        self.assertEqual(F.mask_pii(text, F.PiiPolicy()), F.mask_pii(text))

    def test_intl_phone_option(self):
        p = F.PiiPolicy(phone='intl')
        self.assertEqual(F.mask_pii('call +82-10-0000-1234 now', p), 'call <phone> now')
        self.assertEqual(F.mask_pii('call +82 10 0000 1234 now', p), 'call <phone> now')
        self.assertEqual(F.mask_pii('call (02) 000-1234 now', p), 'call <phone> now')
        self.assertEqual(F.mask_pii('010-0000-1234', p), F.mask_pii('010-0000-1234'))

    def test_rrn_date_checked_option(self):
        p = F.PiiPolicy(rrn='date_checked')
        self.assertEqual(F.mask_pii('900101-1000001', p), '<id>')
        self.assertEqual(F.mask_pii('901301-1000001', p), '901301-1000001')   # month 13
        self.assertEqual(F.mask_pii('900132-1000001', p), '900132-1000001')   # day 32
        self.assertEqual(F.mask_pii('900101-1000001'), '<id>')                 # default unchanged

    def test_email_no_git_option(self):
        text = 'git@github.com:example/repo.git and user@example.com'
        self.assertEqual(F.mask_pii(text), '<email>:example/repo.git and <email>')
        self.assertEqual(F.mask_pii(text, F.PiiPolicy(email='no_git')), 'git@github.com:example/repo.git and <email>')

    def test_find_pii_rejects_unknown_modes(self):
        with self.assertRaises(ValueError):
            F.find_pii('x', 'phone', 'nope')
        with self.assertRaises(ValueError):
            F.find_pii('x', 'card', 'baseline')


class ImportContractTest(unittest.TestCase):
    def test_star_import_does_not_leak_private_helpers(self):
        ns = {}
        exec('from data.filters import *', ns)
        for leaked in ('dataclass', 'asdict', 'replace', 'unicodedata'):
            self.assertNotIn(leaked, ns)
        for needed in ('check_ko', 'mask_pii', 'norm_ws', 'exact_hash', 'KoPolicy', 'PiiPolicy', 'KO_DEFAULT'):
            self.assertIn(needed, ns)

    def test_filters_stays_stdlib_plus_xxhash(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path(F.__file__).read_text(encoding='utf-8'))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or '').split('.')[0])
        self.assertLessEqual(mods, {'re', 'unicodedata', 'dataclasses', 'xxhash', 'collections'})


if __name__ == '__main__':
    unittest.main()
