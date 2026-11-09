"""Opt-in FineWeb-2-style quality filter (data/quality_fw2.py). Heavy dependencies are optional:
tests that need datatrove/kiwipiepy skip when they are absent (CI installs only requirements-cpu.txt)."""
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data import clean
from data import quality_fw2 as Q
from data.filters import KO_DEFAULT, PII_DEFAULT

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'docs/evaluation/evidence/scripts/datatrove/fw2_kor_Hang.yml'
ORACLE = ROOT / 'tests/golden/fw2_oracle_v1.json'
PRESENT = Q.available()


class SpecTest(unittest.TestCase):
    """No heavy dependency needed."""

    def test_parse_spec(self):
        spec = Q.parse_spec(dict(engine='fw2', config=str(CONFIG)))
        self.assertEqual((spec['engine'], spec['language']), ('fw2', 'kor_Hang'))
        self.assertTrue(Path(spec['config']).is_absolute())
        for bad in (dict(engine='other', config='x'), dict(engine='fw2'), dict(engine='fw2', config='x', extra=1)):
            with self.assertRaises(ValueError):
                Q.parse_spec(bad)

    def test_policy_file_with_quality_parses_without_importing_datatrove(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'p.json'
            p.write_text(json.dumps({'quality': {'engine': 'fw2', 'config': str(CONFIG)}}))
            pol = clean.load_policy(str(p))
            self.assertEqual(pol['quality']['language'], 'kor_Hang')
            self.assertEqual(pol['ko'].as_dict(), KO_DEFAULT.as_dict())   # clean.py imports filters by bare name: a distinct class object

    def test_default_signature_shape_is_unchanged_by_the_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'a.parquet'
            pq.write_table(pa.table({'text': ['x']}), f)
            self.assertNotIn('quality', clean.signature('ko', str(f), {'ko': KO_DEFAULT, 'pii': PII_DEFAULT}))

    @unittest.skipIf(PRESENT, 'optional dependencies are installed')
    def test_missing_dependencies_fail_with_an_install_hint(self):
        with self.assertRaisesRegex(ImportError, 'pip install'):
            Q.require()
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'a.parquet'
            pq.write_table(pa.table({'text': ['x']}), f)
            with self.assertRaises(ImportError):
                clean.signature('ko', str(f), {'ko': KO_DEFAULT, 'pii': PII_DEFAULT, 'quality': Q.parse_spec(dict(engine='fw2', config=str(CONFIG)))})


@unittest.skipUnless(PRESENT, 'datatrove/kiwipiepy not installed (pip install -r requirements-fw2.txt)')
class Fw2FilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = Q.Fw2Quality(str(CONFIG))

    def test_matches_the_recorded_oracle_on_the_seed_gold(self):
        from dataqa import seed_gold
        from data.filters import norm_ws
        oracle = json.loads(ORACLE.read_text(encoding='utf-8'))
        env = Q.environment(dict(config=str(CONFIG)))
        if (env['datatrove'], env['kiwipiepy']) != (oracle['datatrove'], oracle['kiwipiepy']):
            self.skipTest(f"library versions differ from the oracle's ({oracle['datatrove']}, {oracle['kiwipiepy']})")
        self.assertEqual(env['config_sha256'], oracle['config_sha256'])
        got = {r['id']: self.q.check(r['text']) for r in seed_gold.build()}
        self.assertEqual(got, oracle['verdicts'])
        self.assertGreater(sum(v is not None for v in got.values()), 20)    # not vacuous
        self.assertGreater(sum(v is None for v in got.values()), 20)

    def test_reasons_are_prefixed_and_deterministic(self):
        from dataqa import seed_gold
        text = next(r['text'] for r in seed_gold.build() if r['id'] == 'syn-keep-list-00')
        first = self.q.check(text)
        self.assertTrue(first.startswith('fw2:gopher_rep:'), first)
        self.assertEqual(first, self.q.check(text))

    def test_bad_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'c.yml'
            bad.write_text('dup_line_frac: 0.3\n')
            with self.assertRaisesRegex(ValueError, 'lacks keys'):
                Q.Fw2Quality(str(bad))


@unittest.skipUnless(PRESENT, 'datatrove/kiwipiepy not installed (pip install -r requirements-fw2.txt)')
class CleanIntegrationTest(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        from dataqa import seed_gold
        by_id = {r['id']: r['text'] for r in seed_gold.build()}
        self.good = by_id['syn-keep-prose-00']
        self.rejected_by_fw2_only = by_id['syn-keep-list-00']       # passes check_ko, fails FW2 (top_3_gram)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / 'input.parquet'
        pq.write_table(pa.table({'text': [self.good, self.rejected_by_fw2_only]}), self.src)
        self.patch = patch.object(clean, 'ROOT', str(self.root))
        self.patch.start()
        self.out = self.root / 'clean/fx/input.parquet'
        self.spec = Q.parse_spec(dict(engine='fw2', config=str(CONFIG)))

    def tearDown(self):
        self.patch.stop()
        self._tmp.cleanup()

    def test_default_keeps_both_quality_policy_drops_one_and_counts_the_reason(self):
        default_policy = {'ko': KO_DEFAULT, 'pii': PII_DEFAULT}
        quality_policy = dict(default_policy, quality=self.spec)
        self.assertNotEqual(clean.signature('ko', str(self.src), default_policy)['id'],
                            clean.signature('ko', str(self.src), quality_policy)['id'])
        _, (counts, _) = clean.work(('fx', 'ko', str(self.src), quality_policy))
        self.assertEqual(counts['kept'], 1)
        self.assertEqual([k for k in counts if k.startswith('fw2:')], ['fw2:gopher_rep:top_3_gram'])
        stats = json.loads(Path(str(self.out) + '.stats.json').read_text())
        q = stats['signature']['quality']
        self.assertEqual(set(q), {'engine', 'config', 'language', 'config_sha256', 'datatrove', 'kiwipiepy', 'module_sha256'})
        # reuse with the same policy, refusal with the default policy
        self.assertEqual(clean.work(('fx', 'ko', str(self.src), quality_policy))[1][0], counts)
        with self.assertRaisesRegex(RuntimeError, 'signature differs'):
            clean.work(('fx', 'ko', str(self.src), default_policy))

    def test_changed_config_content_changes_the_signature(self):
        other = self.root / 'other.yml'
        other.write_text(CONFIG.read_text().replace('dup_line_frac: 0.374', 'dup_line_frac: 0.5'))
        a = clean.signature('ko', str(self.src), {'ko': KO_DEFAULT, 'pii': PII_DEFAULT, 'quality': self.spec})
        b = clean.signature('ko', str(self.src), {'ko': KO_DEFAULT, 'pii': PII_DEFAULT,
                                                  'quality': Q.parse_spec(dict(engine='fw2', config=str(other)))})
        self.assertNotEqual(a['id'], b['id'])

    def test_quality_policy_is_for_korean_only(self):
        with self.assertRaisesRegex(ValueError, 'ko only'):
            clean.work(('fx', 'en', str(self.src), {'ko': KO_DEFAULT, 'pii': PII_DEFAULT, 'quality': self.spec}))


if __name__ == '__main__':
    unittest.main()
