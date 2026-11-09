import collections
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from data import clean
from data.filters import KoPolicy, PiiPolicy, KO_DEFAULT, PII_DEFAULT

GOOD = ' '.join(f'한국어문서내용{i}번째문장입니다' for i in range(60))


def write(path, rows):
    pq.write_table(pa.table({'text': rows}), path)


class CleanSignatureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / 'input.parquet'
        write(self.src, [GOOD, '짧음'])
        self.patch = patch.object(clean, 'ROOT', str(self.root))
        self.patch.start()
        self.out = self.root / 'clean/fx/input.parquet'

    def tearDown(self):
        self.patch.stop()
        self._tmp.cleanup()

    def run_work(self, lang='ko', policy=None, src=None):
        args = ('fx', lang, str(src or self.src))
        return clean.work(args + (policy,) if policy else args)

    def test_signature_is_recorded_and_stable(self):
        self.run_work()
        stats = json.loads(Path(str(self.out) + '.stats.json').read_text())
        sig = stats['signature']
        self.assertEqual(sig['input']['bytes'], self.src.stat().st_size)
        self.assertEqual(sig['ko_policy'], KO_DEFAULT.as_dict())
        self.assertEqual(sig['pii_policy'], PII_DEFAULT.as_dict())
        self.assertEqual(stats['counts'], {'kept': 1, 'short': 1})
        self.assertEqual(clean.signature('ko', str(self.src))['id'], sig['id'])
        counts, _ = self.run_work()[1]
        self.assertEqual(counts, collections.Counter(kept=1, short=1))   # reuse with identical signature

    def test_changed_input_policy_or_lang_refuse_reuse(self):
        self.run_work()
        with self.assertRaisesRegex(RuntimeError, 'signature differs'):
            self.run_work(policy={'ko': KoPolicy(min_chars=10), 'pii': PII_DEFAULT})
        with self.assertRaisesRegex(RuntimeError, 'signature differs'):
            self.run_work(policy={'ko': KO_DEFAULT, 'pii': PiiPolicy(phone='intl')})
        with self.assertRaisesRegex(RuntimeError, 'signature differs'):
            self.run_work(lang='en')
        write(self.src, [GOOD, '짧음', '다른문서'])      # same name, different content
        with self.assertRaisesRegex(RuntimeError, 'signature differs'):
            self.run_work()

    def test_legacy_output_without_signature_is_refused(self):
        self.run_work()
        stats_path = Path(str(self.out) + '.stats.json')
        stats = json.loads(stats_path.read_text())
        del stats['signature']
        stats_path.write_text(json.dumps(stats))
        with self.assertRaisesRegex(RuntimeError, 'Legacy output'):
            self.run_work()

    def test_policy_changes_the_cleaning_result_and_travels_with_the_job(self):
        write(self.src, [GOOD + '\n문의 +82-10-0000-1234 로 연락', '짧음'])
        policy = {'ko': KO_DEFAULT, 'pii': PiiPolicy(phone='intl')}
        self.run_work(policy=policy)
        kept = pq.read_table(self.out).column('text').to_pylist()
        self.assertIn('<phone>', kept[0])
        self.assertNotIn('+82-10-0000-1234', kept[0])

    def test_default_policy_leaves_international_numbers_as_before(self):
        write(self.src, [GOOD + '\n문의 +82-10-0000-1234 로 연락', '짧음'])
        self.run_work()
        self.assertIn('+82-10-0000-1234', pq.read_table(self.out).column('text').to_pylist()[0])

    def test_load_policy_file(self):
        p = self.root / 'p.json'
        p.write_text(json.dumps({'ko': {'min_chars': 10}, 'pii': {'email': 'no_git'}}))
        pol = clean.load_policy(str(p))
        self.assertEqual((pol['ko'].min_chars, pol['pii'].email), (10, 'no_git'))
        self.assertEqual(pol['ko'].min_hangul_ratio, KO_DEFAULT.min_hangul_ratio)
        p.write_text(json.dumps({'ko': {'min_chars': 10}, 'thresholds': {}}))
        with self.assertRaises(SystemExit):
            clean.load_policy(str(p))
        p.write_text(json.dumps({'ko': {'no_such_field': 1}}))
        with self.assertRaises(TypeError):
            clean.load_policy(str(p))


if __name__ == '__main__':
    unittest.main()
