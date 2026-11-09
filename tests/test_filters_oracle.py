"""data/filters.py defaults must stay byte-identical to the frozen pre-refactor oracle."""
import json
from pathlib import Path
import unittest

from dataqa import gold, oracle

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / 'tests' / 'golden' / 'filters_oracle_v1.json'
REAL_LOCAL = ROOT / 'dataqa' / 'review' / 'real_gold.jsonl'


class FiltersOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = json.loads(ORACLE.read_text(encoding='utf-8'))

    def test_seed_gold_outputs_unchanged(self):
        from dataqa import seed_gold
        now = [oracle.probe(r['text']) for r in seed_gold.build()]
        self.assertEqual(len(now), len(self.frozen['seed_gold']))
        for got, want in zip(now, self.frozen['seed_gold']):
            self.assertEqual(got, want)

    def test_ko_fuzz_unchanged_and_not_vacuous(self):
        rows = [oracle.probe(t) for t in oracle.fuzz_ko_texts()]
        self.assertEqual(oracle.digest(rows), self.frozen['fuzz_ko']['digest'])
        reasons = self.frozen['fuzz_ko']['ko_reasons']
        for needed in ('None', 'short', 'few_hangul', 'low_hangul_ratio', 'dup_lines', 'ngram_rep', 'low_alnum', 'bad_encoding'):
            self.assertGreater(reasons.get(needed, 0), 0, needed)

    def test_pii_fuzz_unchanged_and_not_vacuous(self):
        rows = [oracle.probe(t) for t in oracle.fuzz_pii_texts()]
        self.assertEqual(oracle.digest(rows), self.frozen['fuzz_pii']['digest'])
        self.assertGreater(self.frozen['fuzz_pii']['masked_changed'], 50)

    @unittest.skipUnless(REAL_LOCAL.exists(), 'local-only pilot texts (not committed) are absent')
    def test_real_pilot_outputs_unchanged(self):
        texts = {oracle.sha(r['text']): r['text'] for r in gold.load(REAL_LOCAL)}
        for want in self.frozen['real_pilot']:
            self.assertEqual(oracle.probe(texts[want['text_sha256']]), want)


if __name__ == '__main__':
    unittest.main()
