import json
from pathlib import Path
import tempfile
import unittest
from common.data_contracts import atomic_json,sha256_file
from scripts.run_final_tokenizer import byte_budgets,efficiency,wait_review,wait_f4


class FinalTokenizerGatesTest(unittest.TestCase):
    def test_exact_three_gb_budgets(self):
        weights={'ko':.50,'en':.35,'code':.041695186553138114,'math':.10830481344686188}
        r=byte_budgets(weights);self.assertEqual(sum(r.values()),3000000000);self.assertEqual(r['en'],1050000000)
        with self.assertRaises(ValueError):byte_budgets({'bad':-1})

    def test_predeclared_efficiency_category_and_weighted_limits(self):
        cats=['ko','en','code','math'];baseline=dict(categories={c:dict(tokens=100,utf8_bytes=400,documents=3) for c in cats})
        weights={c:.25 for c in cats};policy=dict(maximum_category_token_ratio=1.10,maximum_weighted_token_ratio=1.02)
        self.assertEqual(efficiency(baseline,baseline,weights,policy)['status'],'passed')
        for mode in ['category','weighted']:
            actual=json.loads(json.dumps(baseline))
            if mode=='category':actual['categories']['code']['tokens']=111
            else:
                for c in cats:actual['categories'][c]['tokens']=103
            self.assertEqual(efficiency(actual,baseline,weights,policy)['status'],'needs_review')

    def test_changed_review_and_failed_f4_never_open_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/'distribution.json',dict(counts=dict(train=2,eval=1)))
            atomic_json(root/'distribution_review.json',dict(distribution_sha256='wrong',decision='proceed'))
            with self.assertRaisesRegex(ValueError,'review'):wait_review(root,{})
            atomic_json(root/'distribution_review.json',dict(distribution_sha256=sha256_file(root/'distribution.json'),decision='proceed'))
            self.assertEqual(wait_review(root,{})['decision'],'proceed')
            source=root/'source';source.mkdir();atomic_json(source/'state.json',dict(phase='error',reason='fixture'))
            with self.assertRaisesRegex(RuntimeError,'producer error'):wait_f4(root,source,'expected',3)
