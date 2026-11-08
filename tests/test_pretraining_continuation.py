"""Bounded mix-decision and fail-closed handoff checks, not corpus completion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.mix_review_policy import SOURCES,POLICY,choose_mix
from scripts.continue_pretraining import review_f6
from common.data_contracts import atomic_json,sha256_file


def counts():
    tokens=dict(wiki_ko=300000000,fineweb2_ko=42000000000,hplt_ko=28000000000,fineweb_edu=36000000000,github_code_v1=4000000000,owm=10000000000)
    records={s:dict(tokens=n,holdout_tokens=n//100,train_docs=1000,holdout_docs=10) for s,n in tokens.items()}
    hold={c:sum(r['holdout_tokens'] for s,r in records.items() if SOURCES[s]==c) for c in set(SOURCES.values())}
    return dict(source_counts=records,category_holdout_tokens=hold)


class PretrainingContinuationTest(unittest.TestCase):
    def test_actual_supply_balances_korean_web_and_caps_wiki_repetition(self):
        source=counts();r=choose_mix(source,SOURCES);weights=r['weights']
        for category,target in POLICY['category_targets'].items():self.assertAlmostEqual(sum(w for s,w in weights.items() if SOURCES[s]==category),target)
        self.assertAlmostEqual(weights['wiki_ko'],.005)
        exposures=r['expected_exposures'][str(POLICY['main_tokens'])]
        self.assertAlmostEqual(exposures['fineweb2_ko'],exposures['hplt_ko']);self.assertLess(exposures['wiki_ko'],3)
        self.assertGreater(r['prior_wiki_5pct_exposures'],19)
        small=copy.deepcopy(source);small['source_counts']['wiki_ko']['tokens']=1000000
        self.assertAlmostEqual(choose_mix(small,SOURCES)['expected_exposures'][str(POLICY['main_tokens'])]['wiki_ko'],3)

    def test_unknown_supply_and_excessive_exposures_need_review(self):
        source=counts();bad=copy.deepcopy(source);bad['source_counts']['github_code_v1']['tokens']=1000000000
        with self.assertRaisesRegex(ValueError,'repetition'):choose_mix(bad,SOURCES)
        bad=copy.deepcopy(source);bad['category_holdout_tokens']['ko']-=1
        with self.assertRaisesRegex(ValueError,'holdout'):choose_mix(bad,SOURCES)
        with self.assertRaisesRegex(ValueError,'coverage'):choose_mix(source,dict(SOURCES,extra='ko'))
        bad=copy.deepcopy(source);bad['source_counts']['owm']['tokens']=True
        with self.assertRaisesRegex(ValueError,'invalid'):choose_mix(bad,SOURCES)

    def test_partial_or_experimental_native_marker_never_writes_mix_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);coordinator=root/'coordinator';f6=root/'f6';out=root/'tokens'
            for p in [coordinator,f6,out]:p.mkdir()
            atomic_json(f6/'contract.json',dict(code={},settings=dict(output=str(out))))
            contract=dict(f6_contract_sha256=sha256_file(f6/'contract.json'))
            def native(status,experimental):
                atomic_json(out/'contract.json',dict(experimental=experimental))
                atomic_json(out/'report.json',dict(status=status,shards=6,planned_shards=6))
                atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
                atomic_json(f6/'counts.json',dict(full_report_sha256=sha256_file(out/'report.json')))
            native('pilot_complete',False)
            with self.assertRaisesRegex(ValueError,'complete source'):review_f6(coordinator,f6,contract)
            native('complete',True)
            with self.assertRaisesRegex(ValueError,'Experimental'):review_f6(coordinator,f6,contract)
            self.assertFalse((f6/'mix_review.json').exists())


if __name__=='__main__':unittest.main()
