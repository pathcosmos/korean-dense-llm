import copy
import json
from pathlib import Path
import tempfile
import unittest
from common.data_contracts import atomic_json,sha256_file
from scripts.run_final_tokens import pilot_projection,verify_execution_review,verify_tokenizer_run,token_weights,verify_mix_review,final_counts,GIB


class FinalTokensGateTest(unittest.TestCase):
    def pilot(self):
        p=dict(sources={'code':dict(files=3,utf8_bytes=10000),'en':dict(files=4,utf8_bytes=20000)},selected_files=[])
        r=dict(status='pilot_complete',shards=2,planned_shards=7,tokens=1000,files=[])
        for source in ['code','en']:
            p['selected_files'].append(dict(source=source,path=source,rows=100,bytes=300,utf8_bytes=1000))
            r['files'].append(dict(source=source,path=source,train_docs=90,holdout_docs=10,tokens=450,holdout_tokens=50,
                                   input_bytes=300,utf8_bytes=1000,reused=False))
        return p,r

    def test_projection_full_file_coverage_and_reused_measurements(self):
        p,r=self.pilot();result=pilot_projection(p,r)
        self.assertEqual(result['projected_output_bytes'],30000)
        self.assertEqual(result['upper_planning_output_bytes'],37500)
        self.assertEqual(result['required_remaining_bytes'],35500)
        reused=copy.deepcopy(r)
        for f in reused['files']:f.update(reused=True,utf8_bytes=None)
        self.assertEqual(pilot_projection(p,reused),result)
        for mutate in [lambda x:x['files'].pop(),lambda x:x.update(planned_shards=6),lambda x:x.update(tokens=999),
                       lambda x:x['files'][0].update(utf8_bytes=999),lambda x:x['files'][0].update(holdout_docs=11),
                       lambda x:x['files'][0].update(tokens=89),lambda x:x['files'][0].update(utf8_bytes=None)]:
            bad=copy.deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):pilot_projection(p,bad)

    def test_review_binding_and_storage_before_full_encoding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,r=self.pilot();projection=pilot_projection(p,r)
            atomic_json(root/'pilot.json',r);atomic_json(root/'projection.json',projection);atomic_json(root/'contract.json',dict(version=1))
            review=dict(decision='proceed',pilot_report_sha256=sha256_file(root/'pilot.json'),projection_sha256=sha256_file(root/'projection.json'),
                        contract_sha256=sha256_file(root/'contract.json'),workers=24,rayon_threads=2,durable_budget_gib=1,future_checkpoint_gib=128,reason='bounded measured candidate, not full optimum')
            atomic_json(root/'execution_review.json',review)
            self.assertEqual(verify_execution_review(root,root/'pilot.json',projection,329*GIB),review)
            with self.assertRaisesRegex(RuntimeError,'storage'):verify_execution_review(root,root/'pilot.json',projection,328*GIB)
            for field,value in [('pilot_report_sha256','wrong'),('decision','pending'),('workers',32),('future_checkpoint_gib',float('inf')),('reason','')]:
                bad=dict(review);bad[field]=value;atomic_json(root/'execution_review.json',bad)
                with self.assertRaises(ValueError):verify_execution_review(root,root/'pilot.json',projection,1000*GIB)

    def tokenizer(self,root):
        tok=root/'tokenizer.json';tok.write_text('fixture bytes')
        c=dict(sources_sha256='source',canonical_split_sha256='split')
        report=dict(status='passed',tokenizer=str(tok),tokenizer_sha256=sha256_file(tok))
        for name,field in [('tokenizer_check.json','tokenizer_check_sha256'),('efficiency.json','efficiency_sha256'),
                           ('train_sample/manifest.json','train_sample_manifest_sha256'),('quality_inputs/manifest.json','quality_manifest_sha256'),
                           ('distribution.json','distribution_sha256'),('distribution_review.json','distribution_review_sha256')]:
            atomic_json(root/name,dict(status='passed'));report[field]=sha256_file(root/name)
        atomic_json(root/'contract.json',c);atomic_json(root/'report.json',report)
        atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
        return report,c

    def test_final_tokenizer_quality_evidence_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r,c=self.tokenizer(root)
            self.assertEqual(verify_tokenizer_run(root,'source','split'),(r,c))
            with self.assertRaisesRegex(ValueError,'source/split'):verify_tokenizer_run(root,'other','split')
            atomic_json(root/'efficiency.json',dict(status='needs_review'))
            with self.assertRaisesRegex(ValueError,'evidence changed'):verify_tokenizer_run(root,'source','split')
            r['efficiency_sha256']=sha256_file(root/'efficiency.json');atomic_json(root/'report.json',r)
            atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
            with self.assertRaisesRegex(ValueError,'checks did not pass'):verify_tokenizer_run(root,'source','split')

    def test_training_token_mix_distinct_from_tokenizer_byte_mix(self):
        c=dict(categories=dict(wiki='ko',web='ko',en='en',code='code',math='math'),weights=dict(wiki=.1,web=.4,en=.35,code=.05,math=.1))
        weights=token_weights(c)
        self.assertAlmostEqual(sum(weights.values()),1)
        self.assertAlmostEqual(weights['wiki']+weights['web'],.5)
        for source,expected in [('wiki',.1),('web',.4),('en',.3),('code',.1),('math',.1)]:self.assertAlmostEqual(weights[source],expected)
        with self.assertRaisesRegex(ValueError,'four categories'):token_weights(dict(categories=dict(code='code'),weights=dict(code=1)))

    def test_native_full_counts_and_pilot_cannot_fake_full_coverage(self):
        from data.parallel_tokenize import run
        from data.verified_split_context import VerifiedSplitContext
        from test_parallel_tokenize import fixture
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            with VerifiedSplitContext(sm,split) as ctx:
                chosen=[records[0]['path'] for records in ctx.records.values()]
                run(sm,split,tok,root/'tokens',workers=2,experimental=True,selected_files=chosen,verified_context=ctx)
                with self.assertRaises(FileNotFoundError):final_counts(ctx,root/'tokens',dict(web='ko',code='en'),160,1)
                full=run(sm,split,tok,root/'tokens',workers=2,experimental=True,verified_context=ctx)
                counts,hold=final_counts(ctx,root/'tokens',dict(web='ko',code='en'),160,1)
                self.assertEqual(sum(r['tokens']+r['holdout_tokens'] for r in counts.values()),full['tokens'])
                self.assertEqual(sum(r['train_docs']+r['holdout_docs'] for r in counts.values()),160)
                self.assertGreater(min(hold['ko'],hold['en']),0)
                with self.assertRaisesRegex(ValueError,'below target'):final_counts(ctx,root/'tokens',dict(web='ko',code='en'),160)

    def test_actual_count_bound_mix_review_not_tokenizer_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/'counts.json',dict(actual_tokens=123));atomic_json(root/'contract.json',dict(version=1))
            categories=dict(wiki='ko',web='ko',en='en',code='code',math='math')
            r=dict(decision='proceed',counts_sha256=sha256_file(root/'counts.json'),contract_sha256=sha256_file(root/'contract.json'),
                   weights=dict(wiki=.01,web=.49,en=.3,code=.1,math=.1),reason='actual counts and repetition exposure reviewed')
            atomic_json(root/'mix_review.json',r);self.assertEqual(verify_mix_review(root,categories),r['weights'])
            for field,value in [('counts_sha256','wrong'),('reason',''),('weights',dict(wiki=.01,web=.44,en=.35,code=.1,math=.1)),('weights',dict(wiki=0,web=.5,en=.3,code=.1,math=.1))]:
                bad=dict(r);bad[field]=value;atomic_json(root/'mix_review.json',bad)
                with self.assertRaises(ValueError):verify_mix_review(root,categories)
