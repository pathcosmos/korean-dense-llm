"""CPU orchestration/coverage tests, not actual final3B evaluation acceptance."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.data_contracts import atomic_json,sha256_file
from scripts.evaluate_final_main import main_observation,reject_live_orphan,completed_main,verify_result,admission


class FinalMainEvaluationTest(unittest.TestCase):
    def test_admission_requires_combined_future_main_and_evaluation_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'contract.json',dict(tiered_retention={},gpu_identity=['fixture']*8,max_pss_gib=384))
            c=dict(main_run=str(root),host_rss_gib=256)
            hw=dict(gpus=['fixture']*8,memory=dict(cache_aware_admission=dict(headroom_gib=600)))
            with patch('scripts.resource_admission.inspect',return_value=hw):
                with self.assertRaisesRegex(RuntimeError,'future peak headroom'):admission(c,root)

    def test_late_trigger_requires_actual_main_handles_and_errors_are_not_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'training').mkdir()
            process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
                'scripts/supervise_main_training.py','train.monitored',str(root)])
            try:
                (root/'pid').write_text(str(process.pid))
                atomic_json(root/'state.json',dict(phase='running',worker_pid=process.pid))
                log=root/'training/log.jsonl'
                log.write_text(json.dumps(dict(step=56000,loss=3.))+'\n')
                self.assertFalse(main_observation(root)['ready'])
                log.write_text(json.dumps(dict(step=56500,loss=3.))+'\n{"step":')
                self.assertTrue(main_observation(root)['ready'])
                atomic_json(root/'state.json',dict(phase='error'))
                with self.assertRaisesRegex(RuntimeError,'requires recovery'):main_observation(root)
                self.assertIsNone(process.poll())
            finally:
                process.terminate();process.wait(timeout=5)
            atomic_json(root/'state.json',dict(phase='running',worker_pid=process.pid))
            with self.assertRaisesRegex(RuntimeError,'handle missing'):main_observation(root)

    def test_complete_marker_does_not_replace_full_target_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.done').write_text('{}')
            self.assertTrue(main_observation(root)['ready'])
            with patch('scripts.audit_main_training.audit_run',return_value=dict(validation_source=False,
                    counts=dict(committed_tokens=2097152,successful_target_ranks=list(range(8))))):
                with self.assertRaisesRegex(ValueError,'full8rank'):completed_main(root)
            with patch('scripts.audit_main_training.audit_run',return_value=dict(validation_source=False,
                    counts=dict(committed_tokens=119537664000,successful_target_ranks=list(range(7))))):
                with self.assertRaisesRegex(ValueError,'full8rank'):completed_main(root)

    def test_orphan_worker_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
                                     'evaluate_final_main.py','--worker',str(root)])
            try:
                atomic_json(root/'worker_state.json',dict(pid=process.pid))
                with self.assertRaisesRegex(RuntimeError,'still live'):reject_live_orphan(root)
                self.assertIsNone(process.poll())
            finally:
                process.terminate();process.wait(timeout=5)
            reject_live_orphan(root)

    def test_actual_tiny_native_provenance_full_targets_content_ppl_and_decode(self):
        from test_eval_runner import fixture as checkpoint_fixture
        from test_split_inputs import fixture as source_fixture
        from data.local_split_reader import prepare_reader
        from eval.prepare_holdout import local_verified_inputs
        from data.split_inputs import prepare
        from eval.run import run,records
        from tokenizers import Tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources,split=source_fixture(root)
            reader=prepare_reader(split,root/'reader',scratch=root/'scratch')
            checkpoint,tok,_,prompts=checkpoint_fixture(root)
            meta=json.loads((checkpoint/'meta.json').read_text());meta['step']=57000
            atomic_json(checkpoint/'meta.json',meta)
            prompts.write_text('\n'.join(json.dumps(dict(id=str(i),category='ko' if i<3 else 'en',text='안녕' if i<3 else 'hello')) for i in range(6)))
            with local_verified_inputs(sources,split,reader,cache_mib=64) as inputs:
                prepare(sources,split,root/'holdout',dict(ko=40,en=40),mode='eval',categories=dict(ko='ko',en='en'),verified_inputs=inputs)
                result=run(checkpoint,tok,root/'holdout/samples.jsonl',prompts,root/'eval.json',
                    seq_len=8,stride=4,max_new_tokens=2,holdout_manifest=root/'holdout/manifest.json',verified_inputs=inputs)
                inputs.selection.db.execute('SELECT 1')
            tokenizer=Tokenizer.from_file(str(tok))
            targets={k:sum(len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for r in records(root/'holdout/samples.jsonl') if r['category']==k) for k in ['ko','en']}
            proof=dict(files=[dict(path=str(p.relative_to(checkpoint)),sha256=sha256_file(p)) for p in checkpoint.rglob('*') if p.is_file()])
            reference=copy.deepcopy(result)
            def verify(r,proof=proof):
                verify_result(r,proof,tokenizer,targets,reference,expected_step=57000,expected_device='cpu')
            verify(result)
            bad=copy.deepcopy(result);bad['languages']['ko']['valid_tokens']-=1
            with self.assertRaisesRegex(ValueError,'target coverage'):verify(bad)
            bad=copy.deepcopy(result);bad['aggregate']['mean_nll']+=.1
            with self.assertRaisesRegex(ValueError,'weighted NLL'):verify(bad)
            bad=copy.deepcopy(result);bad['languages']['en']['perplexity']+=1
            with self.assertRaisesRegex(ValueError,'PPL differs'):verify(bad)
            bad=copy.deepcopy(result);bad['generations'][0]['generated_text']='changed'
            with self.assertRaisesRegex(ValueError,'decode differs'):verify(bad)
            bad=copy.deepcopy(result);bad['generations'][0]['stopped_at_eos']=not bad['generations'][0]['stopped_at_eos']
            with self.assertRaisesRegex(ValueError,'EOS/decode'):verify(bad)
            bad=copy.deepcopy(result);bad['generations'][0]['prompt_ids']=[0]
            with self.assertRaisesRegex(ValueError,'prompt token IDs'):verify(bad)
            bad=copy.deepcopy(result);bad['checkpoint_step']=5000
            with self.assertRaisesRegex(ValueError,'step differs'):verify(bad)
            badproof=copy.deepcopy(proof);badproof['files'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'checkpoint content'):verify(result,badproof)
            bad=copy.deepcopy(result);bad['generations'].pop()
            with self.assertRaisesRegex(ValueError,'Fixed6'):verify(bad)


if __name__=='__main__':
    unittest.main()
