"""Final-evaluation orchestration proofs; never a full-size training certificate."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from common.data_contracts import atomic_json, artifact_path, sha256_file
from scripts.evaluate_final_rehearsal import completed_rehearsal, rehearsal_observation, verify_final_result, reject_live_orphan


class FinalEvaluationTest(unittest.TestCase):
    def test_live_orphan_blocks_duplicate_worker_without_killing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)',
                                       'evaluate_final_rehearsal.py', '--worker', str(root)])
            try:
                atomic_json(root/'worker_state.json', dict(pid=process.pid))
                with self.assertRaisesRegex(RuntimeError, 'still live'):
                    reject_live_orphan(root)
                self.assertIsNone(process.poll())
            finally:
                process.terminate(); process.wait(timeout=5)
            reject_live_orphan(root)

    def test_trigger_requires_live_matching_handle_and_late_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'training').mkdir()
            process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)',
                                        'scripts/run_rehearsal.py', root.name])
            try:
                (root/'pid').write_text(str(process.pid))
                atomic_json(root/'state.json', dict(phase='running'))
                log = root/'training/log.jsonl'
                log.write_text(json.dumps(dict(step=100, tok_s=367000))+'\n')
                self.assertFalse(rehearsal_observation(root)['ready'])
                log.write_text(json.dumps(dict(step=4000, tok_s=367000))+'\n'+ '{"step":')
                observation = rehearsal_observation(root)
                self.assertTrue(observation['ready']); self.assertFalse(observation['complete'])
                atomic_json(root/'state.json', dict(phase='error', error='fixture'))
                with self.assertRaisesRegex(RuntimeError, 'recovery'):
                    rehearsal_observation(root)
            finally:
                process.terminate(); process.wait(timeout=5)
            atomic_json(root/'state.json', dict(phase='running'))
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                rehearsal_observation(root)

    def test_final_checkpoint_proof_is_bound_and_wrong_step_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root/'training/ckpt_0005000'; checkpoint.mkdir(parents=True)
            (root/'checkpoint_certificates').mkdir()
            certificate = root/'checkpoint_certificates/ckpt_0005000.json'
            meta = dict(step=5000, config=dict(model='1b', max_steps=5000), training_contract=dict(id='fixture'))
            atomic_json(checkpoint/'meta.json', meta); (checkpoint/'.complete').touch()
            atomic_json(certificate, dict(status='passed_checkpoint_storage', step=5000, checkpoint=str(checkpoint),
                                         training_contract_id='fixture', files=[]))
            atomic_json(root/'contract.json', dict(code={}))
            report = dict(status='passed_actual_rehearsal_training', final_checkpoint=str(checkpoint),
                          counts=dict(committed_tokens=5000*2097152, contract_id='fixture'),
                          evidence={artifact_path(certificate):sha256_file(certificate)})
            def publish():
                atomic_json(root/'report.json', report)
                atomic_json(root/'.done', dict(report_sha256=sha256_file(root/'report.json'), contract_sha256=sha256_file(root/'contract.json')))
            publish(); self.assertEqual(completed_rehearsal(root)[1], checkpoint)
            meta['step'] = 4; atomic_json(checkpoint/'meta.json', meta)
            with self.assertRaisesRegex(ValueError, 'identity'):
                completed_rehearsal(root)
            meta['step'] = 5000; atomic_json(checkpoint/'meta.json', meta)
            atomic_json(certificate, dict(status='passed_checkpoint_storage', step=4))
            with self.assertRaisesRegex(ValueError, 'evidence'):
                completed_rehearsal(root)
            report['counts']['committed_tokens'] = 4*2097152; publish()
            with self.assertRaisesRegex(ValueError, 'count'):
                completed_rehearsal(root)

    def test_actual_tiny_native_full_targets_and_fresh_hash_reuse(self):
        from test_eval_runner import fixture as checkpoint_fixture
        from test_split_inputs import fixture as source_fixture
        from data.local_split_reader import prepare_reader
        from eval.prepare_holdout import local_verified_inputs
        from data.split_inputs import prepare
        from eval.run import run, records
        from tokenizers import Tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sources, split = source_fixture(root)
            reader = prepare_reader(split, root/'reader', scratch=root/'scratch')
            checkpoint, tok, _, prompts = checkpoint_fixture(root)
            meta = json.loads((checkpoint/'meta.json').read_text()); meta['step'] = 5000
            atomic_json(checkpoint/'meta.json', meta)
            prompts.write_text('\n'.join(json.dumps(dict(id=str(i), category='ko' if i<3 else 'en', text='안녕' if i<3 else 'hello')) for i in range(6)))
            with local_verified_inputs(sources, split, reader, cache_mib=64) as inputs:
                prepare(sources, split, root/'holdout', dict(ko=40,en=40), mode='eval', categories=dict(ko='ko',en='en'), verified_inputs=inputs)
                result = run(checkpoint, tok, root/'holdout/samples.jsonl', prompts, root/'eval.json',
                             seq_len=8, stride=4, max_new_tokens=1, holdout_manifest=root/'holdout/manifest.json', verified_inputs=inputs)
                inputs.selection.db.execute('SELECT 1')  # Borrowed context survives.
            tokenizer = Tokenizer.from_file(str(tok))
            targets = {k:sum(len(tokenizer.encode(r['text'], add_special_tokens=False).ids) for r in records(root/'holdout/samples.jsonl') if r['category']==k) for k in ('ko','en')}
            proof = dict(files=[dict(path=str(p.relative_to(checkpoint)),sha256=sha256_file(p)) for p in checkpoint.rglob('*') if p.is_file()])
            reference = copy.deepcopy(result)
            verify_final_result(result, proof, sha256_file(tok), targets, reference)
            bad = copy.deepcopy(result); bad['languages']['ko']['valid_tokens'] -= 1
            with self.assertRaisesRegex(ValueError, 'coverage'):
                verify_final_result(bad, proof, sha256_file(tok), targets, reference)
            bad = copy.deepcopy(result); bad['contract']['checkpoint']['.metadata'] = 'corrupt'
            with self.assertRaisesRegex(ValueError, 'content'):
                verify_final_result(bad, proof, sha256_file(tok), targets, reference)
            bad = copy.deepcopy(result); bad['generations'].pop()
            with self.assertRaisesRegex(ValueError, 'prompt'):
                verify_final_result(bad, proof, sha256_file(tok), targets, reference)
            bad = copy.deepcopy(result); bad['checkpoint_step'] = 4
            with self.assertRaisesRegex(ValueError, 'step'):
                verify_final_result(bad, proof, sha256_file(tok), targets, reference)


if __name__ == '__main__':
    unittest.main()
