import copy
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

from common.data_contracts import atomic_json, identity, resolve
from train.final_resume_probe import (restore_state, save_state, diagnostic_lr,
                                      verify_optimizer_step, file_stats, check_spec, verify_parent_metadata)
from scripts.verify_final_resume_probe import seal, verify_inventory, compare_states, ledger_review, prepare


class FinalResumeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.group_temp = tempfile.TemporaryDirectory()
        dist.init_process_group('gloo', init_method='file://'+cls.group_temp.name+'/rendezvous', rank=0, world_size=1)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.group_temp.cleanup()

    def make_model(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.25), torch.nn.Linear(8, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9,.95), weight_decay=.1)
        return model, optimizer

    def update(self, model, optimizer):
        x, y = torch.randn(3,4), torch.randn(3,2)
        loss = (model(x)-y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    def save(self, model, optimizer, path, step):
        meta = dict(kind='cpu_fixture', step=step, world_size=1)
        save_state(model, optimizer, path, 0, 'cpu', meta)
        proof = seal(path, meta, path.parent/(path.name+'.certificate.json'))
        verify_inventory(path, proof, hashes=True)
        return proof

    def test_real_dcp_full_state_roundtrip_and_rng_sensitive_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(183)
            model, optimizer = self.make_model()
            for _ in range(3):
                self.update(model, optimizer)
            self.save(model, optimizer, root/'parent', 3)
            parent_stats = file_stats(root/'parent')
            continuous_model, continuous_optimizer = self.make_model()
            restore_state(continuous_model, continuous_optimizer, root/'parent', 0, 'cpu', 3)
            self.save(continuous_model, continuous_optimizer, root/'restored', 3)
            comparison = compare_states(root/'parent', root/'restored', world=1)
            self.assertEqual(comparison['model_tensors'], 4)
            self.assertEqual(comparison['optimizer_tensors'], 12)
            self.assertEqual(comparison['tensors'], 16)
            # Comparison deserializes on CPU but must not advance the RNG.
            losses = [self.update(continuous_model, continuous_optimizer) for _ in range(2)]
            self.save(continuous_model, continuous_optimizer, root/'continuous', 5)
            first_model, first_optimizer = self.make_model()
            restore_state(first_model, first_optimizer, root/'parent', 0, 'cpu', 3)
            loss1 = self.update(first_model, first_optimizer)
            self.save(first_model, first_optimizer, root/'first', 4)
            resumed_model, resumed_optimizer = self.make_model()
            restore_state(resumed_model, resumed_optimizer, root/'first', 0, 'cpu', 4)
            loss2 = self.update(resumed_model, resumed_optimizer)
            self.save(resumed_model, resumed_optimizer, root/'resumed', 5)
            self.assertEqual(losses, [loss1, loss2])
            self.assertTrue(compare_states(root/'continuous', root/'resumed', world=1)['exact'])
            self.assertEqual(parent_stats, file_stats(root/'parent'))
            with self.assertRaises(FileExistsError):
                save_state(model, optimizer, root/'parent', 0, 'cpu', {})

    def test_missing_optimizer_wrong_step_corrupt_rng_and_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer = self.make_model()
            self.update(model, optimizer)
            proof = self.save(model, optimizer, root/'source', 1)
            other, opt = self.make_model()
            with self.assertRaisesRegex(ValueError, 'optimizer step'):
                restore_state(other, opt, root/'source', 0, 'cpu', 2)
            torch.save({'cpu': torch.tensor([1.])}, root/'source/rng_rank0.pt')
            with self.assertRaisesRegex(ValueError, 'inventory/stat'):
                verify_inventory(root/'source', proof)
            with self.assertRaisesRegex(ValueError, 'RNG'):
                restore_state(other, opt, root/'source', 0, 'cpu', 1)
            (root/'source/.complete').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                restore_state(other, opt, root/'source', 0, 'cpu', 1)
            missing = root/'weights_only'
            msd, _ = get_state_dict(model, optimizer)
            dcp.save({'model': msd}, checkpoint_id=missing)
            (missing/'.complete').touch()
            with self.assertRaisesRegex(ValueError, 'optimizer state missing'):
                restore_state(other, opt, missing, 0, 'cpu', 1)

    def test_content_difference_not_hidden_by_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer = self.make_model()
            self.update(model, optimizer)
            self.save(model, optimizer, root/'a', 1)
            with torch.no_grad():
                next(model.parameters()).add_(.01)
            self.save(model, optimizer, root/'b', 1)
            with self.assertRaisesRegex(ValueError, 'tensor differs'):
                compare_states(root/'a', root/'b', world=1)
            (root/'a/extra').write_text('unmanaged residue')
            with self.assertRaisesRegex(ValueError, 'completeness'):
                seal(root/'a', dict(kind='cpu_fixture', step=1, world_size=1), root/'rejected.json')

    def rows(self):
        config = dict(max_steps=57000, warmup=1000, lr=.0003, min_lr_ratio=.1,
                      decay_frac=.15, seq_len=4096, global_tokens=2097152)
        rows = []
        for mode, steps in [('continuous', [57001,57002]), ('first', [57001]), ('resumed', [57002])]:
            for rank in range(8):
                rows.append(dict(mode=mode, rank=rank, success=True, plan_id='fixed', parent_step=57000,
                                 diagnostic_only=True, steps=[dict(step=s, source_sequences={'ko':48,'en':16}) for s in steps],
                                 metrics=[dict(step=s,loss=2.+(s-57000)/100,gnorm=1.,lr=diagnostic_lr(config),elapsed_seconds=1.+rank) for s in steps]))
        return config, rows

    def review(self, config, rows):
        return ledger_review(rows, parent_step=57000, world=8, config=config, source_order=['ko','en'], plan_id='fixed')

    def test_actual_step_rank_accounting_replay_and_failures(self):
        config, rows = self.rows()
        before = copy.deepcopy(config)
        report = self.review(config, rows)
        self.assertEqual(report['main_committed_tokens_added'], 0)
        self.assertEqual(report['unique_diagnostic_tokens'], 4194304)
        self.assertEqual(report['actual_diagnostic_committed_tokens'], 8388608)
        self.assertEqual(sum(report['source_tokens'].values()), 4194304)
        self.assertEqual(config, before)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.review(config, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.review(config, rows+[rows[0]])
        for field, value in [('loss', float('nan')), ('lr', .001)]:
            bad = copy.deepcopy(rows)
            bad[-1]['metrics'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'finite metric/LR'):
                self.review(config, bad)
        bad = copy.deepcopy(rows)
        bad[-1]['steps'][0]['source_sequences'] = {'ko':47,'en':17}
        with self.assertRaisesRegex(ValueError, 'sampler choices'):
            self.review(config, bad)
        bad = copy.deepcopy(rows)
        bad[-1]['metrics'][0]['loss'] += .1
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            self.review(config, bad)
        bad = copy.deepcopy(rows)
        bad[-1]['success'] = False
        with self.assertRaisesRegex(ValueError, 'completion identity'):
            self.review(config, bad)

    def test_scope_identity_and_existing_output_rejected(self):
        p = dict(purpose='isolated_final_resume_probe',parent_step=57000,world_size=8,updates=3,
                 lr_policy='freeze_last_parent_update', main_target_modified=False)
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            check_spec(dict(plan=p, plan_id=identity(p)))
        p['updates'] = 2
        with self.assertRaisesRegex(ValueError, 'identity'):
            check_spec(dict(plan=p, plan_id='wrong'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'new independent'):
                prepare(Path(directory)/'parent', Path(directory))

    def test_native_saved_path_normalization_preserves_original_contract(self):
        config = dict(model='3b', max_steps=57000, out_dir='runs/3b_full_v1/training')
        original = copy.deepcopy(config)
        parent = {'id':'unchanged'}
        meta = dict(step=57000, world_size=8, config=dict(config,out_dir=str(resolve(config['out_dir']))), training_contract=parent)
        verify_parent_metadata(meta, config, parent, 'unchanged')
        self.assertEqual(config, original)
        bad = copy.deepcopy(meta)
        bad['config']['max_steps'] = 57002
        with self.assertRaisesRegex(ValueError, 'Parent metadata'):
            verify_parent_metadata(bad, config, parent, 'unchanged')

    def test_independent_process_recreates_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(9123)
            model, optimizer = self.make_model()
            self.update(model, optimizer)
            self.save(model, optimizer, root/'parent', 1)
            expected_loss = self.update(model, optimizer)
            self.save(model, optimizer, root/'expected', 2)
            code = '''
import json, sys
from pathlib import Path
import torch, torch.distributed as dist
from train.final_resume_probe import restore_state, save_state
root=Path(sys.argv[1]);torch.set_num_threads(2)
dist.init_process_group('gloo',init_method='file://'+str(root/'child_group'),rank=0,world_size=1)
try:
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.25),torch.nn.Linear(8,2))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=.1)
    restore_state(model,opt,root/'parent',0,'cpu',1)
    x,y=torch.randn(3,4),torch.randn(3,2)
    loss=(model(x)-y).square().mean();loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step();opt.zero_grad(set_to_none=True)
    save_state(model,opt,root/'child',0,'cpu',dict(kind='cpu_fixture',step=2,world_size=1))
    (root/'child_loss.json').write_text(json.dumps(loss.item()))
finally:
    dist.destroy_process_group()
'''
            child = subprocess.run([sys.executable,'-c',code,str(root)], capture_output=True, text=True, timeout=60)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(json.loads((root/'child_loss.json').read_text()), expected_loss)
            self.assertTrue(compare_states(root/'expected',root/'child',world=1)['exact'])


if __name__ == '__main__':
    unittest.main()
