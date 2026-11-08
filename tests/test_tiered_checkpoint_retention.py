"""Real CPU DCP fixtures on NFS -> local; no production checkpoint deletion."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import uuid
import subprocess
import sys

from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json, artifact_path
from scripts.supervise_main_training import seal, require_authorization
from scripts.tiered_checkpoint_retention import TieredRetention, make_policy, inventory, size_tree


class TieredRetentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch.distributed.checkpoint as dcp
        cls.torch, cls.dcp = torch, dcp
        torch.set_num_threads(2)

    def setUp(self):
        self.nfs = tempfile.TemporaryDirectory(prefix='retention-fixture-', dir=Path(PROJECT_ROOT) / 'runs')
        self.disk = tempfile.TemporaryDirectory(prefix='retention-fixture-')
        self.root = Path(self.nfs.name)
        (self.root / 'training').mkdir()
        (self.root / 'checkpoint_certificates').mkdir()
        atomic_json(self.root / 'training/training_contract.json', dict(id='fixture_training'))
        self.policy = make_policy(self.root, Path(self.disk.name), 1024 * 1024, run_id=uuid.uuid4().hex)
        # These limits describe only tiny CPU fixtures, never production defaults.
        self.policy.update(reserve_bytes=0, temporary_budget_bytes=0)
        self.manager = TieredRetention(self.root, self.policy, authorized=True)

    def tearDown(self):
        self.nfs.cleanup()
        self.disk.cleanup()

    def checkpoint(self, i):
        torch, dcp = self.torch, self.dcp
        step = i * 250
        path = self.root / 'training' / f'ckpt_{step:07d}'
        dcp.save({'model': {'weight': torch.arange(16.) + i},
                  'optim': {'state':{'weight':{'step': torch.tensor(float(i)), 'exp_avg': torch.arange(16.) / 10,
                                               'exp_avg_sq':torch.arange(16.)/100}}}}, checkpoint_id=path)
        atomic_json(path / 'meta.json', dict(step=step, world_size=8, tokens_per_step=2097152,
                config=dict(model='3b', max_steps=57000), training_contract=dict(id='fixture_training')))
        for rank in range(8):
            torch.save(dict(cpu=torch.Generator().manual_seed(rank + i).get_state(), cuda=torch.tensor([], dtype=torch.uint8)),
                       path / f'rng_rank{rank}.pt')
        (path / '.complete').touch()
        seal(path, self.root / 'checkpoint_certificates' / (path.name + '.json'), 'fixture_training')
        return path

    def seed(self, n=11):
        for i in range(1, n + 1):
            self.checkpoint(i)

    def test_full_10_plus_15_distinct_and_full_payload_restores(self):
        # One checkpoint at a time matches the actual producer/copy contract.
        for i in range(1, 28):
            self.checkpoint(i)
            result = self.manager.maintain()
        self.assertEqual(len(result['nfs_checkpoints']), 10)
        self.assertEqual(len(result['local_checkpoints']), 15)
        self.assertEqual(result['distinct_checkpoints'], 25)
        self.assertFalse(set(result['nfs_checkpoints']) & set(result['local_checkpoints']))
        self.assertEqual(result['nfs_checkpoints'][0], 'ckpt_0004500')
        self.assertEqual(result['local_checkpoints'][0], 'ckpt_0000750')
        path = Path(self.policy['local_root']) / 'checkpoints' / result['local_checkpoints'][-1]
        state = dict(model={'weight': self.torch.empty(16)}, optim={'state':{'weight':{
            'step': self.torch.tensor(0.), 'exp_avg': self.torch.empty(16),'exp_avg_sq':self.torch.empty(16)}}})
        self.dcp.load(state, checkpoint_id=path)
        self.assertTrue(self.torch.equal(state['model']['weight'], self.torch.arange(16.) + 17))
        self.assertEqual(float(state['optim']['state']['weight']['step']), 17)
        for rank in range(8):
            rng = self.torch.load(path / f'rng_rank{rank}.pt', weights_only=True)
            self.assertTrue(self.torch.equal(rng['cpu'], self.torch.Generator().manual_seed(rank + 17).get_state()))

    def test_copy_failure_preserves_all_nfs_and_restarts_partial_stage(self):
        self.seed()
        def fail(point):
            if point=='copied_file':raise OSError('injected copy failure')
        self.manager.fault=fail
        with patch('time.time', wraps=__import__('time').time):
            with self.assertRaisesRegex(OSError, 'copy failure'):
                self.manager.maintain()
        self.assertEqual(len(list((self.root / 'training').glob('ckpt_*'))), 11)
        self.assertTrue(any((Path(self.policy['local_root']) / 'staging').iterdir()))
        self.assertEqual(len(TieredRetention(self.root, self.policy, authorized=True).maintain()['nfs_checkpoints']), 10)

    def test_real_process_exit_releases_lock_and_recovers(self):
        self.seed()
        policy=self.root/'test_policy.json';atomic_json(policy,self.policy)
        code='''import os,json,sys
from pathlib import Path
from scripts.tiered_checkpoint_retention import TieredRetention
def crash(point):
    if point=="published_directory":os._exit(91)
TieredRetention(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text()),authorized=True,fault=crash).maintain()
'''
        p=subprocess.run([sys.executable,'-c',code,str(self.root),str(policy)],cwd=PROJECT_ROOT,timeout=30)
        self.assertEqual(p.returncode,91)
        result=self.manager.maintain()
        self.assertEqual(len(result['nfs_checkpoints']),10)
        self.assertEqual(len(result['local_checkpoints']),1)

    def test_same_length_corruption_missing_rng_and_unpublished_copy_not_counted(self):
        self.seed()
        def corrupt(point):
            if point == 'copied_payload':
                path = Path(self.policy['local_root']) / 'staging/ckpt_0000250/rng_rank7.pt'
                data = path.read_bytes();path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        self.manager.fault = corrupt
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.manager.maintain()
        self.assertEqual(len(list((self.root / 'training').glob('ckpt_*'))), 11)
        self.manager.fault = lambda point: None
        result = self.manager.maintain()
        self.assertEqual(result['distinct_checkpoints'], 11)
        local = Path(self.policy['local_root']) / 'checkpoints/ckpt_0000250'
        (local / 'rng_rank7.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'file list'):
            self.manager.maintain()

    def test_crash_publication_and_nfs_retirement_recovery(self):
        for failure in ['copied_file', 'verified_copy', 'published_directory', 'retire_intent', 'retired_directory', 'deleted_file']:
            with self.subTest(point=failure):
                # Use a new independently owned fixture namespace each time.
                self.seed() if not list((self.root / 'training').glob('ckpt_*')) else None
                fired = []
                def crash(point):
                    if point == failure and not fired:
                        fired.append(point)
                        raise InterruptedError(point)
                self.manager.fault = crash
                with self.assertRaises(InterruptedError):
                    self.manager.maintain()
                self.manager.fault = lambda point: None
                result = self.manager.maintain()
                self.assertEqual(len(result['nfs_checkpoints']), 10)
                self.assertEqual(result['distinct_checkpoints'], 11)
                # Re-create the just-retired source for the next fault case;
                # only disposable test data are involved, under the same seal.
                if failure != 'deleted_file':
                    local = Path(self.policy['local_root']) / 'checkpoints/ckpt_0000250'
                    shutil.copytree(local, self.root / 'training/ckpt_0000250')
                    shutil.rmtree(local)
                    (Path(self.policy['local_root']) / 'certificates/ckpt_0000250.json').unlink()

    def test_local_loss_and_uncertified_nfs_never_become_resume_candidates(self):
        self.seed();result = self.manager.maintain()
        shutil.rmtree(self.policy['local_root'])
        result = TieredRetention(self.root, self.policy, authorized=True).maintain()
        self.assertEqual(len(result['nfs_checkpoints']), 10)
        self.assertEqual(result['local_checkpoints'], [])
        path = self.checkpoint(12)
        (self.root / 'checkpoint_certificates' / (path.name + '.json')).unlink()
        result = self.manager.maintain()
        self.assertEqual(len(result['nfs_checkpoints']), 10)
        self.assertEqual(result['uncertified_nfs'], [path.name])
        self.assertEqual(len(list((self.root / 'training').glob('ckpt_*'))), 11)
        from train.runtime import complete_checkpoints
        self.assertEqual(complete_checkpoints(self.root / 'training')[-1], path)
        # The supervisor fresh-seals candidates before trainer resume; a local
        # history directory never lives under training and is never scanned.
        self.assertTrue(all(p.parent == self.root / 'training' for p in complete_checkpoints(self.root / 'training')))

    def test_space_shortage_and_leftover_budget_stop_without_pruning(self):
        self.seed()
        self.policy['local_budget_bytes'] = 1
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            self.manager.maintain()
        self.assertEqual(len(list((self.root / 'training').glob('ckpt_*'))), 11)
        self.policy['local_budget_bytes'] = 100 * 1024**2
        real = shutil.disk_usage
        def no_space(path):
            usage = real(path)
            if Path(path) == Path(self.policy['local_root']):
                return type(usage)(usage.total, usage.total, 0)
            return usage
        with patch('shutil.disk_usage', side_effect=no_space):
            with self.assertRaisesRegex(RuntimeError, 'reserve'):
                self.manager.maintain()
        self.assertEqual(len(list((self.root / 'training').glob('ckpt_*'))), 11)
        def no_nfs_space(path):
            usage=real(path)
            return type(usage)(usage.total,usage.total,0) if Path(path)==self.root else usage
        self.policy['other_nfs_future_bytes']=1
        with patch('shutil.disk_usage',side_effect=no_nfs_space):
            with self.assertRaisesRegex(RuntimeError,'NFS'):
                self.manager.maintain()
        self.assertEqual(len(list((self.root/'training').glob('ckpt_*'))),11)

    def test_unauthorized_wrong_owner_symlink_and_extra_file_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not authorized'):
            TieredRetention(self.root, self.policy).maintain()
        self.assertFalse(Path(self.policy['local_root']).exists())
        self.seed();self.manager.maintain()
        local = Path(self.policy['local_root'])
        atomic_json(local / '.retention_owner.json', dict(wrong='owner'))
        with self.assertRaisesRegex(ValueError, 'owner'):
            self.manager.maintain()
        atomic_json(local / '.retention_owner.json', self.manager.owner)
        cp = self.root / 'training/ckpt_0000500'
        (cp / 'extra').write_text('must preserve')
        with self.assertRaisesRegex(ValueError, 'file list'):
            self.manager.maintain()
        (cp / 'extra').unlink()
        (cp / 'external').symlink_to(self.root / 'training/training_contract.json')
        with self.assertRaisesRegex(ValueError, 'link'):
            self.manager.maintain()

    def test_tiered_main_authorization_and_native_pruning_contract(self):
        c = dict(purpose='main_training', main_training_authorized=True, new_checkpoint_pruning_authorized=True,
                 retention_policy='nfs_latest_with_ephemeral_history', archive_required_before_prune=True,
                 tiered_retention=self.policy)
        require_authorization(c)
        for key in ['main_training_authorized', 'new_checkpoint_pruning_authorized']:
            with self.assertRaises(ValueError):
                require_authorization(dict(c, **{key: False}))
        with self.assertRaises(ValueError):
            require_authorization(dict(c, tiered_retention=dict(self.policy, resume_from_local=True)))

    def test_supervisor_rejects_native_pruning_before_launch(self):
        from scripts.supervise_main_training import run,validated_copy_settings
        from common.data_contracts import sha256_file
        torch=self.torch
        original={'candidate':json.loads((Path(PROJECT_ROOT)/'examples/3b.observed.json').read_text())}
        config=dict(original['candidate'],keep_ckpts=10,out_dir=artifact_path(self.root/'training'))
        atomic_json(self.root/'main.json',config)
        contract=dict(purpose='main_training',main_training_authorized=True,new_checkpoint_pruning_authorized=True,
                retention_policy='nfs_latest_with_ephemeral_history',archive_required_before_prune=True,
                tiered_retention=self.policy,framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda),
                configurations={'main.json':sha256_file(self.root/'main.json')},code={},evidence={},generation_steps=[250])
        atomic_json(self.root/'contract.json',contract)
        with patch('subprocess.Popen',side_effect=AssertionError('must never launch')):
            with self.assertRaisesRegex(ValueError,'Native checkpoint pruning'):
                run(self.root)
        self.assertFalse((self.root/'attempts').exists())

    def test_original_validated_copy_settings(self):
        from scripts.supervise_main_training import validated_copy_settings
        workers,evidence=validated_copy_settings()
        self.assertIn(workers,[4,8]);self.assertEqual(len(evidence),3)

    def test_model_only_and_partial_optimizer_rejected_before_certification(self):
        from scripts.supervise_main_training import validate_optimizer_map
        path=self.checkpoint(1)
        meta=self.dcp.FileSystemReader(path).read_metadata()
        del meta.state_dict_metadata['optim.state.weight.exp_avg_sq']
        with self.assertRaisesRegex(ValueError,'optimizer state missing'):
            validate_optimizer_map(meta)
        meta=self.dcp.FileSystemReader(path).read_metadata()
        meta.state_dict_metadata={k:v for k,v in meta.state_dict_metadata.items() if k.startswith('model.')}
        with self.assertRaisesRegex(ValueError,'optimizer state missing'):
            validate_optimizer_map(meta)

    def test_original_3b_optimizer_metadata(self):
        from scripts.supervise_main_training import validate_optimizer_map
        actual=self.dcp.FileSystemReader(Path(PROJECT_ROOT)/'runs/actual_train_validation_v1/3b_validated_checkpoint').read_metadata()
        validate_optimizer_map(actual)

    def test_real_adamw_next_update_exact_after_tiered_copy_and_rng_restore(self):
        torch=self.torch
        from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict
        torch.manual_seed(123)
        model=torch.nn.Linear(4,4);optim=torch.optim.AdamW(model.parameters(),lr=.001)
        def update(m,o):
            x=torch.randn(5,4);loss=m(x).square().mean();o.zero_grad();loss.backward();o.step()
            return loss.detach().clone()
        update(model,optim)
        path=self.root/'training/ckpt_0000250';ms,osd=get_state_dict(model,optim)
        self.dcp.save(dict(model=ms,optim=osd),checkpoint_id=path)
        atomic_json(path/'meta.json',dict(step=250,world_size=8,tokens_per_step=2097152,
                    config=dict(model='3b',max_steps=57000),training_contract=dict(id='fixture_training')))
        rng=torch.get_rng_state()
        for rank in range(8):torch.save(dict(cpu=rng,cuda=torch.tensor([],dtype=torch.uint8)),path/f'rng_rank{rank}.pt')
        (path/'.complete').touch();seal(path,self.root/'checkpoint_certificates/ckpt_0000250.json','fixture_training')
        expected_loss=update(model,optim)
        expected_model={k:v.clone() for k,v in model.state_dict().items()}
        expected_optim=get_state_dict(model,optim)[1]
        for i in range(2,12):self.checkpoint(i)
        self.manager.maintain()
        local=Path(self.policy['local_root'])/'checkpoints/ckpt_0000250'
        resumed=torch.nn.Linear(4,4);ro=torch.optim.AdamW(resumed.parameters(),lr=.001)
        ms,osd=get_state_dict(resumed,ro);state=dict(model=ms,optim=osd)
        self.dcp.load(state,checkpoint_id=local);set_state_dict(resumed,ro,model_state_dict=state['model'],optim_state_dict=state['optim'])
        torch.set_rng_state(torch.load(local/'rng_rank0.pt',weights_only=True)['cpu'])
        self.assertTrue(torch.equal(expected_loss,update(resumed,ro)))
        self.assertTrue(all(torch.equal(expected_model[k],v) for k,v in resumed.state_dict().items()))
        actual=get_state_dict(resumed,ro)[1]
        for name,values in expected_optim['state'].items():
            for key,value in values.items():self.assertTrue(torch.equal(value,actual['state'][name][key]))

    def test_cancellation_and_local_loss_during_partial_retirement_stop_safely(self):
        self.seed()
        self.manager.cancelled=lambda:True
        with self.assertRaises(InterruptedError):self.manager.maintain()
        self.assertEqual(len(list((self.root/'training').glob('ckpt_*'))),11)
        self.manager.cancelled=lambda:False
        def interrupt(point):
            if point=='deleted_file':raise InterruptedError('after first deletion')
        self.manager.fault=interrupt
        with self.assertRaises(InterruptedError):self.manager.maintain()
        shutil.rmtree(self.policy['local_root'])
        self.manager.fault=lambda point:None
        with self.assertRaises(ValueError):self.manager.maintain()
        self.assertEqual(len(list((self.root/'training').glob('ckpt_*'))),10)
        self.assertTrue((self.root/'retention_trash/ckpt_0000250/intent.json').exists())


if __name__ == '__main__':
    unittest.main()
