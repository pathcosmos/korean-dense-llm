"""Actual tiny DCP/retention fixtures; no main data mutation or GPU operation."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import patch

from common.data_contracts import resolve, atomic_json, sha256_file
from scripts.tiered_checkpoint_retention import TieredRetention, make_policy
from scripts.supervise_main_training import seal
from scripts.audit_checkpoint_retention import storage_inventory, Snapshot, ChangedDuringAudit, audit_run, existing_ancestor


class RetentionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch.distributed.checkpoint as dcp
        cls.torch,cls.dcp=torch,dcp
        torch.set_num_threads(2)

    def setUp(self):
        self.nfs=tempfile.TemporaryDirectory(prefix='retention-audit-fixture-',dir=resolve('runs'))
        self.disk=tempfile.TemporaryDirectory(prefix='retention-audit-fixture-')
        self.root=Path(self.nfs.name);(self.root/'training').mkdir();(self.root/'checkpoint_certificates').mkdir()
        self.parent={'id':'fixture_training'}
        self.config=dict(model='3b',max_steps=57000,global_tokens=2097152,ckpt_every=5000,out_dir=str(self.root/'training'))
        atomic_json(self.root/'training/training_contract.json',self.parent)
        self.policy=make_policy(self.root,Path(self.disk.name),1024*1024,run_id=uuid.uuid4().hex)
        self.policy.update(reserve_bytes=0,temporary_budget_bytes=0)
        self.manager=TieredRetention(self.root,self.policy,authorized=True)

    def tearDown(self):
        self.nfs.cleanup();self.disk.cleanup()

    def checkpoint(self,step):
        torch,dcp=self.torch,self.dcp
        path=self.root/'training'/f'ckpt_{step:07d}'
        dcp.save({'model':{'weight':torch.arange(16.)+step},'optim':{'state':{'weight':{
            'step':torch.tensor(float(step)),'exp_avg':torch.arange(16.)/10,'exp_avg_sq':torch.arange(16.)/100}}}},checkpoint_id=path)
        atomic_json(path/'meta.json',dict(step=step,world_size=8,tokens_per_step=2097152,config=self.config,training_contract=self.parent))
        for rank in range(8):
            torch.save(dict(cpu=torch.Generator().manual_seed(step+rank).get_state(),cuda=torch.tensor([],dtype=torch.uint8)),path/f'rng_rank{rank}.pt')
        (path/'.complete').touch()
        seal(path,self.root/'checkpoint_certificates'/(path.name+'.json'),self.parent['id'])
        self.manager.maintain()

    def seed(self,full25=False):
        if full25:self.config['ckpt_every']=2000
        for step in list(range(self.config['ckpt_every'],57000,self.config['ckpt_every']))+[57000]:
            self.checkpoint(step)

    def audit(self,**kwargs):
        return storage_inventory(self.root,self.policy,self.parent,self.config,production=False,**kwargs)

    def test_25_distinct_certified_copies_fresh_hash_and_no_mutation(self):
        self.seed(full25=True)
        before={str(p):sha256_file(p) for base in [self.root,Path(self.policy['local_root'])] for p in base.rglob('*') if p.is_file()}
        result=self.audit(hashes=True,final=True)
        self.assertEqual(result['status'],'passed_fixture_retention_content')
        self.assertEqual(len(result['nfs']),10);self.assertEqual(len(result['local']),15)
        self.assertEqual(result['distinct_checkpoints'],25)
        self.assertFalse(result['issues'] or result['overlap'] or result['residues'])
        self.assertTrue(all(r['rng_ranks']==8 and r['state_tensors']==4 for r in result['nfs'].values()))
        after={str(p):sha256_file(p) for base in [self.root,Path(self.policy['local_root'])] for p in base.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_missing_local_namespace_is_not_an_nfs_resume_source(self):
        self.seed()
        shutil.rmtree(self.policy['local_root'])
        result=self.audit(hashes=True,final=True)
        self.assertTrue(result['local_namespace_missing']);self.assertTrue(result['local_history_optional_loss'])
        self.assertEqual(len(result['nfs_resume_candidates']),10);self.assertEqual(result['local'],{})
        self.assertFalse(Path(self.policy['local_root']).exists())

    def test_recorded_whole_local_loss_and_reinitialization(self):
        # Lose the early archive, then archive only the next eligible source.
        for step in range(5000,57000,5000):self.checkpoint(step)
        shutil.rmtree(self.policy['local_root'])
        self.checkpoint(57000)
        result=self.audit(hashes=True,final=True)
        self.assertEqual(result['journal']['history_loss_epochs'],1)
        self.assertEqual(sorted(result['local']),['ckpt_0010000'])

    def test_same_size_payload_corruption_and_owner_mismatch_rejected(self):
        self.seed();local=Path(self.policy['local_root'])
        target=next((local/'checkpoints').glob('*/__*.distcp'))
        raw=target.read_bytes();target.write_bytes(bytes([raw[0]^1])+raw[1:])
        with self.assertRaisesRegex(ValueError,'SHA'):
            self.audit(hashes=True,final=True)
        target.write_bytes(raw)
        atomic_json(local/'.retention_owner.json',dict(wrong='owner'))
        with self.assertRaisesRegex(ValueError,'owner'):
            self.audit(hashes=True,final=True)

    def test_incomplete_pending_and_retirement_residue_not_counted(self):
        self.seed()
        local=Path(self.policy['local_root']);pending=local/'staging/ckpt_0005000';pending.mkdir();(pending/'partial').write_bytes(b'partial')
        result=self.audit();self.assertEqual(result['distinct_checkpoints'],12);self.assertEqual(len(result['residues']),1)
        with self.assertRaisesRegex(ValueError,'residues'):
            self.audit(hashes=True,final=True)
        shutil.rmtree(pending)
        missing=local/'certificates/ckpt_0005000.json';data=missing.read_bytes();missing.unlink()
        result=self.audit();self.assertEqual(len(result['local']),1)
        self.assertFalse(missing.exists())  # Read-only audit must not repair this ACK.
        with self.assertRaisesRegex(ValueError,'residues'):
            self.audit(hashes=True,final=True)
        missing.write_bytes(data)

    def test_retirement_journal_ack_coverage_and_changed_snapshot(self):
        self.seed();path=self.root/'retention_journal.jsonl';raw=path.read_text()
        rows=[json.loads(s) for s in raw.splitlines()]
        for row in rows:
            if row['event']=='local_published':row['certificate_sha256']='0'*64;break
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'publication'):
            self.audit(hashes=True,final=True)
        path.write_text(raw)
        snap=Snapshot();snap.read(path,lines=True);path.write_text(raw+'{}\n')
        with self.assertRaises(ChangedDuringAudit):snap.unchanged()
        path.write_text(raw[:-1])
        with self.assertRaisesRegex(ChangedDuringAudit,'Incomplete'):Snapshot().read(path,lines=True)

    def test_duplicate_and_unexplained_history_hole_rejected(self):
        self.seed();local=Path(self.policy['local_root'])
        shutil.rmtree(local/'checkpoints/ckpt_0005000');(local/'certificates/ckpt_0005000.json').unlink()
        with self.assertRaisesRegex(ValueError,'suffix'):
            self.audit(hashes=True,final=True)
        # Do not accept a local-only final checkpoint as an NFS resume candidate.
        final=self.root/'training/ckpt_0057000';shutil.copytree(final,local/'checkpoints/ckpt_0057000')
        shutil.copy2(self.root/'checkpoint_certificates/ckpt_0057000.json',local/'certificates/ckpt_0057000.json')
        observed=self.audit();self.assertIn('ckpt_0057000',observed['overlap'])
        with self.assertRaisesRegex(ValueError,'duplicates'):
            self.audit(hashes=True,final=True)

    def test_actual_main_marker_without_target_cannot_pass_final_gate(self):
        with patch('scripts.evaluate_final_main.completed_main',side_effect=ValueError('Incomplete step/rank coverage')):
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                audit_run('runs/3b_full_v1',final=True)

    def test_storage_ancestor_does_not_recreate_lost_container_directories(self):
        path=Path(self.disk.name)/'lost_container/scratch/history'
        self.assertEqual(existing_ancestor(path),Path(self.disk.name))
        self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
