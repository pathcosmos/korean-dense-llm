"""Actual CPU checkpoint fixtures for the approved3-NFS adaptive policy."""
import json
from pathlib import Path
import unittest

import test_tiered_checkpoint_retention as fixtures
import test_checkpoint_retention_audit as audit_fixtures
from scripts.audit_checkpoint_retention import storage_inventory
from scripts.retention_capacity import enforce_capacity
from scripts.tiered_checkpoint_retention import TieredRetention, size_tree


class Retention3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TieredRetentionTest.setUpClass()

    def setUp(self):
        self.f = fixtures.TieredRetentionTest('test_full_10_plus_15_distinct_and_full_payload_restores')
        self.f.setUp()
        self.f.policy.update(version=2, nfs_keep=3, adaptive_local_pruning=True)
        self.f.manager = TieredRetention(self.f.root, self.f.policy, authorized=True)

    def tearDown(self):
        self.f.tearDown()

    def test_three_nfs_fifteen_local_and_oldest_retired(self):
        for i in range(1, 23):
            self.f.checkpoint(i)
            result = self.f.manager.maintain()
        self.assertEqual(result['nfs_checkpoints'], [f'ckpt_{i*250:07d}' for i in range(20, 23)])
        self.assertEqual(result['local_checkpoints'], [f'ckpt_{i*250:07d}' for i in range(5, 20)])
        self.assertEqual(result['distinct_checkpoints'], 18)
        self.assertFalse(set(result['nfs_checkpoints']) & set(result['local_checkpoints']))

    def test_byte_budget_evicts_only_oldest_local_before_incoming_copy(self):
        for i in range(1, 9):
            self.f.checkpoint(i)
            self.f.manager.maintain()
        local = self.f.manager.local
        first = local/'checkpoints/ckpt_0000250'
        next_oldest = local/'checkpoints/ckpt_0000500'
        self.f.policy['local_budget_bytes'] = size_tree(local) - size_tree(first)
        enforce_capacity(self.f.manager)
        self.assertFalse(first.exists())
        self.assertTrue(next_oldest.exists())
        self.assertEqual(len(self.f.manager.source_rows()[0]), 3)
        journal = [json.loads(s) for s in (self.f.root/'retention_journal.jsonl').read_text().splitlines()]
        retired = [r['checkpoint'] for r in journal if r['event']=='retire_completed' and r['tier']=='local']
        self.assertEqual(retired, ['ckpt_0000250'])

    def test_insufficient_space_without_local_candidates_preserves_all_nfs(self):
        for i in range(1, 5):
            self.f.checkpoint(i)
        self.f.policy['local_budget_bytes'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Local capacity unavailable'):
            self.f.manager.maintain()
        self.assertEqual(len(self.f.manager.source_rows()[0]), 4)
        self.assertFalse(list((self.f.manager.local/'checkpoints').iterdir()))

    def test_wrong_policy_revision_is_refused(self):
        self.f.policy['version'] = 1
        with self.assertRaises(ValueError):
            TieredRetention(self.f.root, self.f.policy, authorized=True)

    def test_fresh_content_audit_accepts_only_journaled_adaptive_history(self):
        audit_fixtures.RetentionAuditTests.setUpClass()
        f=audit_fixtures.RetentionAuditTests('test_25_distinct_certified_copies_fresh_hash_and_no_mutation')
        f.setUp()
        try:
            f.policy.update(version=2,nfs_keep=3,adaptive_local_pruning=True)
            f.manager=TieredRetention(f.root,f.policy,authorized=True)
            for step in (5000,10000,15000,20000,25000,30000):f.checkpoint(step)
            oldest=f.manager.local/'checkpoints/ckpt_0005000'
            f.policy['local_budget_bytes']=size_tree(f.manager.local)-size_tree(oldest)
            f.manager.maintain()
            result=storage_inventory(f.root,f.policy,f.parent,f.config,hashes=True,production=False)
            self.assertFalse(result['issues'])
            self.assertEqual(len(result['nfs']),3)
            self.assertEqual(len(result['local']),2)
        finally:f.tearDown()
