import unittest
from unittest.mock import patch
from types import SimpleNamespace
from scripts.storage_budget import components,guard
class StorageBudgetTest(unittest.TestCase):
    def policy(self):return dict(parameter_count=10,checkpoint_bytes_per_parameter=12,checkpoint_overhead_factor=1,keep_checkpoints=2,split_journal_factor=.5,other_future_gib=0,reserve_gib=0)
    def test_eos_two_byte_storage_and_peak_checkpoint(self):
        r=components(400,10,2,self.policy(),dict(bytes_per_token=4,split_bytes_per_document=20,text_uncertainty_factor=1))
        self.assertEqual(r['tokens_uint16'],220);self.assertEqual(r['token_staging'],44);self.assertEqual(r['split_database'],200);self.assertEqual(r['split_journal'],100);self.assertEqual(r['checkpoint_peak'],360)
    def test_live_free_space_guard_and_invalid_scenario(self):
        with patch('scripts.storage_budget.shutil.disk_usage',return_value=SimpleNamespace(free=100)):
            self.assertEqual(guard(80,'.',20),100)
            with self.assertRaises(RuntimeError):guard(81,'.',20)
        with self.assertRaises(ValueError):components(1,1,1,self.policy(),dict(bytes_per_token=0,split_bytes_per_document=1,text_uncertainty_factor=1))
