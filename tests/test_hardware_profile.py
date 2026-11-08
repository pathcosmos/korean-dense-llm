import unittest
from scripts.hardware_profile import effective_cpu,presets
class HardwareProfileTest(unittest.TestCase):
    def test_quota_and_affinity_both_limit_cpu(self):
        self.assertEqual(effective_cpu(96,'max 100000'),96)
        self.assertEqual(effective_cpu(96,'150000 100000'),1.5)
        self.assertEqual(effective_cpu(2,'800000 100000'),2)
    def test_low_resources_do_not_recommend_many_workers_or_memory(self):
        p=presets(1,2);self.assertFalse(p['launch_permitted'])
        for s in p['stages'].values():
            self.assertEqual(s['start_workers'],1);self.assertLessEqual(s['job_memory_budget_gib'],2)
            self.assertEqual(s['env']['OMP_NUM_THREADS'],'1')
    def test_machine_presets_bound_workers_threads_and_preserve_sqlite_single_writer(self):
        p=presets(96,800);self.assertTrue(p['launch_permitted']);self.assertEqual(p['reserved_cpu'],8)
        t=p['stages']['tokenization'];self.assertEqual(t['start_workers'],8);self.assertEqual(t['max_workers'],8);self.assertEqual(t['env']['RAYON_NUM_THREADS'],'8')
        self.assertEqual(p['stages']['sqlite']['max_workers'],1);self.assertEqual(p['stages']['sqlite']['validated_cache_mib'],131072)
        self.assertEqual(p['stages']['sqlite']['hash_workers'],32)
        self.assertEqual(p['stages']['cpu-prep']['start_workers'],16)
        self.assertEqual(p['stages']['tokenizer-train']['env']['RAYON_NUM_THREADS'],'64')
