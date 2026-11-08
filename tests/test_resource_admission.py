import unittest
from scripts.resource_admission import GIB,memory_admission


def measured(**changes):
    values=dict(limit=100,current=96,file_cache=80,inactive_file=76,shmem=4,dirty=6,writeback=2,host_available=100,reserve=10)
    values.update(changes);return memory_admission(**{k:v*GIB for k,v in values.items()})


class ResourceAdmissionTest(unittest.TestCase):
    def test_clean_inactive_credit_preserves_hard_limit_and_reserve(self):
        result=measured();self.assertEqual(result['clean_inactive_file_credit_bytes'],68*GIB)
        self.assertEqual(result['headroom_gib'],62);self.assertLessEqual(result['cgroup_available_after_credit_bytes'],100*GIB)

    def test_dirty_shmem_and_host_pressure_never_receive_unavailable_credit(self):
        self.assertEqual(measured(dirty=70,writeback=10)['headroom_gib'],0)
        self.assertEqual(measured(shmem=80)['headroom_gib'],0)
        self.assertEqual(measured(file_cache=0,inactive_file=0)['headroom_gib'],0)
        self.assertEqual(measured(host_available=18)['headroom_gib'],8)
        self.assertEqual(measured(host_available=9)['headroom_gib'],0)
        with self.assertRaisesRegex(ValueError,'Invalid'):memory_admission(limit=1,current=-1,file_cache=0,inactive_file=0,shmem=0,dirty=0,writeback=0,host_available=1,reserve=0)


if __name__=='__main__':unittest.main()
