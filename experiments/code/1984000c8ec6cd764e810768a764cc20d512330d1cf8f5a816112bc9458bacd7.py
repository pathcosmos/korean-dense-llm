import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.data_contracts import atomic_json, resolve
from scripts.run_final_resume_probe import (GIB, budget_check, supervise_command, owned_processes,
    reject_previous_execution, require_comparison, materialize_spec, admission, load_plan, make_contract)


class FinalResumeControllerTests(unittest.TestCase):
    def budgets(self):
        plan = dict(durable_budget_gib=1, local_budget_gib=192, max_pss_gib=256)
        policy = dict(nfs_budget_bytes=437*GIB,local_budget_bytes=635*GIB,temporary_budget_bytes=1024*GIB,
                      other_nfs_future_bytes=0,reserve_bytes=200*GIB)
        values = dict(nfs_free=936*GIB,local_free=1888*GIB,main_used=0,history_used=0,temp_used=0,
                      own_used=0,scratch_used=0,native_nfs_used=0,native_temp_used=0,headroom_gib=1700)
        return plan, policy, values

    def test_subdivided_temporary_and_combined_memory_reservations(self):
        plan, policy, values = self.budgets()
        result = budget_check(plan,policy,**values)
        self.assertEqual(result['projected_temporary_bytes'],200*GIB)
        self.assertEqual(result['local_after_budget_reserve_bytes'],29*GIB)
        self.assertEqual(result['nfs_after_budget_reserve_bytes'],290*GIB)
        self.assertEqual(result['combined_memory_gib'],896)
        for key,value,message in [('local_free',1850*GIB,'local history'),('nfs_free',600*GIB,'NFS'),
                                  ('headroom_gib',895,'memory'),('temp_used',900*GIB,'residues'),
                                  ('scratch_used',193*GIB,'cap'),('own_used',2*GIB,'cap')]:
            bad = dict(values, **{key:value})
            with self.assertRaisesRegex(RuntimeError,message):
                budget_check(plan,policy,**bad)
        # Actual bytes already on disk reduce free and the remaining reservation
        # by the same amount. Do not charge the same checkpoint twice.
        changed = dict(values,local_free=values['local_free']-100*GIB,temp_used=100*GIB,scratch_used=100*GIB)
        self.assertEqual(budget_check(plan,policy,**changed)['local_after_budget_reserve_bytes'],29*GIB)

    def test_actual_nonzero_process_preserves_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('partial retained');sys.exit(7)"
            seen = []
            with self.assertRaisesRegex(RuntimeError,'exited 7'):
                supervise_command([sys.executable,'-c',code,str(root/'partial')],root=root,env=os.environ.copy(),
                                  guard=lambda pid,elapsed,peak:seen.append((pid,peak)),interval=.02)
            self.assertTrue(seen)
            self.assertEqual((root/'partial').read_text(),'partial retained')
            self.assertFalse((root/'.done').exists())
            self.assertTrue((root/'worker_console.log').exists())

    def test_budget_stop_kills_only_own_session_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            try:
                def guard(pid, elapsed, peak):
                    if elapsed > .1:
                        raise RuntimeError('fixture budget exceeded')
                with self.assertRaisesRegex(RuntimeError,'fixture budget'):
                    supervise_command([sys.executable,'-c','import time;time.sleep(30)'],root=root,env=os.environ.copy(),guard=guard,interval=.02)
                pid = json.loads((root/'worker_handle.json').read_text())['pid']
                self.assertFalse(Path(f'/proc/{pid}').exists())
                self.assertIsNone(other.poll())
                self.assertFalse((root/'.done').exists())
            finally:
                other.terminate(); other.wait(timeout=5)

    def test_orphan_is_an_actual_exact_handle_and_residue_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
                                      'train.final_resume_probe','--spec',str(root/'spec.json')])
            try:
                self.assertIn(child.pid,owned_processes(root))
                self.assertNotIn(child.pid,owned_processes(Path(str(root)+'_other')))
                with self.assertRaisesRegex(RuntimeError,'actual diagnostic process'):
                    reject_previous_execution(root)
            finally:
                child.terminate();child.wait(timeout=5)
            self.assertEqual(owned_processes(root),[])
            (root/'execution_started.json').write_text('{}')
            with patch('scripts.run_final_resume_probe.load_plan',return_value={'scratch':str(root/'scratch')}):
                with self.assertRaisesRegex(RuntimeError,'Preserve previous'):
                    reject_previous_execution(root)
            self.assertTrue((root/'execution_started.json').exists())

    def test_completion_marker_cannot_materialize_failed_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('scripts.run_final_resume_probe.completed_main',side_effect=ValueError('Incomplete step/rank coverage')):
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    materialize_spec(root,{'main_run':str(root/'fake_main')})
            self.assertEqual(list(root.iterdir()),[])
        good = dict(exact=True,tensors=1020,model_tensors=255,optimizer_tensors=765,rng_ranks=8)
        require_comparison(good)
        for key,value in [('exact',False),('model_tensors',254),('optimizer_tensors',764),('rng_ranks',7)]:
            with self.assertRaisesRegex(ValueError,'incomplete'):
                require_comparison(dict(good,**{key:value}))

    def test_current_prepared_plan_bindings_and_worker_tmpdir_admission(self):
        # Read actual prepared immutable bindings; no producer mutation or GPU.
        root = resolve('runs/3b_final_resume_v1')
        plan = load_plan(root)
        contract = make_contract(root)
        self.assertEqual(contract['worker_max_seconds'],3600)
        self.assertEqual(contract['worker_pss_gib'],256)
        from scripts.resource_admission import inspect
        hardware = inspect()
        # hardware_profile suggests a nested path when TMPDIR is explicit. The
        # admission must verify the real filesystem, not mistake that suggestion
        # for the already approved scratch namespace.
        hardware['storage']['scratch_dir'] = str(Path(plan['scratch']).parent/'suggested_nested_tmp')
        with patch('scripts.resource_admission.inspect',return_value=hardware):
            _, value = admission(root,plan)
        self.assertEqual(value['combined_memory_gib'],896)


if __name__ == '__main__':
    unittest.main()
