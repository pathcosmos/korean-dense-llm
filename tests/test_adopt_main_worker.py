"""Real CPU process handoff identities; no training/GPU process changes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.adopt_main_worker import AdoptedWorker, stamp, gpu_lock


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='adoption-fixture-')
        self.root = Path(self.temp.name)
        code = "import subprocess,sys,time; children=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)','train.monitored',sys.argv[1]]) for _ in range(8)]; time.sleep(60)"
        self.proc = subprocess.Popen([sys.executable, '-c', code, str(self.root), 'train.monitored',
                                      '--attempt', 'fixture-adopt'], start_new_session=True)
        end=time.monotonic()+10
        children=[]
        while len(children)!=8 and time.monotonic()<end:
            children=[int(x) for x in Path(f'/proc/{self.proc.pid}/task/{self.proc.pid}/children').read_text().split()]
            time.sleep(.01)
        self.assertEqual(len(children),8)
        self.ranks={str(pid):stamp(pid)[0] for pid in children}
        self.receipt={'worker':{'pid':self.proc.pid,'start_ticks':stamp(self.proc.pid)[0]},'ranks':self.ranks}

    def tearDown(self):
        import os,signal
        try:os.killpg(self.proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        self.proc.wait(timeout=5)
        self.temp.cleanup()

    def test_adopts_same_live_eight_processes_and_refuses_pid_mismatch(self):
        worker=AdoptedWorker(self.proc.pid,self.root,self.receipt)
        self.assertIsNone(worker.poll())
        self.assertEqual(worker.ranks,self.ranks)
        self.receipt['worker']['start_ticks']='wrong'
        with self.assertRaises(ValueError):AdoptedWorker(self.proc.pid,self.root,self.receipt)

    def test_gpu_lock_allows_only_owned_processes(self):
        worker=AdoptedWorker(self.proc.pid,self.root,self.receipt)
        lockpath=self.root/'gpu.lock'
        output='\n'.join(self.ranks)
        with patch('common.data_contracts.resolve',return_value=lockpath),patch('subprocess.check_output',return_value=output):
            with gpu_lock(worker):self.assertIsNone(worker.poll())
        with patch('common.data_contracts.resolve',return_value=lockpath),patch('subprocess.check_output',return_value=output+'\n999999'):
            with self.assertRaises(ValueError):gpu_lock(worker)

    def test_terminal_requires_all_rank_success_and_final_seal(self):
        worker=AdoptedWorker(self.proc.pid,self.root,self.receipt)
        self.proc.terminate();self.proc.wait(timeout=5)
        self.assertEqual(worker.poll(),1)
        worker.returncode=None
        cp=self.root/'training/ckpt_0057000';cp.mkdir(parents=True);(cp/'.complete').touch()
        ledgers=self.root/'ledgers';ledgers.mkdir()
        for rank in range(8):(ledgers/f'fixture-adopt-rank{rank}.json').write_text(json.dumps({'success':True,'checkpoint_step':57000}))
        self.assertEqual(worker.poll(),0)

    def test_failed_parent_still_cleans_only_original_rank_identities(self):
        worker=AdoptedWorker(self.proc.pid,self.root,self.receipt)
        self.proc.terminate();self.proc.wait(timeout=5)
        worker.stop_owned()
        for pid in self.ranks:
            try:stamp(int(pid))
            except (FileNotFoundError,ProcessLookupError):continue
            self.fail('Original rank survived bounded cleanup')
