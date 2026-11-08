import json,tempfile,unittest,os,time
from pathlib import Path
from scripts.job_registry import observe,load_jobs,validate
from common.data_contracts import atomic_json,sha256_file
class JobRegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.spec=dict(run_id='test',state_file=str(self.p/'state.json'),pid_file=str(self.p/'pid'),log_file=str(self.p/'log'),certificate=str(self.p/'.done'),report_file=str(self.p/'report.json'),proofs=[dict(path=str(self.p/'report.json'),field=['summary_sha256'])],stall_seconds=30)
        (self.p/'pid').write_text('123');(self.p/'log').write_text('');atomic_json(self.p/'state.json',dict(status='running'))
    def tearDown(self):self.tmp.cleanup()
    def obs(self,live=True):return observe(self.spec,lambda pid:live,lambda p:p.read_text() if p.exists() else '')
    def finish(self,status):
        atomic_json(self.p/'report.json',dict(status=status));atomic_json(self.p/'.done',dict(summary_sha256=sha256_file(self.p/'report.json')));atomic_json(self.p/'state.json',dict(status='completed'))
    def test_completion_review_and_tamper(self):
        self.finish('needs_review');self.assertEqual(self.obs(False)['state'],'needs_review')
        self.finish('diagnostic_complete');self.assertEqual(self.obs(False)['state'],'measurement_complete')
        self.finish('passed');self.assertEqual(self.obs(False)['state'],'complete')
        (self.p/'report.json').write_text('{}');self.assertEqual(self.obs()['state'],'error')
    def test_exit_not_success_stall_and_heartbeat(self):
        self.assertEqual(self.obs(False)['state'],'unexpected_exit')
        for p in ['state.json','log']:os.utime(self.p/p,(time.time()-40,time.time()-40))
        self.assertEqual(self.obs()['state'],'stalled');(self.p/'log').write_text('step 2');self.assertEqual(self.obs()['state'],'running')
        atomic_json(self.p/'state.json',dict(status='completed'));self.assertEqual(self.obs()['state'],'error')
    def test_bad_job_isolated_and_unbound_report_rejected(self):
        spec=dict(self.spec,report_file=str(self.p/'other'))
        with self.assertRaises(ValueError):validate(spec)
        atomic_json(self.p/'registry.json',dict(schema_version=1,jobs=[self.spec,dict(run_id='broken')]))
        result=load_jobs(self.p/'registry.json',lambda _:True,lambda _:'');self.assertEqual(result['job:test']['state'],'running');self.assertEqual(result['job:broken']['state'],'error')

    def test_live_watcher_lock_prevents_duplicate_and_errors_keep_monitoring(self):
        import fcntl
        from scripts.job_registry import ensure_watcher,TERMINAL
        (self.p/'logs').mkdir()
        with (self.p/'logs/pipeline_watch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertIsNone(ensure_watcher(self.p))
        self.assertNotIn('error',TERMINAL);self.assertNotIn('paused',TERMINAL)
