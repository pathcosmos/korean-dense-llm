import unittest
from scripts.watch_data_pipeline import classify, transitions


class PipelineWatchTest(unittest.TestCase):
    def test_retirement_requires_bound_receipt_and_exact_missing_paths(self):
        import json, tempfile
        from pathlib import Path
        from common.data_contracts import sha256_file
        from scripts.watch_data_pipeline import verified_retired_payloads
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'runs/data_eviction_v1'
            run.mkdir(parents=True); payload = root / 'payload'
            self.assertFalse(verified_retired_payloads(root, [payload]))
            manifest = run / 'approved_manifest.json'
            manifest.write_text(json.dumps({'files': [{'path': 'payload'}]}))
            report = run / 'eviction_receipt.json'
            report.write_text(json.dumps({'status': 'completed_approved_payload_eviction',
                'manifest_sha256': sha256_file(manifest)}))
            (run / '.done').write_text(json.dumps({'report_sha256': sha256_file(report)}))
            self.assertTrue(verified_retired_payloads(root, [payload]))
            self.assertFalse(verified_retired_payloads(root, [root / 'unapproved']))
            self.assertFalse(verified_retired_payloads(root, []))
            payload.write_text('restored')
            self.assertFalse(verified_retired_payloads(root, [payload]))
            payload.unlink(); payload.symlink_to(root / 'missing')
            self.assertFalse(verified_retired_payloads(root, [payload]))
            payload.unlink(); manifest.write_text('{}')
            self.assertFalse(verified_retired_payloads(root, [payload]))
            report.write_text('{}')
            self.assertFalse(verified_retired_payloads(root, [payload]))

    def test_completion_and_errors(self):
        self.assertEqual(classify(True, False, ''), 'complete')
        self.assertEqual(classify(False, False, ''), 'unexpected_exit')
        self.assertEqual(classify(False, True, 'Traceback (most recent call last)'), 'error')
        self.assertEqual(classify(False, True, 'WAIT', True), 'waiting')

    def test_emit_changes_not_every_poll(self):
        initial = {'job': dict(state='running', done=1, total=100)}
        self.assertEqual(transitions(initial, initial), [])
        milestone = {'job': dict(state='running', done=25, total=100)}
        self.assertEqual(transitions(initial, milestone)[0]['event'], 'progress_milestone')
        complete = {'job': dict(state='complete', done=100, total=100)}
        self.assertEqual(transitions(milestone, complete)[0]['state'], 'complete')

    def test_registered_audit_completion_pause_and_stall(self):
        import json,os,time,tempfile
        from pathlib import Path
        from scripts.watch_data_pipeline import registered_job
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state.json';pid=root/'pid';log=root/'log'
            pid.write_text(str(os.getpid()));log.write_text('')
            registration=dict(state_file=str(state),pid_file=str(pid),log_file=str(log),run_id='fixture',total=6)
            state.write_text(json.dumps(dict(phase='running',done=1,total=6,processed_rows=20)))
            self.assertEqual(registered_job(registration)['state'],'running')
            os.utime(state,(time.time()-1300,time.time()-1300))
            self.assertEqual(registered_job(registration)['state'],'stalled')
            state.write_text(json.dumps(dict(phase='paused',done=1,total=6)))
            self.assertEqual(registered_job(registration)['state'],'paused')
            state.write_text(json.dumps(dict(phase='complete',done=6,total=6)))
            self.assertEqual(registered_job(registration)['state'],'error')
            (root/'.done').touch();self.assertEqual(registered_job(registration)['state'],'complete')
