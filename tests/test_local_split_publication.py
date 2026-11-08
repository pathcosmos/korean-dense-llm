import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from common.data_contracts import sha256_file
from data.split_manifest import build, SplitSelection
from scripts.publish_local_split import publish
from test_split_manifest import fixture
from scripts.run_local_split import violation


class LocalSplitPublicationTest(unittest.TestCase):
    def params(self, root, manifest, name):
        return dict(source_manifest=str(manifest), output_dir=str(root/name/'published'),
                    run_dir=str(root/name/'run'), scratch_dir=str(root/name/'scratch'),
                    eval_fraction=.4, seed=7, sqlite_cache_mib=64,
                    reserve_gib=0, local_budget_gib=.01, durable_budget_gib=.01)

    def rows(self, path):
        reader = SplitSelection(path/'manifest.json')
        try:
            return reader.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall()
        finally:
            reader.close()

    def test_process_kill_at_copy_and_publish_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = fixture(root)
            build(manifest, root/'baseline', .4, 7)
            expected = self.rows(root/'baseline')
            code = '''import json,os,sys
from scripts.publish_local_split import publish
def crash(phase):
    if phase==sys.argv[2]:os._exit(73)
publish(**json.loads(sys.argv[1]),_checkpoint=crash)
'''
            for phase in ['ingest_pending', 'ingest', 'local_verified', 'copy_chunk',
                          'copy_verified', 'staging_verified', 'before_rename',
                          'renamed', 'before_completion']:
                with self.subTest(phase=phase):
                    kwargs = self.params(root, manifest, phase)
                    proc = subprocess.run([sys.executable, '-c', code, json.dumps(kwargs), phase],
                                          capture_output=True, text=True, timeout=30)
                    self.assertEqual(proc.returncode, 73, proc.stderr)
                    run, out = Path(kwargs['run_dir']), Path(kwargs['output_dir'])
                    self.assertFalse((run/'.done').exists())
                    result = publish(**kwargs)
                    self.assertEqual(result['status'], 'passed')
                    self.assertEqual(self.rows(out), expected)
                    self.assertEqual(json.loads((run/'.done').read_text())['report_sha256'],
                                     sha256_file(run/'report.json'))
                    self.assertEqual(publish(**kwargs)['counts'], result['counts'])

    def test_scratch_loss_before_and_after_durable_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = fixture(root)
            for phase in ['ingest', 'copy_chunk', 'staging_verified', 'before_rename', 'renamed']:
                with self.subTest(phase=phase):
                    kwargs = self.params(root, manifest, phase)
                    def stop(observed):
                        if phase == observed:
                            raise RuntimeError('injected')
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        publish(**kwargs, _checkpoint=stop)
                    shutil.rmtree(kwargs['scratch_dir'])  # disposable synthetic fixture only
                    kwargs['scratch_dir'] += '_new_container'
                    result = publish(**kwargs)
                    self.assertEqual(sum(result['counts'].values()), 56)
                    if phase in ['staging_verified', 'before_rename', 'renamed']:
                        self.assertEqual(list(Path(kwargs['scratch_dir']).iterdir()), [])

    def test_monitor_budget_reasons(self):
        limits = dict(max_seconds=10, max_rss_gib=1, local_budget_gib=2,
                      durable_budget_gib=3, reserve_gib=4)
        gib = 1024**3
        self.assertIsNone(violation(1, 1, 1, 1, 5*gib, 5*gib, limits))
        self.assertEqual(violation(11, 1, 1, 1, 5*gib, 5*gib, limits), 'time_budget')
        self.assertEqual(violation(1, 2*gib, 1, 1, 5*gib, 5*gib, limits), 'rss_budget')
        self.assertEqual(violation(1, 1, 3*gib, 1, 5*gib, 5*gib, limits), 'local_disk_budget')
        self.assertEqual(violation(1, 1, 1, 4*gib, 5*gib, 5*gib, limits), 'durable_disk_budget')
        self.assertEqual(violation(1, 1, 1, 1, 5*gib, 3*gib, limits), 'disk_reserve')

    def test_contract_collision_corruption_and_budget_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = fixture(root)
            kwargs = self.params(root, manifest, 'case')
            publish(**kwargs)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                publish(**dict(kwargs, seed=8))
            different = self.params(root, manifest, 'other')
            different['output_dir'] = kwargs['output_dir']
            with self.assertRaisesRegex(ValueError, 'not owned'):
                publish(**different)
            db = Path(kwargs['output_dir'])/'split.sqlite'
            with db.open('r+b') as stream:
                stream.seek(200)
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                publish(**kwargs)
            small = self.params(root, manifest, 'budget')
            small['reserve_gib'] = 10**10
            with self.assertRaisesRegex(RuntimeError, 'space'):
                publish(**small)
            self.assertFalse(Path(small['output_dir']).exists())


if __name__ == '__main__':
    unittest.main()
