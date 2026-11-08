from pathlib import Path
import tempfile
import unittest
from common.data_contracts import sha256_file,resolve
from data.local_split_reader import prepare_reader
from data.verified_split_context import VerifiedSplitContext
from data.parallel_tokenize import run
from test_parallel_tokenize import fixture


class TokenPilotResumeTest(unittest.TestCase):
    def test_whole_file_pilot_reused_by_full_native_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            reader=prepare_reader(split,root/'reader',scratch=root/'scratch')
            with VerifiedSplitContext(sm,split,reader_manifest=reader) as context:
                chosen=[records[0]['path'] for records in context.records.values()]
                pilot=run(sm,split,tok,root/'tokens',workers=2,batch_size=7,experimental=True,selected_files=chosen,verified_context=context)
                self.assertEqual(pilot['status'],'pilot_complete');self.assertEqual(pilot['shards'],2);self.assertEqual(pilot['planned_shards'],4)
                for profile in pilot['files']:
                    self.assertFalse(profile['reused']);self.assertGreater(profile['utf8_bytes'],0)
                    self.assertGreater(profile['elapsed_seconds'],0)
                    self.assertGreater(profile['train_docs']+profile['holdout_docs'],0)
                self.assertFalse((root/'tokens/.done').exists())
                for source in context.records:self.assertFalse((root/'tokens'/source/'.done').exists())
                existing={p:(sha256_file(p),p.stat().st_mtime_ns) for p in (root/'tokens').glob('*/*.bin')}
                full=run(sm,split,tok,root/'tokens',workers=1,batch_size=13,experimental=True,verified_context=context)
                self.assertEqual(full['reused'],2);self.assertEqual(full['shards'],4);self.assertTrue((root/'tokens/.done').exists())
                self.assertTrue(all(p['utf8_bytes'] is None for p in full['files'] if p['reused']))
                for p,(h,t) in existing.items():self.assertEqual(sha256_file(p),h);self.assertEqual(p.stat().st_mtime_ns,t)
                context.selection.db.execute('SELECT 1')
            baseline=run(sm,split,tok,root/'clean',workers=2,batch_size=4,experimental=True,reader_manifest=reader)
            self.assertEqual(full['logical_id'],baseline['logical_id'])

    def test_unverified_pilot_file_or_cross_context_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            with VerifiedSplitContext(sm,split) as ctx:
                for chosen in [[],[str(root/'unknown.parquet')]]:
                    with self.assertRaisesRegex(ValueError,'Pilot files'):run(sm,split,tok,root/'bad',experimental=True,selected_files=chosen,verified_context=ctx)
                with self.assertRaisesRegex(ValueError,'context differs'):run(root/'other.json',split,tok,root/'other',experimental=True,verified_context=ctx)
