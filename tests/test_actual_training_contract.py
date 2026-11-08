import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from common.data_contracts import sha256_file
from scripts.validate_actual_training import compare_trajectories,publish_checkpoint,validation_geometry


class ActualTrainingContractTest(unittest.TestCase):
    def test_selected_batch_geometry_and_fault_after_one_update(self):
        from scripts.validate_distributed import loss_fault_after_microbatches
        selected=validation_geometry(dict(**{'1b':dict(micro_bsz=16,global_tokens=2097152),'3b':dict(micro_bsz=8,global_tokens=2097152)}))
        self.assertEqual(loss_fault_after_microbatches(dict(seq_len=4096,**selected['1b']),8),5)
        self.assertEqual(loss_fault_after_microbatches(dict(seq_len=4096,**selected['3b']),8),9)
        self.assertEqual(loss_fault_after_microbatches(dict(seq_len=32,micro_bsz=1,global_tokens=512),8),3)
        self.assertEqual(selected['1b']['lr'],.0006);self.assertEqual(selected['3b']['lr'],.0003)
        for settings in [{'1b':dict(max_steps=5000)},{'3b':dict(micro_bsz=3)},{'1b':dict(global_tokens=4194304)},{'tiny':{}}]:
            with self.assertRaises(ValueError):validation_geometry(settings)
        for lr in [0,-.001,.002,float('nan'),True]:
            with self.assertRaises(ValueError):validation_geometry({'1b':dict(lr=lr)})

    def test_four_step_actual_trajectory_finite_resume_and_schedule(self):
        with tempfile.TemporaryDirectory() as d,patch('scripts.validate_actual_training.shutil.disk_usage',return_value=SimpleNamespace(free=1000*2**30)):
            root=Path(d);a=root/'a.jsonl';b=root/'b.jsonl'
            rows=[dict(step=i,loss=11-i*.1,lr=.0003,gnorm=1,tok_s=1000,mfu=.1) for i in range(1,5)]
            def write(path,items):path.write_text(''.join(json.dumps(r)+'\n' for r in items))
            write(a,rows);write(b,rows);self.assertEqual(compare_trajectories(a,b)['loss_max_abs'],0)
            for key,value in [('step',2),('loss',11),('gnorm',float('nan')),('lr',.001)]:
                bad=[dict(r) for r in rows];bad[0][key]=value;write(b,bad)
                with self.assertRaises(ValueError):compare_trajectories(a,b)
            write(b,rows[:3])
            with self.assertRaisesRegex(ValueError,'four'):compare_trajectories(a,b)

    def test_checkpoint_publication_fresh_hashes_preserve_source_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d,patch('scripts.validate_actual_training.shutil.disk_usage',return_value=SimpleNamespace(free=1000*2**30)):
            root=Path(d);source=root/'local';source.mkdir();(source/'weights').write_bytes(b'checkpoint fixture')
            with self.assertRaisesRegex(ValueError,'Incomplete'):publish_checkpoint(source,root/'final')
            (source/'.complete').touch();before={p.name:sha256_file(p) for p in source.iterdir()}
            with patch('scripts.validate_actual_training.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(RuntimeError,'reserve'):publish_checkpoint(source,root/'final')
            result=publish_checkpoint(source,root/'final')
            self.assertEqual(result['bytes'],len(b'checkpoint fixture'))
            publication=json.loads((root/'final/publication.json').read_text())
            for r in publication['files']:
                self.assertEqual(sha256_file(root/'final'/r['path']),r['sha256'])
            self.assertEqual(before,{p.name:sha256_file(p) for p in source.iterdir()})
            with self.assertRaisesRegex(ValueError,'overwrite'):publish_checkpoint(source,root/'final')
            (root/'other.publishing').mkdir()
            with self.assertRaisesRegex(ValueError,'interrupted'):publish_checkpoint(source,root/'other')
