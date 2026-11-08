import json
from pathlib import Path
import tempfile
import unittest
from common.data_contracts import atomic_json,sha256_file
from data.pilot_mix import build_pilot_mix,verify_pilot_mix
from data.parallel_tokenize import run
from data.verified_split_context import VerifiedSplitContext
from train.data import MixSampler,HoldoutSet
from test_parallel_tokenize import fixture


class PilotMixTest(unittest.TestCase):
    def inputs(self,root):
        sm,split,tok=fixture(root)
        with VerifiedSplitContext(sm,split) as ctx:
            chosen=[r[0]['path'] for r in ctx.records.values()]
            report=run(sm,split,tok,root/'tokens',selected_files=chosen,workers=2,experimental=True,verified_context=ctx)
        return report

    def test_diagnostic_sampler_and_native_full_markers_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.inputs(root)
            before={str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in (root/'tokens').rglob('*') if p.is_file()}
            mix=build_pilot_mix(root/'tokens',r['report_path'],root/'diagnostic',dict(web='ko',code='en'),dict(web=.6,code=.4),seq_len=8,require_production=False)
            self.assertFalse(mix['full_corpus_complete']);self.assertEqual(mix['scope'],'diagnostic_pilot_only')
            sampler=MixSampler(str(root/'diagnostic/mix.json'),8);x,y=sampler.batch(0,0,0,3)
            self.assertEqual(x.shape,(3,8));self.assertTrue((x[:,1:]==y[:,:-1]).all())
            self.assertGreaterEqual(len(HoldoutSet(mix['holdout'],8)),9)
            self.assertEqual(verify_pilot_mix(root/'diagnostic',require_production=False),mix)
            with self.assertRaisesRegex(ValueError,'Experimental'):verify_pilot_mix(root/'diagnostic')
            self.assertEqual(before,{str(p):(sha256_file(p),p.stat().st_mtime_ns) for p in (root/'tokens').rglob('*') if p.is_file()})
            self.assertFalse((root/'tokens/.done').exists())
            self.assertFalse((root/'tokens/web/.done').exists())
            with self.assertRaisesRegex(ValueError,'final tokenizer'):build_pilot_mix(root/'tokens',r['report_path'],root/'bad',dict(web='ko',code='en'),dict(web=.6,code=.4))

    def test_changed_pilot_or_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.inputs(root);pp=Path(r['report_path']);pp=pp if pp.is_absolute() else Path.cwd()/pp
            binary=next((root/'tokens/web').glob('*.bin'));body=binary.read_bytes();binary.write_bytes(bytes([body[0]^1])+body[1:])
            with self.assertRaisesRegex(ValueError,'corrupt'):build_pilot_mix(root/'tokens',pp,root/'bad',dict(web='ko',code='en'),dict(web=.5,code=.5),seq_len=8,require_production=False)
            binary.write_bytes(body);report=json.loads(pp.read_text());report['files'][0]['train_docs']-=1;atomic_json(pp,report)
            atomic_json(str(pp)+'.done',dict(report_sha256=sha256_file(pp),contract_sha256=sha256_file(root/'tokens/contract.json')))
            with self.assertRaisesRegex(ValueError,'counts differ'):build_pilot_mix(root/'tokens',pp,root/'bad2',dict(web='ko',code='en'),dict(web=.5,code=.5),seq_len=8,require_production=False)
