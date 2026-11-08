import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from common.data_contracts import sha256_file
from data.local_split_reader import prepare_reader
from data.split_inputs import prepare
from eval.prepare_holdout import prepare_holdout,local_verified_inputs
from eval.run import run
from test_eval_runner import fixture as checkpoint_fixture
from test_split_inputs import fixture as source_fixture


class LocalHoldoutEvalTest(unittest.TestCase):
    def test_cli_local_reader_canonical_provenance_same_ppl_and_live_context(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split=source_fixture(root)
            reader=prepare_reader(split,root/'reader',scratch=root/'scratch')
            canonical=prepare(sm,split,root/'canonical',dict(ko=40,en=40),mode='eval',categories=dict(ko='ko',en='en'))
            command=[sys.executable,'eval/prepare_holdout.py','--sources-manifest',str(sm),'--split-manifest',str(split),
                     '--reader-manifest',str(reader),'--sqlite-cache-mib','64','--source','ko:ko:40','--source','en:en:40','--output-dir',str(root/'local')]
            done=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(done.returncode,0,done.stderr)
            local=json.loads((root/'local/manifest.json').read_text())
            self.assertEqual(local['samples']['sha256'],canonical['samples']['sha256'])
            self.assertEqual(local['contract']['split_manifest'],str(split))
            ckpt,tok,_,prompts=checkpoint_fixture(root)
            args=(ckpt,tok,root/'local/samples.jsonl',prompts)
            settings=dict(seq_len=8,stride=4,max_new_tokens=1,holdout_manifest=root/'local/manifest.json')
            direct=run(*args,root/'direct.json',**settings)
            physical=run(*args,root/'physical.json',**settings,reader_manifest=reader,reader_cache_mib=64)
            self.assertEqual(direct['contract_id'],physical['contract_id'])
            self.assertEqual(direct['languages'],physical['languages'])
            self.assertEqual(physical['contract']['input_provenance']['level'],'verified-eval-split')
            with local_verified_inputs(sm,split,reader,cache_mib=64) as inputs:
                borrowed=run(*args,root/'borrowed.json',**settings,verified_inputs=inputs)
                self.assertEqual(borrowed['contract_id'],direct['contract_id'])
                inputs.selection.db.execute('SELECT 1')
            command=[sys.executable,'eval/run.py','--checkpoint',str(ckpt),'--tokenizer',str(tok),'--samples',str(args[2]),'--prompts',str(prompts),
                     '--output',str(root/'cli.json'),'--holdout-manifest',str(root/'local/manifest.json'),'--reader-manifest',str(reader),
                     '--sqlite-cache-mib','64','--seq-len','8','--stride','4','--max-new-tokens','1']
            done=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertEqual(json.loads((root/'cli.json').read_text())['contract_id'],direct['contract_id'])

    def test_reader_never_promotes_manual_or_changed_database(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split=source_fixture(root);reader=prepare_reader(split,root/'reader',scratch=root/'scratch')
            args=checkpoint_fixture(root)
            with self.assertRaisesRegex(ValueError,'holdout manifest'):run(*args,root/'bad.json',reader_manifest=reader)
            original=sha256_file(split)
            db=Path(json.loads(reader.read_text())['database']['path'])
            with db.open('r+b') as stream:stream.seek(1024);byte=stream.read(1);stream.seek(1024);stream.write(bytes([byte[0]^1]))
            with self.assertRaisesRegex(ValueError,'DB content changed'):
                prepare_holdout(sm,split,root/'bad_hold',dict(ko=40,en=40),dict(ko='ko',en='en'),reader_manifest=reader,cache_mib=64)
            self.assertEqual(sha256_file(split),original)
