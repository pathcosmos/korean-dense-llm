import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from common.data_contracts import sha256_file
from data.split_manifest import SplitSelection
from data.verified_split_context import VerifiedSplitContext
from data.split_inputs import SplitInputs,prepare,verify_prepared
from tokenizer.train_tokenizer import train_prepared
from test_split_inputs import fixture


class SharedSplitInputsTest(unittest.TestCase):
    def test_one_fresh_verification_for_prepare_check_and_train(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,split=fixture(root)
            with patch('data.verified_split_context.SplitSelection',wraps=SplitSelection) as checks,patch('data.verified_split_context.sha256_file',wraps=sha256_file) as hashed:
                with VerifiedSplitContext(sources,split) as context:
                    inputs=SplitInputs(sources,split,verified_context=context)
                    self.assertTrue(context.selection.db.in_transaction)
                    train=prepare(sources,split,root/'train',dict(ko=120,en=120,code=120),mode='train',max_chars=20,verified_inputs=inputs)
                    hold=prepare(sources,split,root/'eval',dict(ko=120,en=120),mode='eval',categories=dict(ko='ko',en='en'),verified_inputs=inputs)
                    self.assertEqual(verify_prepared(root/'train/manifest.json','train',verified_inputs=inputs),train)
                    self.assertEqual(verify_prepared(root/'eval/manifest.json',verified_inputs=inputs),hold)
                    train_prepared(root/'train/manifest.json',root/'tokenizer.json',272,verified_inputs=inputs)
                    inputs.close();context.selection.db.execute('SELECT 1')
                self.assertEqual(checks.call_count,1)
                self.assertEqual(sum(str(c.args[0]).endswith('.parquet') for c in hashed.call_args_list),3)
            other=prepare(sources,split,root/'independent',dict(ko=120,en=120,code=120),mode='train',max_chars=20)
            self.assertEqual(other['samples']['sha256'],train['samples']['sha256'])

    def test_cross_path_closed_and_changed_context_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,split=fixture(root);context=VerifiedSplitContext(sources,split)
            with self.assertRaisesRegex(ValueError,'paths differ'):SplitInputs(root/'other.json',split,verified_context=context)
            inputs=SplitInputs(sources,split,verified_context=context)
            original=sources.read_text();sources.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'changed'):prepare(sources,split,root/'bad',dict(ko=10),mode='train',verified_inputs=inputs)
            sources.write_text(original);context.close()
            with self.assertRaises(sqlite3.ProgrammingError):prepare(sources,split,root/'closed',dict(ko=10),mode='train',verified_inputs=inputs)
