import json
import os
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.data_contracts import atomic_json
from scripts.process_accounting import parse_rollup,proportional_tree
from scripts.supervise_main_training import require_authorization,memory_violation,run,seal,prepare


class MainSupervisionTest(unittest.TestCase):
    def test_bundle_is_not_prepared_from_missing_actual_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'main_bundle'
            with self.assertRaises(FileNotFoundError):
                prepare(Namespace(review_run=str(root/'missing_review'),keep_checkpoints=2),output)
            self.assertFalse(output.exists())

    def test_unauthorized_run_never_launches_and_archive_policy_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'contract.json',dict(purpose='main_training',main_training_authorized=False))
            with patch('subprocess.Popen',side_effect=AssertionError('must not launch')):
                with self.assertRaisesRegex(ValueError,'not authorized'):run(root)
            self.assertFalse((root/'training').exists())
        valid=dict(purpose='main_training',main_training_authorized=True,new_checkpoint_pruning_authorized=True,
                   retention_policy='local_latest',archive_required_before_prune=False)
        require_authorization(valid)
        with self.assertRaisesRegex(ValueError,'retention is not authorized'):
            require_authorization(dict(valid,new_checkpoint_pruning_authorized=False))
        with self.assertRaisesRegex(ValueError,'archive-ack'):
            require_authorization(dict(valid,archive_required_before_prune=True))

    def test_proportional_memory_avoids_shared_rss_false_positive(self):
        gib=2**30
        self.assertFalse(memory_violation(800*gib,250*gib,384*gib))
        self.assertTrue(memory_violation(800*gib,385*gib,384*gib))
        with self.assertRaises(ValueError):memory_violation(10,11,100)
        values=parse_rollup('Rss: 100 kB\nPss: 60 kB\nPss_Anon: 20 kB\nPss_File: 30 kB\nPss_Shmem: 10 kB\n')
        self.assertEqual(values['Pss'],60*1024)
        with self.assertRaises(ValueError):parse_rollup('Rss: 100 kB\n')
        actual=proportional_tree(os.getpid())['totals_bytes']
        self.assertGreater(actual['Pss'],0);self.assertLessEqual(actual['Pss'],actual['Rss'])

    def test_main_checkpoint_actual_tiny_dcp_map_and_content_reverification(self):
        import torch
        import torch.distributed.checkpoint as dcp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/'ckpt_0000250'
            dcp.save({'model':{'weight':torch.arange(4.)}},checkpoint_id=ck)
            atomic_json(ck/'meta.json',dict(step=250,world_size=8,tokens_per_step=2097152,
                        config=dict(model='3b',max_steps=57000),training_contract=dict(id='fixture')))
            (ck/'.complete').touch()
            for r in range(8):(ck/f'rng_rank{r}.pt').write_bytes(bytes([r]))
            proof=root/'certificate.json';first=seal(ck,proof,'fixture')
            self.assertEqual(first,seal(ck,proof,'fixture'))
            # Same-length payload mutation is detected against the original seal.
            p=next(ck.glob('*.distcp'));value=p.read_bytes();p.write_bytes(bytes([value[0]^1])+value[1:])
            with self.assertRaisesRegex(ValueError,'changed'):seal(ck,proof,'fixture')
            p.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'truncated'):seal(ck,proof,'fixture')


if __name__=='__main__':unittest.main()
