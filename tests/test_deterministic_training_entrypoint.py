import json
import os
from pathlib import Path
import tempfile
import unittest
import torch
from train import train as trainer
from train.runtime import validate_resume
from train.deterministic_train import configure
from scripts.benchmark_deterministic_training import repeat_numerics


class DeterministicTrainingTest(unittest.TestCase):
    def test_determinism_binds_resume_and_configuration_is_idempotent(self):
        original=trainer.training_contract;enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark=torch.backends.cudnn.benchmark;deterministic=torch.backends.cudnn.deterministic;workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=root/'data.bin';data.write_bytes(b'\0\0'*32)
                mix=root/'mix.json';mix.write_text(json.dumps(dict(sources=dict(s=dict(weight=1,files=[str(data)])))))
                config=dict(trainer.DEFAULT,mix=str(mix));plain=original(config,8,[])
                configure();wrapped=trainer.training_contract;bound=wrapped(config,8,[])
                self.assertTrue(torch.are_deterministic_algorithms_enabled());self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
                self.assertTrue(bound['execution']['deterministic_algorithms']);self.assertNotEqual(plain['id'],bound['id'])
                with self.assertRaisesRegex(ValueError,'identity mismatch'):validate_resume(dict(training_contract=bound),plain)
                configure();self.assertIs(trainer.training_contract,wrapped)
                self.assertEqual(bound,wrapped(dict(config,stop_after=4,out_dir='changed-operational-path'),8,[]))
                torch.use_deterministic_algorithms(False)
                with self.assertRaisesRegex(ValueError,'identity mismatch'):validate_resume(dict(training_contract=bound),wrapped(config,8,[]))
        finally:
            trainer.training_contract=original;torch.use_deterministic_algorithms(enabled,warn_only=warn)
            torch.backends.cudnn.benchmark=benchmark;torch.backends.cudnn.deterministic=deterministic
            if workspace is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
            else:os.environ['CUBLAS_WORKSPACE_CONFIG']=workspace

    def test_independent_repeats_must_match_numerics_not_only_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';rows=[dict(step=i,loss=10-i/10,gnorm=2.) for i in range(1,9)]
            a.write_text(''.join(json.dumps(r)+'\n' for r in rows));b.write_text(a.read_text())
            self.assertTrue(repeat_numerics(a,b)['passed'])
            rows[-1]['loss']+=.01;b.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertFalse(repeat_numerics(a,b)['passed'])
            b.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'):repeat_numerics(a,b)


if __name__=='__main__':unittest.main()
