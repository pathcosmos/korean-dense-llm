import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from train.train import DEFAULT
from train.runtime import training_contract as serial
from train.parallel_contract import training_contract as parallel,native_expected_inputs
from common.data_contracts import sha256_file


class ParallelTrainingContractTest(unittest.TestCase):
    def test_actual_native_digests_reject_same_size_token_corruption_and_marker_edits(self):
        from data.parallel_tokenize import run
        from test_parallel_tokenize import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sm,split,tok=fixture(root);tokens=root/'tokens';run(sm,split,tok,tokens,workers=2,experimental=True)
            sources={};holds=[];markers=[]
            for directory in sorted(p for p in tokens.iterdir() if p.is_dir()):
                files=[]
                for marker in sorted(directory.glob('*.bin.done')):
                    record=json.loads(marker.read_text());markers.append(marker)
                    if record['tokens']:files.append(str(directory/record['outputs'][0]['name']))
                    if record['holdout_tokens']:holds.append(str(directory/record['outputs'][1]['name']))
                sources[directory.name]=dict(weight=1,files=files)
            mix=dict(sources=sources);mp=root/'mix.json';mp.write_text(json.dumps(mix));config=dict(DEFAULT,mix=str(mp),seq_len=8)
            expected=native_expected_inputs(tokens,mix,holds,require_production=False)
            self.assertEqual(serial(config,8,holds),parallel(config,8,holds,expected_files=expected))
            with self.assertRaisesRegex(ValueError,'production'):native_expected_inputs(tokens,mix,holds)
            target=Path(next(iter(sources.values()))['files'][0]);data=bytearray(target.read_bytes());data[0]^=1;target.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'certified native'):parallel(config,8,holds,expected_files=expected)
            marker=next(p for p in markers if p.with_suffix('')==target);record=json.loads(marker.read_text());record['outputs'][0]['sha256']=sha256_file(target);marker.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'records changed'):native_expected_inputs(tokens,mix,holds,require_production=False)

    def test_fresh_parallel_proof_is_identical_including_duplicate_hold_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.bin';b=root/'b.bin';a.write_bytes(b'\1\0'*4097);b.write_bytes(b'\2\0'*4097)
            mix=root/'mix.json';mix.write_text(json.dumps(dict(sources=dict(a=dict(weight=.3,files=[str(a)]),b=dict(weight=.7,files=[str(b)])))))
            config=dict(DEFAULT,mix=str(mix));metrics=[]
            self.assertEqual(serial(config,8,[str(b)]),parallel(config,8,[str(b)],observe=metrics.append))
            self.assertEqual(metrics[0]['files'],2);self.assertEqual(metrics[0]['bytes'],a.stat().st_size+b.stat().st_size)
            self.assertEqual(parallel(config,8,[]),parallel(dict(config,stop_after=4,out_dir='another',log_every=1),8,[]))
            before=parallel(config,8,[]);a.write_bytes(b'\3\0'*4097)
            self.assertNotEqual(before['id'],parallel(config,8,[])['id']);self.assertEqual(serial(config,8,[]),parallel(config,8,[]))

    def test_input_mutation_during_hashing_is_rejected_before_identity_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'a.bin';data.write_bytes(b'\0\0'*32);mix=root/'mix.json'
            mix.write_text(json.dumps(dict(sources=dict(a=dict(weight=1,files=[str(data)])))))
            def change(path):
                result=sha256_file(path)
                if Path(path)==data:data.write_bytes(data.read_bytes()+b'XX')
                return result
            with patch('train.parallel_contract.sha256_file',side_effect=change),self.assertRaisesRegex(ValueError,'changed during'):
                parallel(dict(DEFAULT,mix=str(mix)),8,[])
            with self.assertRaisesRegex(ValueError,'concurrency'):parallel(dict(DEFAULT,mix=str(mix)),8,[],workers=0)


if __name__=='__main__':unittest.main()
