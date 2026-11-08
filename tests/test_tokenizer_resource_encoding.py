import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from tokenizers import Tokenizer,models,pre_tokenizers
from common.data_contracts import sha256_file
from scripts.benchmark_tokenizer_resources import encode


class TokenizerEncodingTest(unittest.TestCase):
    def test_pool_output_has_one_eos_per_document_and_same_uint16_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tok=Tokenizer(models.WordLevel({'[UNK]':0,'<|eos|>':1,'hello':2,'world':3},unk_token='[UNK]'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace();tp=root/'tokenizer.json';tok.save(str(tp))
            shards=[]
            for i in range(2):
                p=root/f'part{i}.jsonl';p.write_text(''.join(json.dumps({'text':'hello world'})+'\n' for _ in range(257)))
                shards.append({'path':str(p),'sha256':sha256_file(p)})
            fp=root/'frame.json';fp.write_text(json.dumps({'status':'experimental_performance_only','shards':shards}))
            identities=[]
            for w in [1,2]:
                out=root/f'w{w}';out.mkdir()
                encode(SimpleNamespace(frame=str(fp),output=str(out),scratch=str(root/'scratch'),tokenizer=str(tp),workers=w))
                result=json.loads((out/'result.json').read_text());identities.append(result['output_identity'])
                self.assertEqual(result['documents'],514);self.assertEqual(result['tokens'],1542)
                for i in range(2):
                    data=np.fromfile(root/'scratch'/out.name/f'part{i}.bin',dtype='<u2')
                    np.testing.assert_array_equal(data,np.tile([2,3,1],257))
            self.assertEqual(*identities)
