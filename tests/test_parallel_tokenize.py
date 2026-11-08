import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import trainers,pre_tokenizers
from common.data_contracts import atomic_json,source_files,sha256_file,artifact_path
from data.split_manifest import build
from data.tokenize_shards import tokenize_source
from data.parallel_tokenize import run
from data.verified_split_context import VerifiedSplitContext,distribution
from tokenizer.train_tokenizer import make_tokenizer,SPECIAL


def fixture(root):
    sources={};alltexts=[]
    for source in ['web','code']:
        files=[]
        for i in range(2):
            p=root/f'{source}{i}.parquet'
            texts=[f'{source} {i} {j} 한글 e\u0301\n\treturn 12 <|eos|>' for j in range(40)]
            texts[0]='shared duplicate';alltexts.extend(texts)
            obj={'text':texts}
            if source=='code':obj['repo_name']=[f'org/r{j//4}' for j in range(40)]
            pq.write_table(pa.table(obj),p,row_group_size=7);Path(str(p)+'.done').touch();files.append(str(p))
        mark=root/(source+'.done');mark.touch()
        sources[source]=dict(files=files,text_policy='raw-utf8-code-v1' if source=='code' else 'nfc-web-exact-v1',completion_marker=str(mark))
    sm=root/'sources.json';atomic_json(sm,dict(schema_version=1,status='ready',sources=sources));build(sm,root/'split',eval_fraction=.3)
    tok=make_tokenizer('raw');tok.train_from_iterator(alltexts,trainers.BpeTrainer(vocab_size=320,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    tp=root/'tok.json';tok.save(str(tp));return sm,root/'split/manifest.json',tp


class ParallelTokenTest(unittest.TestCase):
    def test_native_bytes_all_workers_batch_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            for source in ['web','code']:
                files,snapshot=source_files(source,sm)
                tokenize_source(source,tok,files,root/'serial'/source,batch_size=9,snapshot=snapshot,split_manifest=split)
            first=run(sm,split,tok,root/'p1',workers=1,batch_size=3,experimental=True)
            other=run(sm,split,tok,root/'p2',workers=2,batch_size=17,experimental=True)
            self.assertEqual(first['logical_id'],other['logical_id'])
            for path in (root/'serial').glob('*/*.bin'):
                rel=path.relative_to(root/'serial')
                self.assertEqual(path.read_bytes(),(root/'p1'/rel).read_bytes())
                self.assertEqual(path.read_bytes(),(root/'p2'/rel).read_bytes())
            reused=run(sm,split,tok,root/'p2',workers=1,batch_size=4,experimental=True);self.assertEqual(reused['reused'],4)
            from data.build_mix import build_mix
            with patch('data.build_mix.PROJECT_ROOT',str(root)):
                mix=build_mix({'web':1,'code':1},root/'p2',root/'mix.json',seq_len=4)
            self.assertEqual(mix['format'],'uint16-eos-split-v3')
            f=next((root/'p2').glob('*/*.bin'));body=f.read_bytes();f.write_bytes(b'xx'+body[2:])
            with self.assertRaisesRegex(ValueError,'corrupt'):run(sm,split,tok,root/'p2',workers=1,experimental=True)

    def test_distribution_missing_certificate_and_content_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            with VerifiedSplitContext(sm,split) as ctx:
                r=distribution(ctx);self.assertEqual(sum(r['counts'].values()),160)
                self.assertEqual(sum(x['documents'] for s in r['sources'].values() for x in s['splits'].values()),160)
                self.assertGreaterEqual(r['largest_components'][0]['documents'],4)
            marker=split.parent/'.done';marker.rename(split.parent/'saved.done')
            with self.assertRaisesRegex(ValueError,'Incomplete'):VerifiedSplitContext(sm,split)
            (split.parent/'saved.done').rename(marker)
            pq.write_table(pa.table({'text':['changed']}),root/'web0.parquet')
            with self.assertRaisesRegex(ValueError,'files/content'):VerifiedSplitContext(sm,split)

    def test_fresh_verified_records_not_rehashed_and_production_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            actual=source_files
            def freshly_verified(source,manifest,**kwargs):
                files,snapshot=actual(source,manifest)
                records=[dict(path=artifact_path(p),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in files]
                return files,snapshot,records
            with patch('data.verified_split_context.source_files',side_effect=freshly_verified),patch('data.verified_split_context.sha256_file',wraps=sha256_file) as h:
                with VerifiedSplitContext(sm,split):pass
                self.assertFalse(any(str(c.args[0]).endswith('.parquet') for c in h.call_args_list))
            with self.assertRaises((ValueError,FileNotFoundError)):run(sm,split,tok,root/'prod',workers=1)

    def test_process_crash_after_pending_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root);out=root/'partial'
            code="from data.parallel_tokenize import run; import sys; run(*sys.argv[1:],workers=1,experimental=True,_crash='after_pending')"
            p=subprocess.run([sys.executable,'-c',code,str(sm),str(split),str(tok),str(out)],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(p.returncode,0);self.assertFalse((out/'.done').exists())
            self.assertTrue(list(out.glob('*/*.pending.json')))
            r=run(sm,split,tok,out,workers=2,experimental=True);self.assertGreaterEqual(r['reused'],1)
            baseline=run(sm,split,tok,root/'clean',workers=1,experimental=True);self.assertEqual(r['logical_id'],baseline['logical_id'])
