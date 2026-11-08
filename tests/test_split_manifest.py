import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer,models,pre_tokenizers
from common.data_contracts import identity
from data.split_manifest import build,SplitSelection,validate_db
from data.tokenize_shards import tokenize_source
from data.build_mix import build_mix
from scripts.validate_source_manifest import validate


def fixture(root):
    web=root/'web.parquet';code1=root/'code1.parquet';code2=root/'code2.parquet'
    texts=['dup']+[f'web{i}' for i in range(30)]
    pq.write_table(pa.table({'text':texts}),web)
    pq.write_table(pa.table({'text':['dup','repoAfile']+[f'code{i}' for i in range(20)],
                 'repo_name':['org/A','org/A']+[f'org/r{i}' for i in range(20)]}),code1)
    pq.write_table(pa.table({'text':['dup','repoBfile','code0'],'repo_name':['org/B','org/B','org/r0']}),code2)
    for f in [web,code1,code2]:Path(str(f)+'.done').touch()
    marker=root/'source.done';marker.touch()
    sources={'web':dict(files=[str(web)],text_policy='nfc-web-exact-v1',completion_marker=str(marker)),
             'code':dict(files=[str(code1),str(code2)],text_policy='raw-utf8-code-v1',completion_marker=str(marker))}
    manifest=root/'sources.json';manifest.write_text(json.dumps(dict(schema_version=1,status='ready',sources=sources)))
    alltext=texts+['repoAfile','repoBfile']+[f'code{i}' for i in range(20)]
    vocab={'<unk>':0,'<|eos|>':1,**{x:i+2 for i,x in enumerate(dict.fromkeys(alltext))}}
    tok=Tokenizer(models.WordLevel(vocab,unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=root/'tok.json';tok.save(str(tokenizer))
    return manifest,tokenizer,[web,code1,code2]


class SplitManifestTest(unittest.TestCase):
    def test_eval_only_shard_is_excluded_from_train_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker=root/'source.done';marker.touch()
            files=[]
            for i in range(12):
                f=root/f'web{i}.parquet';pq.write_table(pa.table({'text':[f'text{i}']}),f)
                Path(str(f)+'.done').touch();files.append(f)
            manifest=root/'sources.json';manifest.write_text(json.dumps(dict(schema_version=1,status='ready',sources={
                'web':dict(files=[str(f) for f in files],text_policy='nfc-web-exact-v1',completion_marker=str(marker))})))
            tok=Tokenizer(models.WordLevel({'<unk>':0,'<|eos|>':1,**{f'text{i}':i+2 for i in range(12)}},unk_token='<unk>'))
            path=root/'tok.json';tok.save(str(path));report=build(manifest,root/'split',.5,7)
            self.assertEqual(set(report['counts']),{'train','eval'})
            tokenize_source('web',path,files,root/'tokens/web',snapshot=identity(json.loads(manifest.read_text())),split_manifest=root/'split/manifest.json')
            with patch('data.build_mix.PROJECT_ROOT',str(root)):
                mix=build_mix({'web':1},root/'tokens',root/'mix.json',seq_len=1)
            self.assertEqual(len(mix['sources']['web']['files']),report['counts']['train'])
            self.assertEqual(len(mix['holdout']),report['counts']['eval'])

    def test_group_reproducibility_and_token_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,tokpath,files=fixture(root)
            report=build(manifest,root/'split',.4,7)
            self.assertEqual(set(report['counts']),{'train','eval'})
            self.assertEqual(build(manifest,root/'split',.4,7),report)
            with self.assertRaisesRegex(ValueError,'changed'):build(manifest,root/'split',.3,7)
            other=build(manifest,root/'split2',.4,7)
            self.assertEqual(report['contract_id'],other['contract_id'])
            select=SplitSelection(root/'split/manifest.json');other_select=SplitSelection(root/'split2/manifest.json')
            try:
                rows=select.db.execute('SELECT file,row,group_id,split FROM docs ORDER BY file,row').fetchall()
                self.assertEqual(rows,other_select.db.execute('SELECT file,row,group_id,split FROM docs ORDER BY file,row').fetchall())
                linked=select.db.execute("SELECT DISTINCT group_id,split FROM docs WHERE repo IN ('org/a','org/b') OR (file=? AND row=0)",(str(files[0]),)).fetchall()
                self.assertEqual(len(linked),1)
                tokenizer=Tokenizer.from_file(str(tokpath));snapshot=identity(json.loads(manifest.read_text()))
                for source,sourcefiles in [('web',files[:1]),('code',files[1:])]:
                    result=tokenize_source(source,tokpath,sourcefiles,root/'tokens'/source,snapshot=snapshot,split_manifest=root/'split/manifest.json',batch_size=3)
                    self.assertEqual(result['train_docs']+result['holdout_docs'],sum(pq.ParquetFile(f).metadata.num_rows for f in sourcefiles))
                    for f in sourcefiles:
                        texts=pq.read_table(f,columns=['text']).column(0).to_pylist();expected={'train':[],'eval':[]}
                        for row,split in select.rows(f):expected[split]+=tokenizer.encode(texts[row]).ids+[1]
                        actual=np.fromfile(root/'tokens'/source/(f.stem+'.bin'),dtype=np.uint16).tolist()
                        self.assertEqual(actual,expected['train'])
                        hold=root/'tokens'/source/(f.stem+'.hold.bin')
                        self.assertEqual(np.fromfile(hold,dtype=np.uint16).tolist() if hold.exists() else [],expected['eval'])
                with patch('data.build_mix.PROJECT_ROOT',str(root)):
                    mix=build_mix({'web':.5,'code':.5},root/'tokens',root/'mix.json',seq_len=1)
                self.assertTrue(mix['holdout']);self.assertEqual(mix['format'],'uint16-eos-split-v3')
            finally:select.close();other_select.close()

    def test_leakage_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,tok,files=fixture(root);report=build(manifest,root/'split',.4,7)
            snapshot=identity(json.loads(manifest.read_text()))
            with self.assertRaisesRegex(ValueError,'combine'):tokenize_source('web',tok,files[:1],root/'bad',holdout_docs=1,snapshot=snapshot,split_manifest=root/'split/manifest.json')
            pq.write_table(pa.table({'text':['changed']}),files[0])
            with self.assertRaisesRegex(ValueError,'files/content'):tokenize_source('web',tok,files[:1],root/'changed',snapshot=snapshot,split_manifest=root/'split/manifest.json')
            db=sqlite3.connect(root/'split/split.sqlite')
            row=db.execute("SELECT file,row,split FROM docs WHERE repo='org/a' LIMIT 1").fetchone()
            db.execute('UPDATE docs SET split=? WHERE file=? AND row=?',('eval' if row[2]=='train' else 'train',row[0],row[1]));db.commit()
            with self.assertRaisesRegex(ValueError,'leakage'):validate_db(db)
            db.close()
            with self.assertRaisesRegex(ValueError,'DB content changed'):SplitSelection(root/'split/manifest.json')

    def test_manifest_audit_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,tok,files=fixture(root)
            report=validate(manifest);self.assertEqual(report['structural_status'],'passed')
            self.assertEqual(report['quality_status'],'not_certified_by_this_tool')
            raw=json.loads(manifest.read_text());raw['status']='draft';manifest.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'draft'):validate(manifest)
            raw['status']='ready';manifest.write_text(json.dumps(raw))
            pq.write_table(pa.table({'text':['code']}),files[1])
            with self.assertRaisesRegex(ValueError,'provenance'):build(manifest,root/'bad',.4,7)
