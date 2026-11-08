import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data.public_corpus_reader import iter_split


def fixture(tmp_path,texts=('train body','eval body')):
    t=tmp_path/'text.parquet';a=tmp_path/'assignment.parquet'
    pq.write_table(pa.table({'text':list(texts)}),t)
    table=pa.table({'row':[0,1],'content_sha256':[hashlib.sha256(s.encode()).digest() for s in ('train body','eval body')],
                    'split':['train','eval'],'group_id':['g1','g2']}).replace_schema_metadata({b'original_final_file':b'f'})
    pq.write_table(table,a)
    return t,a


def test_public_loader_never_mixes_validation_with_train(tmp_path):
    t,a=fixture(tmp_path)
    kw=dict(source='s',original_file='f',inventory_id='i',batch_size=1)
    train=list(iter_split(t,a,split='train',**kw));val=list(iter_split(t,a,split='validation',**kw))
    assert [r['text'] for r in train]==['train body']
    assert [r['text'] for r in val]==['eval body']
    assert train[0]['doc_id']!=val[0]['doc_id']


def test_modified_text_fails_before_yield(tmp_path):
    t,a=fixture(tmp_path,texts=('changed','eval body'))
    with pytest.raises(ValueError,match='body differs'):
        list(iter_split(t,a,split='train',source='s',original_file='f',inventory_id='i'))


def test_unknown_split_is_not_implicitly_train(tmp_path):
    t,a=fixture(tmp_path)
    with pytest.raises(ValueError,match='Explicit'):
        list(iter_split(t,a,split='all',source='s',original_file='f',inventory_id='i'))
