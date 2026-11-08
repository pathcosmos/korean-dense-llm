import hashlib
import sqlite3
import pyarrow.parquet as pq
import pytest
from data.release_split import export_one,logical_digest


def database():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE docs(file,row,content_sha256,repo,group_id,split)')
    return db


def test_portable_split_preserves_unicode_nulls_and_full_content_identity(tmp_path):
    db=database();a=hashlib.sha256(b'a').hexdigest();b=hashlib.sha256(b'b').hexdigest()
    db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?)',[
        ('f',0,a,None,'shared','train'),('f',1,b,'한글/repo','repo:한글/repo','eval')])
    dest=tmp_path/'assignments.parquet'
    result=export_one(db,{'path':'f','source':'s','rows':2},dest,'cid',batch_size=1)
    rows=pq.read_table(dest).to_pylist()
    assert result['counts']=={'train':1,'eval':1}
    assert rows[0]['repo'] is None and rows[1]['repo']=='한글/repo'
    assert rows[1]['content_sha256']==bytes.fromhex(b)
    assert pq.read_schema(dest).metadata[b'original_final_file']==b'f'


def test_canonical_gap_never_publishes_a_completed_file(tmp_path):
    db=database();db.execute('INSERT INTO docs VALUES(?,?,?,?,?,?)',('f',1,'00'*32,None,'g','train'))
    dest=tmp_path/'bad.parquet'
    with pytest.raises(ValueError,match='gap'):
        export_one(db,{'path':'f','source':'s','rows':1},dest,'cid')
    assert not dest.exists()


def test_logical_digest_distinguishes_null_from_empty_and_train_from_eval():
    a=(0,'00'*32,None,'g','train');b=(0,'00'*32,'','g','train');c=(0,'00'*32,None,'g','eval')
    assert len({logical_digest([r]) for r in (a,b,c)})==3
