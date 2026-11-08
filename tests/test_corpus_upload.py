import hashlib
import os
from pathlib import Path
import pytest
import scripts.upload_public_corpus as upload


def test_bounded_batches_keep_large_single_files_and_never_drop_a_file():
    rows=[{'bytes':n,'path':str(i)} for i,n in enumerate([2,3,20,1,1])]
    result=list(upload.batches(rows,max_files=2,max_bytes=5))
    assert [[r['bytes'] for r in batch] for batch in result]==[[2,3],[20],[1,1]]
    assert [r['path'] for batch in result for r in batch]==[r['path'] for r in rows]


def test_frozen_input_change_is_rejected_before_network(tmp_path,monkeypatch):
    monkeypatch.setattr(upload,'PROJECT_ROOT',str(tmp_path));p=tmp_path/'file.bin';p.write_bytes(b'one');s=p.stat()
    record={'local_path':'file.bin','path':'remote.bin','bytes':3,'sha256':hashlib.sha256(b'one').hexdigest(),
            'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
    p.write_bytes(b'two')
    # Coarse filesystem timestamps can otherwise exercise the SHA guard instead.
    os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1_000_000_000))
    with pytest.raises(ValueError,match='changed'):upload.make_operation(record)


def test_correct_size_but_wrong_hash_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(upload,'PROJECT_ROOT',str(tmp_path));p=tmp_path/'file.bin';p.write_bytes(b'one');s=p.stat()
    r={'local_path':'file.bin','path':'remote.bin','bytes':3,'sha256':'00'*32,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
    with pytest.raises(ValueError,match='SHA256'):upload.make_operation(r)


def test_storage_quota_is_distinguished_from_request_rate():
    assert upload.is_quota_error(Exception('You exceeded your storage quota'))
    assert not upload.is_quota_error(Exception('429 API quota exceeded: request rate'))
