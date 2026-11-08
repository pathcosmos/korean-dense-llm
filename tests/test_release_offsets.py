import hashlib
import numpy as np
import pytest
from data.release_offsets import scan_boundaries


def payload(tmp_path,values):
    p=tmp_path/'tokens.bin';p.write_bytes(np.asarray(values,dtype='<u2').tobytes())
    return p,hashlib.sha256(p.read_bytes()).hexdigest()


def test_boundaries_across_chunks_and_single_token_document(tmp_path):
    p,h=payload(tmp_path,[9,8,0,0,7,0]);starts,ends,proof=scan_boundaries(p,0,3,h,chunk_bytes=4)
    assert starts.tolist()==[0,3,4] and ends.tolist()==[3,4,6]
    assert proof['tokens']==6


def test_literal_eos_never_silently_becomes_an_extra_document(tmp_path):
    p,h=payload(tmp_path,[4,0,5,0])
    with pytest.raises(ValueError,match='literal'):scan_boundaries(p,0,1,h)


def test_changed_token_content_fails_before_publication(tmp_path):
    p,h=payload(tmp_path,[4,0]);p.write_bytes(np.asarray([5,0],dtype='<u2').tobytes())
    with pytest.raises(AssertionError,match='SHA'):scan_boundaries(p,0,1,h)
