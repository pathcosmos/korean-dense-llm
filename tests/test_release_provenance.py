import hashlib

from data.filters import exact_hash, mask_pii, norm_ws
from data.release_provenance import doc_id, first_accepted_rows, match_candidates


def final(text):
    normalized = norm_ws(text)
    return {'text': mask_pii(normalized), 'h': exact_hash(normalized) - 2**63}


def test_masked_equal_bodies_keep_distinct_raw_origins():
    a, b = 'Contact a@example.com', 'Contact b@example.com'
    assert final(a)['text'] == final(b)['text']
    matches = match_candidates([final(a), final(b)], [dict(text=b, url='b'), dict(text=a, url='a')])
    assert [m[0][0] for m in matches] == [1, 0]
    assert all('text' not in m[0][1] for m in matches)


def test_all_candidates_are_kept_and_unmatched_rows_remain_missing():
    text = '한국어 본문\n두 번째 줄'
    rows = [dict(text=text, url='first'), dict(text='\r'+text+'\n', url='second')]
    matches = match_candidates([final(text), final('다른 문서')], rows)
    assert [n for n, _ in matches[0]] == [0, 1]
    assert matches[1] == []


def test_doc_id_component_boundaries_and_shard_row_identity():
    assert doc_id('a', 'bc', 'file', 0) != doc_id('ab', 'c', 'file', 0)
    assert doc_id('a', 'bc', 'file', 0) != doc_id('a', 'bc', 'file', 1)
    assert len(doc_id('a', 'bc', 'file', 0)) == len(hashlib.sha256().hexdigest())


def test_survivor_replay_checks_filter_before_marking_seen():
    text = 'body'
    rows = [dict(text=None), dict(text=text), dict(text=text), dict(text=text)]
    calls = []
    def check(value):
        calls.append(value)
        return 'reject' if len(calls) == 1 else None
    h = final(text)['h']
    assert first_accepted_rows(rows, {h}, check) == {h: 2}
    assert len(calls) == 2
