"""Check a candidate tokenizer against explicit JSONL {category,text} fixtures.
Exit: 0 passed, 1 failed, 2 code Unicode normalization needs policy review.
No network access, no final tokenizer replacement, no implicit dataset selection.
"""
import argparse
import json
from pathlib import Path
import sys
import unicodedata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from tokenizer.train_tokenizer import SPECIAL
from common.data_contracts import resolve, identity, sha256_file, atomic_json

CATEGORIES = ('ko', 'en', 'code', 'math')
PROBES = [
    ('ko', '한글과 English, 숫자 0123456789\n다음 줄\t끝'),
    ('en', 'Cafe\u0301 and café\n\tIndented  line'),
    ('code', 'def f(X):\n\treturn X + 1\n'),
    ('code', 's = "e\u0301"\nt = "é"\n'),
    ('math', r'\int_0^1 x^2 dx = 1/3; α + β ≤ 10'),
] + [('en', f'before {token} after') for token in SPECIAL]


def check(tokenizer_path, samples_path, output, expected_vocab=65536, normalization='NFC'):
    if normalization not in ('raw','NFC'):raise ValueError('Unsupported normalization policy')
    tok_path, samples_path, output = map(Path, (tokenizer_path, samples_path, output))
    contract = dict(version=2, normalization=normalization, tokenizer_sha256=sha256_file(tok_path),
                    samples_sha256=sha256_file(samples_path), expected_vocab=expected_vocab,
                    required_categories=list(CATEGORIES), probes=[list(p) for p in PROBES],
                    encode_special=False, decode_skip_special=False,
                    checker_sha256=sha256_file(Path(__file__)))
    contract_id = identity(contract)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get('contract_id') != contract_id:
            raise ValueError('Checker input/tokenizer/config changed; use a new report path')
    tok = Tokenizer.from_file(str(tok_path)); vocab = tok.get_vocab()
    errors=[]; examples=[]; code_changes=0
    actual_normalizer=json.loads(tok.to_str()).get('normalizer')
    if actual_normalizer != (None if normalization=='raw' else {'type':'NFC'}):
        errors.append('Tokenizer normalizer differs from declared policy')
    if len(vocab) != expected_vocab: errors.append('Vocabulary size differs from requested size')
    ids=list(vocab.values())
    if any(type(i) is not int or not 0 <= i <= 65535 for i in ids) or len(set(ids)) != len(ids):
        errors.append('Invalid uint16 vocabulary IDs')
    if set(ids) != set(range(len(vocab))): errors.append('Vocabulary IDs are not contiguous')
    special_ids={t:tok.token_to_id(t) for t in SPECIAL}
    for token, token_id in special_ids.items():
        if token_id is None or tok.encode(token,add_special_tokens=False).ids != [token_id]:
            errors.append(f'Missing or non-atomic special token: {token}')
        elif tok.decode([token_id],skip_special_tokens=False) != token:
            errors.append(f'Special token decode mismatch: {token}')
    totals={c:dict(documents=0,characters=0,utf8_bytes=0,tokens=0,nfc_failures=0,policy_failures=0,raw_changes=0) for c in CATEGORIES}
    def inspect(category,text,label,aggregate):
        nonlocal code_changes
        if not isinstance(text,str) or not text:
            raise ValueError('Each sample text must be a nonempty string')
        encoded=tok.encode(text,add_special_tokens=False).ids
        if not encoded or any(not 0 <= i <= 65535 for i in encoded):
            errors.append(f'Invalid encoding at {label}')
        decoded=tok.decode(encoded,skip_special_tokens=False)
        nfc=unicodedata.normalize('NFC',text)
        expected=text if normalization=='raw' else nfc
        failed=decoded!=expected; raw_changed=decoded!=text
        if failed:errors.append(f'{normalization} roundtrip mismatch at {label}')
        if category=='code' and raw_changed:
            code_changes+=1
        if (failed or raw_changed) and len(examples)<20:
            examples.append(dict(sample=label,category=category,nfc_failed=decoded!=nfc,policy_failed=failed,
                                 raw_changed=raw_changed,text_sha256=identity(text)))
        if aggregate:
            t=totals[category];t['documents']+=1;t['characters']+=len(text)
            t['utf8_bytes']+=len(text.encode());t['tokens']+=len(encoded)
            t['nfc_failures']+=int(decoded!=nfc);t['policy_failures']+=int(failed);t['raw_changes']+=int(raw_changed)
    with samples_path.open() as stream:
        for row,line in enumerate(stream):
            if not line.strip():continue
            item=json.loads(line);category=item.get('category')
            if category not in totals:raise ValueError(f'Unknown category at row {row}')
            inspect(category,item.get('text'),f'row:{row}',True)
    for category,total in totals.items():
        if total['documents']==0:errors.append(f'Missing category: {category}')
        total['chars_per_token']=total['characters']/total['tokens'] if total['tokens'] else None
        total['bytes_per_token']=total['utf8_bytes']/total['tokens'] if total['tokens'] else None
    for i,(category,text) in enumerate(PROBES):inspect(category,text,f'probe:{i}',False)
    status='failed' if errors else ('needs_review' if code_changes else 'passed')
    report=dict(status=status,contract_id=contract_id,contract=contract,vocab_size=len(vocab),
                special_ids=special_ids,categories=totals,errors=errors,
                code_raw_change_count=code_changes,examples=examples,
                note='Code raw Unicode changes include built-in probes and require policy review; NFC success does not prove raw code preservation.')
    if output.exists() and json.loads(output.read_text())!=report:
        raise ValueError('Existing report differs; refusing overwrite')
    if not output.exists():atomic_json(output,report)
    if status=='passed':atomic_json(Path(str(output)+'.done'),dict(contract_id=contract_id,sha256=sha256_file(output)))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizer',required=True);ap.add_argument('--samples',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--expected-vocab',type=int,default=65536)
    ap.add_argument('--normalization',choices=['raw','NFC'],default='NFC')
    a=ap.parse_args()
    try:
        r=check(resolve(a.tokenizer),resolve(a.samples),resolve(a.output),a.expected_vocab,a.normalization)
        print(json.dumps(dict(status=r['status'],categories=r['categories'],code_raw_change_count=r['code_raw_change_count']),ensure_ascii=False))
        return {'passed':0,'failed':1,'needs_review':2}[r['status']]
    except (ValueError,OSError,KeyError) as exc:
        print(f'CHECK ERROR: {exc}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
