"""Behavioural oracle for data/filters.py: freeze outputs, assert byte-identical defaults later.

Generated BEFORE refactoring filters.py and committed as tests/golden/filters_oracle_v1.json.
It uses only the original public API (check_ko, check_en, check_math, norm_ws, exact_hash,
mask_pii), so it runs unchanged before and after the refactor. No raw real text is stored:
real documents appear only as sha256 keys.

  python -m dataqa.oracle --out tests/golden/filters_oracle_v1.json [--real dataqa/review/real_gold.jsonl]
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from data import filters as F
from dataqa import gold, seed_gold

SCHEMA = 1


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def fuzz_ko_texts(n=1500, seed=7):
    """Seven document shapes so every check_ko reason (and keep) is exercised."""
    rng = random.Random(seed)
    syl = [chr(c) for c in range(0xAC00, 0xAC00 + 80)]
    ascii_words = ['cache', 'thread', 'index', 'value', 'merge', 'token', 'batch', 'queue', 'x1', 'ok']

    def word():
        return ''.join(rng.choice(syl) for _ in range(rng.randint(1, 5)))

    def build(unit, length, sep=' '):
        parts, total = [], 0
        while total < length:
            u = unit()
            parts.append(u)
            total += len(u) + len(sep)
        return sep.join(parts)[:length]

    out = []
    for _ in range(n):
        length = rng.choice([0, 5, 50, 199, 200, 201, 400, 900, 1500])
        kind = rng.randrange(7)
        if kind == 0:
            t = build(word, length)
        elif kind == 1:
            ratio = rng.uniform(0.05, 0.7)
            t = build(lambda: rng.choice(ascii_words) if rng.random() < ratio else word(), length)
        elif kind == 2:
            lines = [build(word, rng.randint(15, 40)) for _ in range(rng.randint(1, 4))]
            t = ('\n'.join(rng.choice(lines) for _ in range(rng.randint(6, 14))))[:max(length, 0) or None]
        elif kind == 3:
            sentence = build(word, rng.randint(25, 60))
            t = build(lambda: sentence, length)
        elif kind == 4:
            deco = rng.choice(['★☆★', '▶▷▶', '###', '---'])
            t = build(lambda: word() + ' ' + deco, length)
        elif kind == 5:
            t = list(build(word, length))
            for _ in range(rng.randint(0, 4)):
                if t:
                    t.insert(rng.randrange(len(t) + 1), '\ufffd')
            t = ''.join(t)
        else:
            alphabet = list('가나다라마바사아자차카타파하abcdefgh0123456789 \n-.@\ufffd')
            t = ''.join(rng.choice(alphabet) for _ in range(length))
        out.append(t)
    return out


def fuzz_pii_texts(n=1500, seed=11):
    rng = random.Random(seed)
    parts = ['010', '-', ' ', '+82', '82', '0', '900101', '1', '5', '1234', '5678', 'a', '@', 'example.com', '.', '(02)', '\n']
    return [''.join(rng.choice(parts) for _ in range(rng.randint(1, 14))) for _ in range(n)]


def probe(text):
    """All observable per-document outputs of the original filter API."""
    normalized = F.norm_ws(text)
    return dict(
        text_sha256=sha(text),
        norm_sha256=sha(normalized),
        ko=F.check_ko(normalized), en=F.check_en(normalized), math=F.check_math(normalized),
        ko_len_knobs=F.check_ko(normalized, min_chars=0, max_chars=60),
        exact_hash=F.exact_hash(normalized),
        masked_sha256=sha(F.mask_pii(normalized)),
        masked_raw_sha256=sha(F.mask_pii(text)))


def digest(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode() + b'\n')
    return h.hexdigest()


def compute(real_texts=()):
    seed = [probe(r['text']) for r in seed_gold.build()]
    ko_fuzz = [probe(t) for t in fuzz_ko_texts()]
    pii_fuzz = [probe(t) for t in fuzz_pii_texts()]
    real = [probe(t) for t in real_texts]
    return dict(
        schema=SCHEMA,
        note='Outputs of the original data/filters.py API; text keys are sha256, no raw text.',
        seed_gold=seed,
        real_pilot=real,
        fuzz_ko=dict(n=len(ko_fuzz), digest=digest(ko_fuzz),
                     ko_reasons=dict(Counter(map(str, (r['ko'] for r in ko_fuzz))))),
        fuzz_pii=dict(n=len(pii_fuzz), digest=digest(pii_fuzz),
                      masked_changed=sum(r['masked_raw_sha256'] != r['text_sha256'] for r in pii_fuzz)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--real', help='local real gold JSONL (texts are hashed, never written)')
    a = ap.parse_args()
    real = [r['text'] for r in gold.load(a.real)] if a.real else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(compute(real), indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print(f'oracle written: {a.out}')


if __name__ == '__main__':
    main()
