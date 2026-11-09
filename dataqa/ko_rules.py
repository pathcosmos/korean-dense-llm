"""Parameterised copy of data.filters.check_ko plus the parity gate that keeps it honest.

check_ko_param(t, KoParams()) must return the same reason string as data.filters.check_ko(t)
for every input (reason ORDER is part of the behaviour: first failing check wins).
`parity_mismatches` is the gate; tests and the CLI refuse to report if it finds any.

The per-document sequence of data/clean.py is: norm_ws -> check -> exact_hash dedup -> mask_pii.
`filter_reason` reproduces the norm_ws -> check part (dedup is cross-document, not measured here).
"""
import unicodedata
from dataclasses import dataclass, asdict, replace

from data import filters as F


@dataclass(frozen=True)
class KoParams:
    min_chars: int = 200
    max_chars: int = 100_000
    min_hangul: int = 50
    min_hangul_ratio: float = 0.30
    max_dup_line_frac: float = 0.30
    dup_line_min_lines: int = 5        # production: len(lines) > 5
    max_ngram_rep: float = 0.30
    min_alnum_ratio: float = 0.50
    max_bad_encoding_ratio: float = 0.001
    pre_normalize: str = 'none'        # candidate knob: 'none' (production) | 'nfc'

    def as_dict(self):
        return asdict(self)


DEFAULT = KoParams()


def check_ko_param(t, p=DEFAULT):
    """Same control flow as data.filters.check_ko with thresholds taken from `p`."""
    if p.pre_normalize == 'nfc':
        t = unicodedata.normalize('NFC', t)
    elif p.pre_normalize != 'none':
        raise ValueError('pre_normalize must be none or nfc')
    n = len(t)
    if n < p.min_chars: return 'short'
    if n > p.max_chars: return 'long'
    nonspace = sum(1 for c in t if not c.isspace())
    if nonspace == 0: return 'empty'
    h = len(F.HANGUL.findall(t))
    if h < p.min_hangul: return 'few_hangul'
    if h / nonspace < p.min_hangul_ratio: return 'low_hangul_ratio'
    lines = [l for l in t.split('\n') if l.strip()]
    if len(lines) > p.dup_line_min_lines and F.dedup_lines_stat(lines) > p.max_dup_line_frac: return 'dup_lines'
    if F.ngram_rep(t.split()) > p.max_ngram_rep: return 'ngram_rep'
    alpha = sum(c.isalnum() for c in t)
    if alpha / n < p.min_alnum_ratio: return 'low_alnum'
    if '�' in t and t.count('�') / n > p.max_bad_encoding_ratio: return 'bad_encoding'
    return None


def filter_reason(text, p=DEFAULT):
    """norm_ws -> check, the first two per-document steps of data/clean.py."""
    return check_ko_param(F.norm_ws(text), p)


def parity_mismatches(texts, p=DEFAULT):
    """Return [(index, production_reason, copy_reason)] for texts where the copy differs.

    Only meaningful when `p` equals the defaults; the production function has no knobs
    other than min_chars/max_chars, which are checked too when they are the only non-default.
    """
    only_len = replace(DEFAULT, min_chars=p.min_chars, max_chars=p.max_chars)
    if p != only_len:
        raise ValueError('parity gate only applies to production-equivalent parameters')
    bad = []
    for i, t in enumerate(texts):
        prod = F.check_ko(t, min_chars=p.min_chars, max_chars=p.max_chars)
        mine = check_ko_param(t, p)
        if prod != mine:
            bad.append((i, prod, mine))
    return bad
