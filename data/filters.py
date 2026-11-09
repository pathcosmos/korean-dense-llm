"""Text quality filters + PII masking. Pure functions, no I/O.

Thresholds (KoPolicy) and PII behaviour (PiiPolicy) are frozen dataclasses. Their DEFAULTS are the
historical behaviour and are pinned byte for byte by tests/golden/filters_oracle_v1.json; every
non-default option is opt-in and changes outputs, so a run that uses one must record it
(data/clean.py stores the policy in the output signature).

This module is imported by bare name (`from filters import *` in data/clean.py) before the project
root is on sys.path: keep it stdlib + xxhash only, and import helpers under underscore aliases so the
star import does not leak them.
"""
import re
import unicodedata as _unicodedata
from dataclasses import dataclass as _dataclass, asdict as _asdict, replace as _replace
import xxhash

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?82[- ]?)?0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}(?!\d)")
RRN = re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)")
HANGUL = re.compile(r"[가-힣]")
MULTI_NL = re.compile(r"\n{3,}")


# International and parenthesised-area-code phone forms (opt-in 'intl'). +82 is the Korean country
# code and is written without the trunk 0 ("+82-10-..."), which the baseline PHONE cannot match.
PHONE_INTL = re.compile(
    r"(?<!\d)(?:\+82[- ]?\(?0?\)?[- ]?|82[- ]?0|0)\d{1,2}[- ]?\d{3,4}[- ]?\d{4}(?!\d)"
    r"|(?<!\d)\(0\d{1,2}\)\s?\d{3,4}[- ]?\d{4}(?!\d)")


@_dataclass(frozen=True)
class PiiPolicy:
    """phone: baseline | intl.  rrn: baseline | date_checked (first six digits must be a plausible
    YYMMDD: month 01-12, day 01-31).  email: baseline | no_git (skip ssh remotes git@host:path)."""
    phone: str = "baseline"
    rrn: str = "baseline"
    email: str = "baseline"

    def as_dict(self):
        return _asdict(self)


PII_DEFAULT = PiiPolicy()
_PII_MODES = {"phone": ("baseline", "intl"), "rrn": ("baseline", "date_checked"), "email": ("baseline", "no_git")}
_PII_MASKS = (("email", "<email>"), ("rrn", "<id>"), ("phone", "<phone>"))


def find_pii(t, kind, mode="baseline"):
    """[(start, end)] of one PII kind in `t` under `mode`; the building block of mask_pii."""
    if mode not in _PII_MODES.get(kind, ()):
        raise ValueError(f"unsupported pii mode {kind}={mode}")
    if kind == "email":
        return [(m.start(), m.end()) for m in EMAIL.finditer(t)
                if not (mode == "no_git" and t.startswith("git@", m.start()) and t[m.end():m.end() + 1] == ":")]
    if kind == "rrn":
        out = []
        for m in RRN.finditer(t):
            d = m.group()
            if mode == "date_checked" and not (1 <= int(d[2:4]) <= 12 and 1 <= int(d[4:6]) <= 31):
                continue
            out.append((m.start(), m.end()))
        return out
    pattern = PHONE if mode == "baseline" else PHONE_INTL
    return [(m.start(), m.end()) for m in pattern.finditer(t)]


def mask_pii(t, policy=None):
    if policy is None or policy == PII_DEFAULT:
        t = EMAIL.sub("<email>", t)
        t = RRN.sub("<id>", t)
        return PHONE.sub("<phone>", t)
    for kind, mask in _PII_MASKS:  # same sequential order as the default path
        spans = find_pii(t, kind, getattr(policy, kind))
        if spans:
            pieces, pos = [], 0
            for a, b in spans:
                pieces += [t[pos:a], mask]
                pos = b
            t = "".join(pieces) + t[pos:]
    return t


def norm_ws(t):
    t = "\n".join(l.strip() for l in t.replace("\r", "").split("\n"))
    return MULTI_NL.sub("\n\n", t).strip()


def dedup_lines_stat(lines):
    n = len(lines)
    uniq = len(set(lines))
    return 1 - uniq / n if n else 0


def ngram_rep(words, n=5):
    if len(words) < n * 4:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    from collections import Counter
    c = Counter(grams)
    rep = sum(v for v in c.values() if v > 1)
    return rep / len(grams)


@_dataclass(frozen=True)
class KoPolicy:
    """Thresholds of check_ko. Defaults reproduce the historical behaviour exactly.

    pre_normalize='nfc' (opt-in) applies NFC before every check, so decomposed (NFD) Hangul, which
    the [가-힣] counter does not see, is judged like its composed form."""
    min_chars: int = 200
    max_chars: int = 100_000
    min_hangul: int = 50
    min_hangul_ratio: float = 0.30
    max_dup_line_frac: float = 0.30
    dup_line_min_lines: int = 5      # the check applies when there are MORE than this many lines
    max_ngram_rep: float = 0.30
    min_alnum_ratio: float = 0.50
    max_bad_encoding_ratio: float = 0.001
    pre_normalize: str = "none"

    def as_dict(self):
        return _asdict(self)


KO_DEFAULT = KoPolicy()


def check_ko(t, min_chars=None, max_chars=None, policy=None):
    """Return None if kept, else a reject-reason string. Korean web text.

    `min_chars`/`max_chars` keep their historical keyword/positional meaning and override `policy`."""
    p = KO_DEFAULT if policy is None else policy
    if min_chars is not None or max_chars is not None:
        p = _replace(p, min_chars=p.min_chars if min_chars is None else min_chars,
                     max_chars=p.max_chars if max_chars is None else max_chars)
    if p.pre_normalize == "nfc":
        t = _unicodedata.normalize("NFC", t)
    elif p.pre_normalize != "none":
        raise ValueError("pre_normalize must be none or nfc")
    n = len(t)
    if n < p.min_chars: return "short"
    if n > p.max_chars: return "long"
    nonspace = sum(1 for c in t if not c.isspace())
    if nonspace == 0: return "empty"
    h = len(HANGUL.findall(t))
    if h < p.min_hangul: return "few_hangul"
    if h / nonspace < p.min_hangul_ratio: return "low_hangul_ratio"
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) > p.dup_line_min_lines and dedup_lines_stat(lines) > p.max_dup_line_frac: return "dup_lines"
    if ngram_rep(t.split()) > p.max_ngram_rep: return "ngram_rep"
    alpha = sum(c.isalnum() for c in t)
    if alpha / n < p.min_alnum_ratio: return "low_alnum"
    if "�" in t and t.count("�") / n > p.max_bad_encoding_ratio: return "bad_encoding"
    return None


def check_en(t, min_chars=200, max_chars=100_000):
    n = len(t)
    if n < min_chars: return "short"
    if n > max_chars: return "long"
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) > 5 and dedup_lines_stat(lines) > 0.30: return "dup_lines"
    if ngram_rep(t.split()) > 0.30: return "ngram_rep"
    if sum(c.isalnum() for c in t) / n < 0.5: return "low_alnum"
    return None


def exact_hash(t):
    return xxhash.xxh64_intdigest(" ".join(t.lower().split()).encode("utf-8"))


MATH_MARKER = re.compile(r'\$|\\(?:frac|dfrac|sqrt|sum|int|begin)\b|[=∑∫]')


def check_math(t, min_chars=200, max_chars=100_000):
    """OWM policy: preserve math-heavy text; retain length/repetition checks.

    Only relax low_alnum (0.50 -> 0.25) when explicit math markers occur.
    This is a source-specific heuristic, not a mathematical quality score.
    """
    reason = check_en(t, min_chars=min_chars, max_chars=max_chars)
    if reason != 'low_alnum':
        return reason
    if sum(c.isalnum() for c in t) / len(t) >= 0.25 and len(MATH_MARKER.findall(t)) >= 2:
        return None
    return reason
