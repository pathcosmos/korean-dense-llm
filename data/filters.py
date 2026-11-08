"""Text quality filters + PII masking. Pure functions, no I/O."""
import re
import xxhash

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?82[- ]?)?0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}(?!\d)")
RRN = re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)")
HANGUL = re.compile(r"[가-힣]")
MULTI_NL = re.compile(r"\n{3,}")


def mask_pii(t):
    t = EMAIL.sub("<email>", t)
    t = RRN.sub("<id>", t)
    return PHONE.sub("<phone>", t)


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


def check_ko(t, min_chars=200, max_chars=100_000):
    """Return None if kept, else a reject-reason string. Korean web text."""
    n = len(t)
    if n < min_chars: return "short"
    if n > max_chars: return "long"
    nonspace = sum(1 for c in t if not c.isspace())
    if nonspace == 0: return "empty"
    h = len(HANGUL.findall(t))
    if h < 50: return "few_hangul"
    if h / nonspace < 0.30: return "low_hangul_ratio"
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) > 5 and dedup_lines_stat(lines) > 0.30: return "dup_lines"
    if ngram_rep(t.split()) > 0.30: return "ngram_rep"
    alpha = sum(c.isalnum() for c in t)
    if alpha / n < 0.5: return "low_alnum"
    if "�" in t and t.count("�") / n > 0.001: return "bad_encoding"
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
