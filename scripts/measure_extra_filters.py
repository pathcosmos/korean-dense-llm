"""Measure removal rates of extra rule-based filters on already-cleaned sample docs (pre-registered thresholds, not tuned).
usage: python scripts/measure_extra_filters.py <samples_dir> <label_docs.json> <out.json>
samples_dir: <src>.jsonl with {"text":...}; label_docs.json: [{id,text,...}] (labeled docs, for a descriptive check only)."""
import sys, os, re, json, math, collections
HANGUL = re.compile(r"[가-힣]")
# core spam/gambling/adult-service terms. Deliberately excludes ambiguous words (사설=editorial, 배팅/베팅=sports, 슬롯=hardware, 대출=news, 토토로).
SPAM = re.compile(r"출장안마|출장마사지|여성전용마사지|카지노|바카라|파워볼|먹튀|토토(?!로)|toto|룰렛|블랙잭|바이너리\s*옵션|이진\s*옵션|홀덤|오피스텔?\s*출장|마사지샵", re.I)
COMMERCE = re.compile(r"배송|반품|교환|환불|결제|장바구니|주문|할인|적립|판매자|상품|구매|쿠폰|재고")
LONGTOK = re.compile(r"[A-Za-z0-9+/=%]{40,}")

def f_spam(t):
    m = SPAM.findall(t.lower()); return len(m) >= 3 or len(set(m)) >= 2
def f_commerce(t):
    ls = [l for l in t.split("\n") if l.strip()]; k = sum(1 for l in ls if COMMERCE.search(l))
    return len(ls) >= 8 and k >= 4 and k / len(ls) >= 0.30
def f_encoded(t): return len(LONGTOK.findall(t)) >= 2
def f_hangul50(t):
    ns = sum(1 for c in t if not c.isspace()); return ns > 0 and len(HANGUL.findall(t)) / ns < 0.50
def f_shortlines(t):
    ls = [l for l in t.split("\n") if l.strip()]; return len(ls) >= 10 and sum(1 for l in ls if len(l.strip()) < 20) / len(ls) >= 0.70
FILTERS = {"spam_kw": f_spam, "commerce_lines": f_commerce, "encoded_tokens": f_encoded, "hangul_lt50": f_hangul50, "short_lines": f_shortlines}
KO_ONLY = {"hangul_lt50"}

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 2), round(100 * (c + h), 2))

def hits(t, ko):
    return [k for k, f in FILTERS.items() if (ko or k not in KO_ONLY) and f(t)]

if __name__ == "__main__":
    sd, ld, out = sys.argv[1:4]; res = {"thresholds": "see source", "sources": {}}
    for src in ["wiki_ko", "fineweb2_ko", "hplt_ko", "fineweb_edu", "owm"]:
        ko = src.endswith("_ko"); n = 0; any_ = 0; c = collections.Counter()
        for l in open(f"{sd}/{src}.jsonl", encoding="utf-8"):
            t = json.loads(l)["text"]; n += 1; h = hits(t, ko)
            for k in h: c[k] += 1
            any_ += bool(h)
        res["sources"][src] = {"n": n, "removed": any_, "pct": round(100 * any_ / n, 2), "ci95": wilson(any_, n), "by_filter": {k: {"n": v, "pct": round(100 * v / n, 2)} for k, v in c.items()}}
    docs = json.load(open(ld, encoding="utf-8")); res["labeled_hits"] = {d["id"]: hits(d["text"], True) for d in docs}
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    for s, r in res["sources"].items(): print(s, r["n"], r["removed"], r["pct"], r["ci95"], {k: v["n"] for k, v in r["by_filter"].items()})
