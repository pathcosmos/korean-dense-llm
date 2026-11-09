"""Feasibility probe: can cheap structural signals separate 'semantic_low_quality' from 'keep'?

Rule-based filters miss most semantically poor documents (ads, SEO spam, product lists, boilerplate). Before
building a learned filter it is worth knowing how far trivial features go. This probe reports, on a labelled
gold file:
  * per-feature ROC AUC with a bootstrap interval over documents (0.5 = no signal)
  * a leave-one-out L2 logistic regression over all features (AUC and the share of junk caught at 90% keep retention)

Reading it: it is a feasibility signal under two limits. The labels are model-made and unreviewed unless the
gold says otherwise, and n is small, so an AUC near 0.5-0.7 means 'cheap features are weak', not 'no classifier
can work'. Documents rejected for rule-expressible reasons (e.g. short) and ambiguous ones are excluded.

  python -m dataqa.semantic_probe --gold dataqa/review100/real_gold100.jsonl --out docs/evaluation/semantic_probe_v1.json
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np

from data import filters as F
from dataqa import gold

END = re.compile(r'(다|요|죠|까|네|음|함|임|\.|\?|!|”|"|』|」)\s*$')
URL = re.compile(r'https?://\S+')
NAMES = ['log_chars', 'median_line_chars', 'short_line_frac', 'sentence_end_frac', 'urls_per_1k', 'digit_frac',
         'ascii_alpha_frac', 'unique_word_ratio', 'mean_word_len', 'lines_per_1k', 'hangul_frac']


def features(text):
    lines = [l for l in text.split('\n') if l.strip()]
    words = text.split()
    n = max(len(text), 1)
    lens = [len(l) for l in lines] or [0]
    return [math.log(n), float(np.median(lens)), sum(x < 20 for x in lens) / len(lens),
            sum(bool(END.search(l)) for l in lines) / max(len(lines), 1), 1000 * len(URL.findall(text)) / n,
            sum(c.isdigit() for c in text) / n, sum(c.isascii() and c.isalpha() for c in text) / n,
            len(set(words)) / max(len(words), 1), float(np.mean([len(w) for w in words])) if words else 0.0,
            1000 * len(lines) / n, len(F.HANGUL.findall(text)) / n]


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos) / (len(pos) * len(neg)))


def fit_logistic(X, y, l2, iters=400, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(Z) + l2 * w / len(Z))
        b -= lr * float((p - y).mean())
    return mu, sd, w, b


def loo_scores(X, y, l2):
    scores = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        mu, sd, w, b = fit_logistic(X[keep], y[keep], l2)
        scores[i] = float(((X[i] - mu) / sd) @ w + b)
    return scores


def select(records):
    return [r for r in records if not r.get('ambiguous') and (r['label'] == 'keep' or r.get('reject_reason') == 'semantic_low_quality')]


def probe(records, boots=300, seed=1):
    sel = select(records)
    X = np.array([features(r['text']) for r in sel])
    y = np.array([1 if r['label'] == 'reject' else 0 for r in sel])
    rng = random.Random(seed)
    per_feature = {}
    for j, name in enumerate(NAMES):
        a = auc(X[:, j], y)
        draws = []
        for _ in range(boots):
            idx = [rng.randrange(len(y)) for _ in y]
            v = auc(X[idx, j], y[idx])
            if v is not None:
                draws.append(v)
        draws.sort()
        per_feature[name] = dict(auc=a, ci95=[draws[int(.025 * len(draws))], draws[int(.975 * len(draws)) - 1]])
    loo = {}
    for l2 in (1.0, 10.0):
        s = loo_scores(X, y, l2)
        thr = float(np.quantile(s[y == 0], 0.90))
        caught = int((s[y == 1] > thr).sum())
        loo[f'l2={l2}'] = dict(auc=auc(s, y), junk_caught_at_90pct_keep_retention=caught, junk_total=int(y.sum()))
    return dict(n=len(y), semantic_low_quality=int(y.sum()), keep=int((1 - y).sum()), per_feature=per_feature,
                loo_logistic=loo, best_single_feature=max(per_feature, key=lambda k: abs(per_feature[k]['auc'] - .5)),
                caveats=['labels are model-made unless the gold is human-reviewed; n is small',
                         'a weak result means cheap structural features are weak, not that a learned filter cannot work'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    result = probe(gold.load(a.gold))
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    print(f"n={result['n']} semantic={result['semantic_low_quality']} best single feature {result['best_single_feature']} "
          f"AUC {result['per_feature'][result['best_single_feature']]['auc']:.3f}; LOO "
          + ', '.join(f"{k} AUC {v['auc']:.3f}" for k, v in result['loo_logistic'].items()))


if __name__ == '__main__':
    main()
