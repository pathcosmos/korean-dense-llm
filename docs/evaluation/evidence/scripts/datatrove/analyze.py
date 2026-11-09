"""Join repo (check_ko) and upstream (FW2 reconstruction) verdicts; print markdown tables. Plain python (stdlib)."""
import json, random, statistics as st, collections, sys
import os
D = os.path.dirname(os.path.abspath(__file__)) + '/'   # inputs/outputs live next to the script (local, uncommitted)
def rd(p): return [json.loads(l) for l in open(D + p)]
def pct(x): return f"{100*x:.1f}%"

def part_b():
    # ---------- (b) distribution ----------
    meta = [json.loads(l) for l in open(D + 'sample_b.jsonl')] if False else None
    ko, up = rd('sample_b_ko.jsonl'), rd('sample_b_up.jsonl')
    assert len(ko) == len(up) == 60000, (len(ko), len(up))
    src = []; grp = []
    global rows
    for l in open(D + 'sample_b.jsonl'):
        # read only light fields without keeping text
        d = json.loads(l); src.append(d['source']); grp.append(d['group'])
    assert len(src) == len(grp) == 60000
    rows = []
    for k, u, s, g in zip(ko, up, src, grp):
        assert k['i'] == u['i']
        rows.append(dict(src=s, g=g, ko=k['ko'], up=u['first'], stg=u['stages'], n=k['n'], hr=k['hr']))
    def cell(r):
        return (r['ko'] is not None, r['up'] is not None)   # (repo_reject, fw2_reject)
    metrics = {'both_reject': lambda r: cell(r) == (True, True), 'both_pass': lambda r: cell(r) == (False, False),
               'repo_only': lambda r: cell(r) == (True, False), 'fw2_only': lambda r: cell(r) == (False, True),
               'repo_reject': lambda r: r['ko'] is not None, 'fw2_reject': lambda r: r['up'] is not None}
    def boot(rs, f, B=2000, seed=20240):
        by = collections.defaultdict(list)
        for r in rs: by[r['g']].append(r)
        gs = list(by); rng = random.Random(seed)
        num = {g: sum(1 for r in by[g] if f(r)) for g in gs}; den = {g: len(by[g]) for g in gs}
        est = sum(num.values()) / sum(den.values()); out = []
        for _ in range(B):
            pick = [rng.choice(gs) for _ in gs]
            out.append(sum(num[g] for g in pick) / sum(den[g] for g in pick))
        out.sort(); return est, out[int(.025 * B)], out[int(.975 * B) - 1]
    print('## (b) cross tables (rows: repo check_ko, cols: FW2 reconstruction)')
    for s in ['fineweb2_ko', 'hplt_ko', 'wiki_ko']:
        rs = [r for r in rows if r['src'] == s]
        c = collections.Counter(cell(r) for r in rs)
        print(f"\n### {s}: {len(rs)} docs, {len(set(r['g'] for r in rs))} row groups\n")
        print('| | FW2 pass | FW2 reject | total |\n|---|---|---|---|')
        print(f"| repo pass | {c[(False,False)]} | {c[(False,True)]} | {c[(False,False)]+c[(False,True)]} |")
        print(f"| repo reject | {c[(True,False)]} | {c[(True,True)]} | {c[(True,False)]+c[(True,True)]} |")
        print('\n| metric | rate | 95% CI (group bootstrap) |\n|---|---|---|')
        for m, f in metrics.items():
            e, lo, hi = boot(rs, f); print(f"| {m} | {pct(e)} | {pct(lo)} - {pct(hi)} |")
        for side, key in (('repo only', (True, False)), ('FW2 only', (False, True))):
            sel = [r for r in rs if cell(r) == key]
            if not sel: continue
            print(f"\nReasons, {side} rejected (n={len(sel)}; median len {st.median(r['n'] for r in sel):.0f}, median hangul ratio {st.median(r['hr'] for r in sel):.2f}):\n")
            cc = collections.Counter((r['ko'] if key[0] else f"{r['up'][0]}/{r['up'][1]}") for r in sel)
            print('| reason | n | share |\n|---|---|---|')
            for k_, v in cc.most_common(): print(f"| {k_} | {v} | {pct(v/len(sel))} |")
        both = [r for r in rs if cell(r) == (True, True)]
        if both:
            cc = collections.Counter((r['ko'], f"{r['up'][0]}/{r['up'][1]}") for r in both)
            print(f"\nBoth reject: top reason pairs (repo | FW2 first):\n")
            for k_, v in cc.most_common(8): print(f"- {k_[0]} | {k_[1]}: {v}")
        # FW2 per-stage independent rejection
        print('\nFW2 per-stage standalone reject rate (each stage evaluated on every doc, not only survivors):')
        for stg in ['gopher_rep', 'fineweb_qual', 'gopher_qual']:
            print(f"- {stg}: {pct(sum(1 for r in rs if r['stg'][stg] is not None)/len(rs))}")


    ls = [json.loads(l) for l in open(D + 'fw2_langscore.jsonl')]
    print(f"\nfineweb2_ko language_score >= 0.842: {sum(x>=0.842 for x in ls)}/{len(ls)} (min {min(ls):.3f})")

# ---------- (a) gold ----------
gold = rd('gold100_text.jsonl'); gk = rd('gold100_text_ko.jsonl'); gu = rd('gold100_up.jsonl')
G = []
for g, k, u in zip(gold, gk, gu):
    G.append(dict(id=g['id'], source=g['source'], label=g['label'], rr=g['reject_reason'], ko=k['ko'], up=u['first'], stg=u['stages'], n=k['n']))
print('\n## (a) gold100\n')
print('Counts by source:', dict(collections.Counter(g['source'] for g in G)))
def cat(g): return 'keep' if g['label'] == 'keep' else g['rr']
print('\n| label class | n | repo rejects | FW2 rejects | both | repo only | FW2 only | neither |\n|---|---|---|---|---|---|---|---|')
for c in ['keep', 'short', 'semantic_low_quality', 'ALL']:
    rs = [g for g in G if c == 'ALL' or cat(g) == c]
    a = sum(1 for g in rs if g['ko'] and g['up']); b = sum(1 for g in rs if g['ko'] and not g['up'])
    cc = sum(1 for g in rs if not g['ko'] and g['up']); d = sum(1 for g in rs if not g['ko'] and not g['up'])
    print(f"| {c} | {len(rs)} | {a+b} | {a+cc} | {a} | {b} | {cc} | {d} |")
print('\nFilter-vs-label metrics (positive = label reject):\n\n| filter | scope | TP | FP | FN | TN | precision | recall |\n|---|---|---|---|---|---|---|---|')
def met(name, f, scope, rs):
    tp = sum(1 for g in rs if f(g) and g['label'] == 'reject'); fp = sum(1 for g in rs if f(g) and g['label'] == 'keep')
    fn = sum(1 for g in rs if not f(g) and g['label'] == 'reject'); tn = sum(1 for g in rs if not f(g) and g['label'] == 'keep')
    p = f"{tp/(tp+fp):.2f}" if tp + fp else 'n/a'; r = f"{tp/(tp+fn):.2f}" if tp + fn else 'n/a'
    print(f"| {name} | {scope} | {tp} | {fp} | {fn} | {tn} | {p} | {r} |")
for scope, rs in (('all reject labels', G), ('keep + semantic_low_quality only', [g for g in G if cat(g) != 'short']), ('keep + short only', [g for g in G if cat(g) != 'semantic_low_quality'])):
    met('repo check_ko', lambda g: bool(g['ko']), scope, rs)
    met('FW2 reconstruction', lambda g: bool(g['up']), scope, rs)
    met('union (either)', lambda g: bool(g['ko'] or g['up']), scope, rs)
print('\nReasons by label class (repo | FW2 first):')
for c in ['keep', 'short', 'semantic_low_quality']:
    print(f"\n{c}:")
    cc = collections.Counter((g['ko'] or '-', (g['up'][0] + '/' + g['up'][1]) if g['up'] else '-') for g in G if cat(g) == c)
    for k_, v in cc.most_common(): print(f"- repo={k_[0]}, FW2={k_[1]}: {v}")
print('\nBy source (label, repo reject, FW2 reject):')
for s in sorted(set(g['source'] for g in G)):
    rs = [g for g in G if g['source'] == s]
    print(f"- {s}: n={len(rs)} label-reject={sum(g['label']=='reject' for g in rs)} repo-reject={sum(bool(g['ko']) for g in rs)} FW2-reject={sum(bool(g['up']) for g in rs)}")

from math import comb, sqrt
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2*sum(comb(n, i) for i in range(k+1))/2**n)
print('\nPaired recall on label=reject (Wilson 95% CI), discordant counts, exact McNemar p:\n')
print('| class | n | repo recall | FW2 recall | repo-only catches | FW2-only catches | McNemar p |\n|---|---|---|---|---|---|---|')
for c in ['short', 'semantic_low_quality', 'all reject']:
    rs = [g for g in G if (g['label']=='reject' if c=='all reject' else cat(g)==c)]
    kr = sum(bool(g['ko']) for g in rs); fr = sum(bool(g['up']) for g in rs)
    bo = sum(1 for g in rs if g['ko'] and not g['up']); co = sum(1 for g in rs if g['up'] and not g['ko'])
    w1, w2 = wilson(kr, len(rs)), wilson(fr, len(rs))
    print(f"| {c} | {len(rs)} | {kr}/{len(rs)} ({w1[0]:.2f}-{w1[1]:.2f}) | {fr}/{len(rs)} ({w2[0]:.2f}-{w2[1]:.2f}) | {bo} | {co} | {mcnemar_exact(bo, co):.3f} |")

if len(sys.argv)>1 and sys.argv[1]=='b': part_b()
