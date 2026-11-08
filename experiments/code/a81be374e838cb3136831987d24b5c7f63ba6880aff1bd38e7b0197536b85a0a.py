"""Sample N docs from one parquet and report reject-reason rates."""
import sys, collections, glob, pyarrow.parquet as pq
sys.path.insert(0, "data")
from filters import *
f = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
p = pq.ParquetFile(f); cnt = collections.Counter(); kept_chars = tot_chars = 0; ex = collections.defaultdict(list)
seen = 0
for rg in range(p.num_row_groups):
    for t in p.read_row_group(rg, columns=["text"]).column("text").to_pylist():
        t = norm_ws(t); r = check_ko(t); cnt[r] += 1; tot_chars += len(t)
        if r is None: kept_chars += len(t)
        elif len(ex[r]) < 3: ex[r].append(t[:200].replace("\n", " | "))
        seen += 1
        if seen >= N: break
    if seen >= N: break
for k, v in cnt.most_common(): print(f"{str(k):18s}{v:7d} {v/seen:6.1%}")
print("char keep ratio", kept_chars / tot_chars, "avg chars", tot_chars / seen)
for k, v in ex.items():
    print("##", k)
    for s in v: print("  ", s)
