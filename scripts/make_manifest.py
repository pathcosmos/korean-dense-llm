"""Write data/MANIFEST.json (size + xxh64 of each file under data/clean, data/tokens, tokenizer/). 
Run before moving, and `python scripts/make_manifest.py --verify` after moving."""
import os, sys, json, glob, xxhash
from concurrent.futures import ProcessPoolExecutor
from common.paths import PROJECT_ROOT
def h(p):
    x = xxhash.xxh64()
    with open(p, "rb") as f:
        while b := f.read(1 << 24): x.update(b)
    return p, os.path.getsize(p), x.hexdigest()
if __name__ == "__main__":
    pats = ["data/clean/**/*.parquet", "data/tokens/**/*.bin", "data/tokens/mix.json", "tokenizer/*.json"]
    files = sorted({os.path.relpath(f, PROJECT_ROOT) for p in pats for f in glob.glob(os.path.join(PROJECT_ROOT, p), recursive=True)})
    mp = os.path.join(PROJECT_ROOT, "data/MANIFEST.json")
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(h, [os.path.join(PROJECT_ROOT, f) for f in files]))
    cur = {os.path.relpath(p, PROJECT_ROOT): {"size": s, "xxh64": d} for p, s, d in res}
    if "--verify" in sys.argv:
        man = json.load(open(mp)); bad = [k for k, v in man.items() if cur.get(k) != v]
        print("verified", len(man), "files; mismatches:", bad[:20] or "none"); sys.exit(1 if bad else 0)
    json.dump(cur, open(mp, "w"), indent=1); print("manifest", len(cur), "files")
