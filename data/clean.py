"""Clean raw parquet shards in parallel. Resumable (skips shards with .done marker).
usage: python data/clean.py <src_name> <lang ko|en|math> [workers]
Output: data/clean/<src>/<shard>.parquet with columns [text, h]  + <shard>.stats.json
"""
import sys, os, glob, json, collections, time
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from filters import *

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); from common.paths import PROJECT_ROOT, DATA
ROOT = DATA

def work(args):
    src, lang, f = args
    name = os.path.basename(f).replace(".parquet", "")
    out = f"{ROOT}/clean/{src}/{name}.parquet"
    if os.path.exists(out + ".done"):
        if not os.path.exists(out):
            raise RuntimeError(f"Completion marker without output: {out}")
        pq.ParquetFile(out)  # validate footer before skipping
        with open(out + ".stats.json") as sf:
            stats = json.load(sf)
        return name, (collections.Counter(stats["counts"]), stats["chars_kept"])
    if os.path.exists(out):
        raise RuntimeError(f"Unmarked output exists; refusing overwrite: {out}")
    chk = check_ko if lang == "ko" else (check_math if lang == "math" else check_en)
    cnt = collections.Counter(); seen = set(); chars = 0
    schema = pa.schema([("text", pa.string()), ("h", pa.int64())])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with pq.ParquetWriter(out + ".tmp", schema, compression="zstd") as writer:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["text"]):
            texts = []; hs = []
            for text in batch.column(0).to_pylist():
                if text is None:
                    cnt["null"] += 1
                    continue
                text = norm_ws(text)
                reason = chk(text)
                if reason is None:
                    h = exact_hash(text)
                    if h in seen:
                        reason = "exact_dup"
                    else:
                        seen.add(h)
                        text = mask_pii(text)
                        texts.append(text); hs.append(h - (1 << 63)); chars += len(text)
                cnt[reason or "kept"] += 1
            if texts:
                writer.write_table(pa.table({"text": texts, "h": hs}, schema=schema))
    # Keep the existing hash semantics; cross-source dedup is a separate stage.
    os.replace(out + ".tmp", out)
    with open(out + ".stats.json.tmp", "w") as sf:
        json.dump({"counts": cnt, "chars_kept": chars}, sf)
    os.replace(out + ".stats.json.tmp", out + ".stats.json")
    open(out + ".done", "w").close()
    return name, (cnt, chars)

if __name__ == "__main__":
    src, lang = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    files = sorted(glob.glob(f"{ROOT}/raw/{src}/**/*.parquet", recursive=True))
    print(len(files), "shards", flush=True); t0 = time.time()
    tot = collections.Counter(); chars = 0
    if lang not in ("ko", "en", "math") or nw < 1 or not files:
        raise SystemExit("Require lang ko/en/math, positive workers, and at least one input parquet")
    with ProcessPoolExecutor(nw) as pool:
        jobs = [pool.submit(work, (src, lang, f)) for f in files]
        for i, job in enumerate(as_completed(jobs)):
            name, res = job.result()
            if res: tot.update(res[0]); chars += res[1]
            print(f"[{i+1}/{len(files)}] {name} {time.time()-t0:.0f}s kept_chars={chars/1e9:.2f}B", flush=True)
    print(dict(tot))
