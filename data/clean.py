"""Clean raw parquet shards in parallel. Resumable (skips shards with .done marker).
usage: python data/clean.py <src_name> <lang ko|en|math> [workers] [--policy policy.json]
Output: data/clean/<src>/<shard>.parquet with columns [text, h]  + <shard>.stats.json

Every output records a signature (input file sha256, lang, filter thresholds, PII policy, filters.py
sha256, CLEAN_LOGIC_VERSION). A `.done` shard is reused only if its signature equals the current one;
a different or missing signature (legacy output) raises instead of silently reusing stale results.
--policy takes JSON {"ko": {KoPolicy fields}, "pii": {PiiPolicy fields}}; omitted fields keep the
historical defaults, which are pinned by tests/golden/filters_oracle_v1.json.
"""
import sys, os, glob, json, collections, time, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from filters import *

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); from common.paths import PROJECT_ROOT, DATA
ROOT = DATA
CLEAN_LOGIC_VERSION = 1  # bump when the per-document sequence in work() changes (not for refactors)
POLICY = {"ko": KO_DEFAULT, "pii": PII_DEFAULT}  # default; main() passes the active policy to workers explicitly


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def signature(lang, input_path, policy=None):
    """Everything the cleaned output depends on, as a dict plus its content id."""
    policy = policy or POLICY
    body = dict(
        logic_version=CLEAN_LOGIC_VERSION, lang=lang,
        input=dict(name=os.path.basename(input_path), bytes=os.path.getsize(input_path), sha256=_sha256(input_path)),
        ko_policy=policy["ko"].as_dict(), pii_policy=policy["pii"].as_dict(),
        filters_sha256=_sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "filters.py")))
    body["id"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def load_policy(path):
    cfg = json.loads(open(path, encoding="utf-8").read())
    unknown = set(cfg) - {"ko", "pii"}
    if unknown:
        raise SystemExit(f"unknown policy sections: {sorted(unknown)}")
    return {"ko": KoPolicy(**cfg.get("ko", {})), "pii": PiiPolicy(**cfg.get("pii", {}))}


def work(args):
    # (src, lang, file[, policy]); the policy travels with the job so spawn/forkserver workers cannot
    # silently fall back to the module default.
    src, lang, f, *rest = args
    policy = rest[0] if rest else POLICY
    name = os.path.basename(f).replace(".parquet", "")
    out = f"{ROOT}/clean/{src}/{name}.parquet"
    sig = signature(lang, f, policy)
    if os.path.exists(out + ".done"):
        if not os.path.exists(out):
            raise RuntimeError(f"Completion marker without output: {out}")
        pq.ParquetFile(out)  # validate footer before skipping
        with open(out + ".stats.json") as sf:
            stats = json.load(sf)
        recorded = stats.get("signature", {}).get("id")
        if recorded is None:
            raise RuntimeError(f"Legacy output without policy signature; refusing silent reuse: {out}")
        if recorded != sig["id"]:
            raise RuntimeError(f"Output signature differs from current input/policy/code; use a new output tree: {out}")
        return name, (collections.Counter(stats["counts"]), stats["chars_kept"])
    if os.path.exists(out):
        raise RuntimeError(f"Unmarked output exists; refusing overwrite: {out}")
    if lang == "ko":
        chk = lambda text, _p=policy["ko"]: check_ko(text, policy=_p)
    else:
        chk = check_math if lang == "math" else check_en
    pii_policy = policy["pii"]
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
                        text = mask_pii(text, pii_policy)
                        texts.append(text); hs.append(h - (1 << 63)); chars += len(text)
                cnt[reason or "kept"] += 1
            if texts:
                writer.write_table(pa.table({"text": texts, "h": hs}, schema=schema))
    # Keep the existing hash semantics; cross-source dedup is a separate stage.
    os.replace(out + ".tmp", out)
    with open(out + ".stats.json.tmp", "w") as sf:
        json.dump({"counts": cnt, "chars_kept": chars, "signature": sig}, sf)
    os.replace(out + ".stats.json.tmp", out + ".stats.json")
    open(out + ".done", "w").close()
    return name, (cnt, chars)

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--policy" in argv:
        i = argv.index("--policy")
        POLICY.update(load_policy(argv[i + 1]))
        del argv[i:i + 2]
    src, lang = argv[0], argv[1]
    nw = int(argv[2]) if len(argv) > 2 else 10
    files = sorted(glob.glob(f"{ROOT}/raw/{src}/**/*.parquet", recursive=True))
    print(len(files), "shards", flush=True); t0 = time.time()
    tot = collections.Counter(); chars = 0
    if lang not in ("ko", "en", "math") or nw < 1 or not files:
        raise SystemExit("Require lang ko/en/math, positive workers, and at least one input parquet")
    with ProcessPoolExecutor(nw) as pool:
        jobs = [pool.submit(work, (src, lang, f, POLICY)) for f in files]
        for i, job in enumerate(as_completed(jobs)):
            name, res = job.result()
            if res: tot.update(res[0]); chars += res[1]
            print(f"[{i+1}/{len(files)}] {name} {time.time()-t0:.0f}s kept_chars={chars/1e9:.2f}B", flush=True)
    print(dict(tot))
