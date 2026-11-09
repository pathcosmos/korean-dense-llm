import sys,unicodedata; sys.path.insert(0,'.'); sys.dont_write_bytecode=True
from data.filters import *
from data.dedup import fingerprint
import pyarrow as pa, pyarrow.parquet as pq, io
def within(x): return exact_hash(norm_ws(x))
def cross(x): return fingerprint(mask_pii(norm_ws(x)))
P={
"case Hello/hello":("Hello World","hello world"),
"internal 2 spaces":("a  b","a b"),
"tab vs space":("a\tb","a b"),
"NBSP vs space":("a\u00a0b","a b"),
"newline vs space (same line wrap)":("a\nb","a b"),
"\\n\\n vs \\n":("a\n\nb","a\nb"),
"3+ newlines vs 2":("a\n\n\n\nb","a\n\nb"),
"CRLF vs LF":("a\r\nb","a\nb"),
"leading indent":("  a\n    b","a\nb"),
"trailing ws lines":("a  \nb","a\nb"),
"Turkish İ":("İ","i̇"),
"phone only differs":("call 010-1234-5678 now","call 010-9999-0000 now"),
"email only differs":("a@x.com hi","b@y.org hi"),
"Latin NFC vs NFD (café)":("caf\u00e9 au lait","cafe\u0301 au lait"),
"Korean NFC vs NFD":("한글 문서 입니다","한글 문서 입니다".encode().decode()),
}
P["Korean NFC vs NFD"]=("한글 문서 입니다",unicodedata.normalize("NFD","한글 문서 입니다"))
print(f"{'pair':36s} within-shard(clean.py) cross-shard(dedup.py)  판정")
for k,(a,b) in P.items():
    w=within(a)==within(b); c=cross(a)==cross(b)
    tag="일치" if w==c else ("within만 병합 (dedup은 둘 다 유지)" if w else "dedup만 병합 (within은 둘 다 유지)")
    print(f"{k:36s} {str(w):>18s} {str(c):>22s}  {tag}")
print("== lone surrogate reachable via arrow?")
for fn in (lambda: pa.array(['\ud800']), lambda: pa.table({'t':['\ud800']})):
    try: fn(); print("pa accepted")
    except Exception as e: print("pa rejects:",type(e).__name__)
