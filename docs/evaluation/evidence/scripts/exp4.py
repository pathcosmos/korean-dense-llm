import sys,hashlib,unicodedata; sys.path.insert(0,'.'); sys.dont_write_bytecode=True
from data.filters import *
def t(label,s):
    o=mask_pii(s); print(f"{'MASKED   ' if o!=s else 'UNCHANGED'} | {label:30s} | {s!r} -> {o!r}")
for l,s in [("+82-010","+82-010-1234-5678"),("82 02","82-02-123-4567"),("EAN-13 바코드","8801234567893"),("EAN-13 하이픈","880123-4567893"),("공백 숫자표","05 100 2000"),("금액 열거","매출 011 2345 6789 원"),("13자리 주문번호 7번째 1~4","주문번호 2411221234567"),("epoch ms","1700000000123"),("UUID 숫자부","id 123456 3 123456"),("전화 뒤 숫자 붙음","010-1234-56789"),("전화 11자리 앞 숫자","9010-1234-5678"),("영문+전화 개행","\n010-1234-5678\n")]: t(l,s)
print(PHONE.pattern); print(RRN.pattern)
print("== float boundary dup_lines")
x=1-7/10; print(repr(x), x>0.30)
print("20 lines 7 dup:", 1-14/20, 1-14/20>0.30)
print("exact-30% with 10 lines, 4 identical copies means uniq=7:", dedup_lines_stat(["a"]*4+list("bcdefg")))
print("exact 0.30 by 10 lines w/ 3 duplicates removed (uniq=7)")
print("== norm_ws / exact_hash vs dedup fingerprint")
def fp(t): return hashlib.sha256(unicodedata.normalize('NFC',t).encode()).hexdigest()
# need dedup.fingerprint without importing pyarrow? import is fine
from data.dedup import fingerprint
def clean_path_key(t):  # what clean.py does (norm_ws then hash)
    return exact_hash(norm_ws(t))
pairs={
 "대소문자 (Hello/hello)":("Hello World "+"가"*3, "hello world "+"가"*3),
 "NFC/NFD 한글":("한글 문서","한글 문서".encode().decode()),
 "내부 공백 수 (a  b / a b)":("a  b","a b"),
 "탭 vs 공백":("a\tb","a b"),
 "줄바꿈 vs 공백 (a\\nb / a b)":("a\nb","a b"),
 "NBSP vs 공백":("a\u00a0b","a b"),
 "문단 구분 \\n\\n vs \\n":("a\n\nb","a\nb"),
 "CRLF vs LF":("a\r\nb","a\nb"),
 "줄 앞 들여쓰기":("  a\n    b","a\nb"),
 "전각/반각 Ａ vs A":("Ａ","A"),
 "ZWSP 삽입":("a\u200bb","ab"),
 "코드 들여쓰기 (파이썬)":("if x:\n    y()\n","if x:\ny()\n"),
 "Turkish İ 소문자":("İ","i̇"),
 "ß vs ss (lower)":("STRASSE","straße"),
}
nfd=unicodedata.normalize("NFD","한글 문서"); pairs["NFC vs NFD"]=("한글 문서",nfd)
print(f"{'pair':32s} {'exact_hash(raw)':>15s} {'clean.py(norm_ws+hash)':>22s} {'dedup.fingerprint':>18s}  -> clean.py가 같다고 보는데 dedup은 다름?")
for k,(a,b) in pairs.items():
    r1=exact_hash(a)==exact_hash(b); r2=clean_path_key(a)==clean_path_key(b); r3=fingerprint(a)==fingerprint(b)
    print(f"{k:32s} {str(r1):>15s} {str(r2):>22s} {str(r3):>18s}   {'<<GAP(clean=동일,dedup=다름)' if r2 and not r3 else ('<<GAP2(clean=다름,dedup=동일)' if (not r2) and r3 else '')}")
print("== hash collision width: xxh64 on 64bit; birthday for N docs")
import math
for N in (1e8,2.8e8,1.29e8,1e9): print(f"N={N:.3g}  expected colliding pairs ~ N^2/2^65 = {N*N/2**65:.4f}")
print("h - 2**63 range ok:", exact_hash("x")-(1<<63))
# per-shard dedup scope in clean.py: seen is local to shard
