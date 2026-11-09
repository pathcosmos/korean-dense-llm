import sys,random,unicodedata,string; sys.path.insert(0,'.'); sys.dont_write_bytecode=True
from data.filters import *
R=random.Random(7)
SY=[chr(0xAC00+i*11) for i in range(200)]
def kw(): return "".join(R.choice(SY) for _ in range(R.randint(2,4)))
def ew(): return "".join(R.choice(string.ascii_lowercase) for _ in range(R.randint(3,8)))
def show(name,t,fn=check_ko,extra=""):
    ns=sum(1 for c in t if not c.isspace()); h=len(HANGUL.findall(t))
    al=sum(c.isalnum() for c in t)
    print(f"{name:44s} len={len(t):6d} h={h:5d} h/ns={h/max(ns,1):.3f} alnum/n={al/len(t):.3f} -> {fn(t)} {extra}")
def doc_ratio(target,total_ns=1000):
    # words until hangul count hits target*total_ns exactly
    hs=int(round(target*total_ns)); toks=[];h=0;e=0
    while h<hs:
        w=kw(); toks.append(w); h+=len(w)
    while h+e<total_ns:
        w=ew(); toks.append(w); e+=len(w)
    R.shuffle(toks); return " ".join(toks)
print("--- 한글비율 0.30 경계 (정확 구성)")
for tg in (0.28,0.299,0.30,0.31,0.35):
    show(f"target {tg}",doc_ratio(tg))
# find exact <0.30 vs >=0.30 with controlled counts
def exact(h,e):
    toks=[]; 
    hh=0
    while hh<h:
        w=kw(); w=w[:h-hh]; toks.append(w); hh+=len(w)
    ee=0
    while ee<e:
        w=ew(); w=w[:e-ee]; toks.append(w); ee+=len(w)
    R.shuffle(toks); return " ".join(toks)
for h,e in ((299,701),(300,700),(301,699)):
    show(f"h={h} e={e}",exact(h,e))
print("--- few_hangul 50 경계 (비율 높고 짧음)")
for h in (49,50):
    show(f"h={h}, 길이200+ 공백/숫자로 채움", exact(h,0)+" "+" ".join(str(R.randint(1000,99999)) for _ in range(40)))
print("--- 한글 위주지만 한글 49자 (짧은 한글 문서 200자 이상 가능?)")
show("한글 49 + 공백패딩", exact(49,0)+" "*0)
print("--- 한자혼용 (실제 한국어 한자어)")
HJ="大韓民國 憲法 國民 基本權 保障 國家 確認 保護 義務"
def hanja_doc(frac):
    toks=[]
    for _ in range(120):
        toks.append(R.choice(HJ.split()) if R.random()<frac else kw())
    return " ".join(toks)
for fr in (0.2,0.5,0.7):
    show(f"한자어 비중 {fr}",hanja_doc(fr))
print("--- 영어 학술 + 한글 요약")
en=" ".join(ew() for _ in range(300)); ko=" ".join(kw() for _ in range(40))
show("영어300어 + 한글40어",en+"\n"+ko)
print("--- 코드섞인 글")
code="\n".join(f"    result_{i} = compute_value({i}, offset={i*3})" for i in range(25))
prose=" ".join(kw() for _ in range(120))
show("산문120어 + 코드25줄",prose+"\n"+code)
show("산문60어 + 코드25줄"," ".join(kw() for _ in range(60))+"\n"+code)
print("--- 존댓말 짧은 글 (유효, 고유)")
for n in (40,55,70):
    t=" ".join(kw()+"습니다" for _ in range(n)); show(f"고유 {n}어 존댓말",t)
print("--- 200자 이상이지만 한글 50자 미만 짧은 한국어 공지")
t="안녕하세요 "+" ".join(str(R.randint(100,999)) for _ in range(60)); show("한글 5자 + 숫자",t)
print("--- 시: 후렴 반복 / 고유 행 20")
verse=[" ".join(kw() for _ in range(5)) for _ in range(6)]
song="\n".join(sum([verse[:4]+[verse[4+(i%2)]] for i in range(5)],[]))
show("가사 후렴(4줄)×5 + 절",song)
poem="\n".join(" ".join(kw() for _ in range(4)) for _ in range(20)); show("시 20행 고유",poem)
poem_ref="\n".join([" ".join(kw() for _ in range(4)) if i%3 else "님은 갔습니다 아아 사랑하는 나의 님은 갔습니다" for i in range(24)])
show("시 24행, 1/3 행이 반복 후렴",poem_ref)
print("--- 표/목록")
tbl="\n".join("| "+" | ".join(kw() for _ in range(4))+" |" for _ in range(40)); show("한글 표 40행 고유",tbl)
tblnum="\n".join(f"| {kw()} | {R.randint(1000,99999)} | {R.randint(0,99)} | {R.random():.3f} |" for _ in range(40)); show("한글+숫자 표",tblnum)
lst="\n".join(f"- 항목 {i}: 한국어 설명 {kw()}입니다" for i in range(40)); show("템플릿 목록(번호만 고유)",lst)
tem="\n".join(f"{i}. 이 제품은 최고의 품질을 자랑하는 상품입니다 지금 주문하세요 {R.randint(10,99)}" for i in range(30)); show("템플릿 스팸 30줄",tem)
print("--- NFD 한국어(정상 문서)")
base=" ".join(kw() for _ in range(120)); nfd=unicodedata.normalize("NFD",base)
show("NFC",base); show("NFD 동일 문서",nfd)
print("--- 호환 자모/반각")
show("ㅋㅋ/자모 문서 + 한글 120어", " ".join(kw() for _ in range(120))+" ㅋㅋㅋㅋㅋㅋㅋㅋ"*0)
print("--- 인코딩")
b=" ".join(kw() for _ in range(120)); n=len(b)
for k in (int(n*0.001),int(n*0.001)+1,int(n*0.01)): show(f"FFFD x{k}",b+"�"*k)
print("--- max_chars 경계(100000)")
big=" ".join(kw() for _ in range(40000))
t=big[:100000]; show("100000자",t); show("100001자",big[:100001])
print("--- 중복라인 0.30 경계 (>5줄)")
def dl(total,dups):
    L=[" ".join(kw() for _ in range(6)) for _ in range(total-dups)]+[ "반복되는 줄 입니다 이것은 동일" ]*dups
    return "\n".join(L)
for tot,d in ((10,3),(10,4),(20,7)):
    # 1-uniq/n: dup lines d identical copies -> uniq = total-d+1
    t=dl(tot,d); n=tot; uniq=len(set(t.split("\n")))
    show(f"{tot}줄 중 동일 {d}줄 (dup_stat={1-uniq/n:.2f})",t)
print("--- 5줄 이하는 dup_lines 면제")
t="\n".join(["같은 줄 입니다 한글 문서 내용이 길게 이어집니다 "*3]*5)+"\n"+" ".join(kw() for _ in range(60)); show("5줄 동일 + 산문",t)
