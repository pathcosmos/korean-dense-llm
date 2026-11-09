import sys; sys.path.insert(0,'.'); sys.dont_write_bytecode=True
from data.filters import *
def show(name,t,fn=check_ko):
    print(f"{name:42s} len={len(t):6d} -> {fn(t)}")
KO="오늘은 날씨가 맑고 바람이 시원하게 불어서 공원에서 산책을 하기에 아주 좋은 날이었습니다. "
# 200 boundary
base=("가나다라마바사아자차카타파하"*20)
for n in (199,200,201): show(f"200자 경계 n={n} (반복 음절)",base[:n])
uniq=" ".join(f"단어{i}번째문장내용" for i in range(60))
show("고유 한글 문서", uniq)
for n in (199,200): show(f"고유 n={n}", uniq[:n])
# few_hangul 50 boundary
t=("한"*49)+" ".join(f"word{i}" for i in range(80)); show("한글49자+영어",t)
t=("한"*50)+" "+" ".join(f"word{i}" for i in range(80)); show("한글50자+영어(ratio?)",t)
# ratio 0.30 boundary: construct nonspace count
def mk(hr, total=1000):
    h=int(total*hr); eng=total-h
    hs="".join(chr(0xAC00+(i*37)%2000) for i in range(h))
    es="".join(chr(97+(i*7)%26) for i in range(eng))
    # interleave in chunks of 5 w/ spaces for varied words
    ws=[]; 
    hw=[hs[i:i+4] for i in range(0,len(hs),4)]; ew=[es[i:i+5]+str(i) for i in range(0,len(es),5)]
    return " ".join(hw+ew), h
for hr in (0.28,0.29,0.30,0.31):
    t,h=mk(hr); ns=sum(1 for c in t if not c.isspace()); print(hr,"h/ns=",round(h/ns,3),end=" "); show("ratio",t)
# 존댓말 짧은 글
show("존댓말 150자",KO*2)
show("존댓말 ~230자 (3 반복)",KO*3)
show("존댓말 4반복 (ngram_rep?)",KO*4)
# 시/가사 반복
lyr="\n".join(["사랑해 사랑해 너를 사랑해","영원히 영원히 함께 하고 싶어","눈부신 하늘 아래 우리 둘이서"]*3 + ["마지막 후렴 노래를 불러요 모두 함께","한 번 더 불러요 우리 모두 함께서"]*2+["조용한 밤하늘의 별을 세어보아요 지금","새벽이 올 때까지 이야기해요 우리는"])
show("가사(반복 후렴)",lyr)
poem="\n".join(f"{i}번째 연 바람이 불어오면 나는 그대를 생각하네 깊은 밤에" for i in range(8))*1
show("시 8행 고유",poem+"\n"+poem[:60]*3)
# 표/목록
tbl="\n".join(f"| 항목{i} | {i*1000} | {i%7} | 비고 |" for i in range(40))+"\n"+KO
show("마크다운 표+한글문장",tbl)
lst="\n".join(f"- 항목 {i}: 한국어 설명 문장입니다" for i in range(40)); show("목록 40",lst)
lst2="\n".join(f"- 사과" for i in range(40))+KO*2; show("목록 동일 40줄+문장",lst2)
# 코드 섞임
code="def f(x):\n    return x*2\n"*30 ; show("순수 코드(한글0)",code)
mix=KO*2+"\n```python\n"+"\n".join(f"def func_{i}(a, b):\n    return a + b + {i}" for i in range(30))+"\n```\n"+KO
show("한글설명+코드블록",mix)
# 한자 혼용
hanja="大韓民國의 憲法은 國民의 基本權을 保障하며 國家는 이를 確認하고 保護할 義務를 진다. "*1
hj="".join(chr(0x4E00+i*3) for i in range(300))+" 이다 "*20
show("한자 위주 300+한글",hj)
engmix="This paper introduces 딥러닝 based model for 한국어 처리 and we evaluate on several benchmarks across tasks. "*6
show("영한 혼용(영어 우세)",engmix)
# 영어 문서 한글 50자 + 길이 큼 -> 
en=" ".join(f"Sentence number {i} talks about different topic {i*7}." for i in range(60))
show("영어문서",en); show("영어문서+한글 짧은 서명",en+" 감사합니다 안녕하세요 "*5)
# 일본어/중국어
jp="これは日本語の文章です。"*40; show("일본어",jp)
# 호환 자모 ㅋㅋㅋ
kk="ㅋㅋㅋㅋㅋ ㅎㅎㅎㅎ ㅠㅠㅠ "*30+KO; show("ㅋㅋ 자모(HANGUL 범위 밖)",kk)
# 낱자모 only
jam="ㅋ"*300; show("자모 only",jam)
# NFD hangul
import unicodedata
nfd=unicodedata.normalize('NFD',KO*4+"".join(f"고유{i}문장 " for i in range(40))); show("NFD 한글(정상 문서)",nfd)
nfc=unicodedata.normalize('NFC',nfd); show("  동일 NFC",nfc)
# bad encoding
be=(KO*3+" ".join(f"문서{i}고유" for i in range(30)))
show("U+FFFD 0.1% 미만", be+"�"*0); 
n=len(be); show("FFFD 비율 0.0009", be+"�"*int(n*0.0009)); show("FFFD 비율 0.002", be+"�"*int(n*0.002+3))
# long
show("100001자", ("가"+" 나다라마바 %d"%0)*0+" ".join(f"w{i}한글" for i in range(30000))[:100001])
# alnum: 한글 isalnum True
show("특수문자 많은 한글",("한글문서 !!! ??? ~~~ ### ***\n")*0+" ".join(f"한글{i}!!!!!!!!!" for i in range(40)))
# 숫자표
num=" ".join(str(i*37919) for i in range(200))+" 이 숫자는 통계표의 일부입니다"*3
show("숫자 위주(한글 적음)",num)
# ngram: 5gram 12 words ->  len<20 words returns 0
w=("로 시작하는 문장 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 ").split()
rep=" ".join(w*4); rep+=" "+KO*0; print(len(rep.split()),end=" "); show("20단어 이하 반복",rep.ljust(250,'가'))
