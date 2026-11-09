# check_ko(+norm_ws) vs FineWeb-2 kor_Hang 필터 재구성 비교

생성: 2026-11-09. 저장소 추적 파일 수정 없음. 모든 원문 jsonl(`compare_datatrove/sample_b.jsonl`, `gold100_text.jsonl`)은 로컬 전용이며 보고서에 인용하지 않음(수치만).

## 0. 핵심 요약
1. **노이즈 플로어(재구성 충실도)**: 이미 FW2 step 5를 통과한 fineweb2_ko 20,000건 중 재구성 FW2 필터가 0.3%(49+10건; CI 0.2–0.4%)를 거절. 이 이하의 차이는 재구성 오차로 봐야 함. 반면 check_ko는 3.7% 거절(주로 short 68%·dup_lines 24%) -> **FW2 이후 데이터 위에서 check_ko가 추가로 빼는 것은 거의 전부 '짧은 문서'와 '중복 줄'**.
2. HPLT(FW2 미경유): FW2 재구성 30.1% 거절 vs check_ko 7.7%. 둘 다 거절 7.0%, FW2만 23.1%, check_ko만 0.7%.
3. wiki: FW2 53.5% 거절 vs check_ko 29.7%. FW2만 28.6%(line_punct_ratio 51%, alpha 비율 34%) -> 웹 텍스트용 필터가 위키(문장부호로 안 끝나는 제목/목록 줄)에 부적합. 위키 거절 다수는 의미 판단이 아닌 도메인 불일치로 보임(가설).
4. 라벨 100건: 의미적 저품질(semantic_low_quality 27) 재현율 check_ko 3/27, FW2 10/27 (check_ko-only 0, FW2-only 7, exact McNemar p=0.016). 단 keep 67건 중 FW2 오거절 7(10.4%) vs check_ko 2(3.0%). short 6건은 둘 다 6/6.
5. FW2가 더 잡지만 정밀도는 낮음(keep+semantic 범위 precision 0.59 vs 0.60; recall 0.37 vs 0.11). union은 FW2 단독 대비 TP 증가 없음(check_ko 추가분은 FP 1건뿐).

## 1. 설치/버전
- datatrove **0.10.1** (venv `/tmp/claude-0/venv_dt`), kiwipiepy **0.21.0** + kiwipiepy_model 0.21.0. kiwipiepy 0.22+는 datatrove KiwiTokenizer 기본 모델 `sbg`를 기본 파일에서 제외(`Cannot open required files for skipbigram model` 확인)하여 0.21.0으로 내림. 
- **FW2가 사용한 datatrove 버전/커밋: 불확실.** FW2 저장소(커밋 d0defb24, 2025-10-27)에는 requirements/lock이 없고 README "Tools versions"는 datatrove 링크만 있음. kiwipiepy 버전도 불확실.

## 2. 업스트림 파이프라인 재구성 (출처: https://github.com/huggingface/fineweb-2/blob/d0defb24f193bb9a5a11b8b14524a03c4858e1b6/fineweb-2-pipeline.py , 설정 .../configs/kor_Hang.yml 같은 커밋)
| 단계 | FW2 구성요소 | 적용 | 비고 |
|---|---|---|---|
| 1 | URLFilter | 건너뜀 | URL/WARC 단계, 입력에 원본 없음 |
| 1 | Trafilatura 추출 | 건너뜀 | |
| 1 | LanguageFilter(ft176) | 건너뜀 | |
| 2 | GlotLID 라벨링 | 건너뜀 | |
| 3 | LambdaFilter language_score >= 0.842 | **fineweb2_ko만 점검**: 20,000건 전부 통과(최소 0.848) / hplt_ko·wiki_ko는 **적용 안 함**(GlotLID 점수 열 없음; HPLT `prob`는 다른 LID라 비교 불가) / gold100은 열 없어 적용 안 함 | 판정 열 비교에는 사용 안 함 |
| 4 | MinHash 중복제거 | 건너뜀 | 문서 간 연산, 문서 단위 판정 불가 |
| 5a | GopherRepetitionFilter(language=kor_Hang, dup_para_frac=0, dup_line_char_frac=0, dup_para_char_frac=0, dup_line_frac=0.374, top_n_grams=[(2,.282),(3,.228),(4,.185)], dup_n_grams=[(5,.201)..(10,.127)]) | 적용 | |
| 5b | FineWebQualityFilter(short_line_thr=999, char_duplicates_ratio=0.1, line_punct_thr=0.062, new_line_ratio=0.196) | 적용 | |
| 5c | GopherQualityFilter(min_avg_word_length=1, max_avg_word_length=6, max_non_alpha_words_ratio=0.791, min_stop_words=2, stop_words=kor_Hang 21개) | 적용 | 나머지 인자는 datatrove 0.10.1 기본값(min_doc_words=50 등) |
| 6 | FTFY, PIIFormatter, SymbolLinesFormatter('|'->줄바꿈) | 건너뜀 | 필터 이후 단계, 공개 데이터는 이 변환 후 텍스트(가설: 재구성 오차 원인 후보) |

순서 5a->5b->5c는 소스 코드 순서 그대로. 각 단계를 독립 평가하고 `first`=첫 실패 단계. 형태소 분할은 datatrove의 kor_Hang 토크나이저(Kiwi sbg)를 사용(속도 때문에 문서당 토큰화 1회 캐시, 결과 동일).

## 3. 샘플
(b) 소스별 row group 20개(seed=77, `R.plan`), 문서 20,000건/소스, 총 60,000건. 리비전: fineweb2_ko HuggingFaceFW/fineweb-2@af9c1333, hplt_ko HPLT/HPLT2.0_cleaned@d1324a52, wiki_ko wikimedia/wikipedia@b04c8d1c. 문서 수는 그룹당 1000건(고정). 업스트림은 원문 raw, 저장소는 `norm_ws` 후 check_ko.
(a) real_gold100: fineweb2_ko 40 / hplt_ko 40 / wiki_ko 20. 라벨: keep 67, reject 33 (short 6, semantic_low_quality 27). 텍스트는 이미 norm_ws 처리본(업스트림도 이 텍스트를 받음 -> (b)와 비대칭).

## 4. (b) 교차표 (행=check_ko, 열=FW2 재구성; 비율 95% CI = 그룹 단위 부트스트랩 2000회, seed 20240)
| 소스 | 문서/그룹 | 둘 다 통과 | 둘 다 거절 | check_ko만 거절 | FW2만 거절 | check_ko 거절률 | FW2 거절률 |
|---|---|---|---|---|---|---|---|
| fineweb2_ko | 20000/20 | 96.0% (95.6–96.4) | 0.1% (0.0–0.1) | 3.7% (3.3–4.1) | 0.2% (0.2–0.3) | 3.7% (3.4–4.1) | 0.3% (0.2–0.4) |
| hplt_ko | 20000/20 | 69.2% (61.0–74.5) | 7.0% (5.8–8.1) | 0.7% (0.5–1.0) | 23.1% (17.7–31.9) | 7.7% (6.4–8.9) | 30.1% (24.8–38.3) |
| wiki_ko | 20000/20 | 41.8% (35.9–48.0) | 24.9% (18.9–31.7) | 4.7% (3.9–5.6) | 28.6% (25.0–32.0) | 29.7% (23.3–36.4) | 53.5% (47.5–59.6) |

건수 (check_ko 통과/거절 x FW2 통과/거절):
- fineweb2_ko: 19204 / 49 ; 737 / 10
- hplt_ko: 13839 / 4622 ; 146 / 1393
- wiki_ko: 8353 / 5716 ; 948 / 4983

## 5. 한쪽만 거절한 문서의 사유 (원문 인용 없음; 길이/한글 비율은 중앙값)
check_ko만 거절:
| 소스 | n | 중앙 길이(자) | 중앙 한글비율 | 사유 분포 |
|---|---|---|---|---|
| fineweb2_ko | 737 | 178 | 0.81 | short 68.7%, dup_lines 24.4%, low_hangul_ratio 6.0%, ngram_rep 0.5%, bad_encoding 0.4% |
| hplt_ko | 146 | 1938 | 0.75 | dup_lines 82.2%, low_hangul_ratio 14.4%, bad_encoding 2.7%, ngram_rep 0.7% |
| wiki_ko | 948 | 166 | 0.81 | short 96.2%, low_hangul_ratio 3.5% |

FW2만 거절 (첫 실패 단계/사유, 상위):
| 소스 | n | 중앙 길이 | 중앙 한글비율 | 상위 사유 |
|---|---|---|---|---|
| fineweb2_ko | 49 | 689 | 0.75 | gopher_below_alpha_threshold 42.9%, list_ratio 26.5%, gopher_rep n-gram류 ~22% |
| hplt_ko | 4622 | 1813 | 0.78 | line_punct_ratio 33.4%, gopher_below_alpha_threshold 24.0%, too_many_end_ellipsis 9.8%, duplicated_5_n_grams 9.4%, 기타 n-gram 반복 ~16% |
| wiki_ko | 5716 | 468 | 0.74 | line_punct_ratio 51.0%, gopher_below_alpha_threshold 34.4%, list_ratio 9.2%, 나머지 <5% |

둘 다 거절한 HPLT 중 가장 흔한 쌍: check_ko dup_lines + FW2 dup_line_frac 627건, ngram_rep + duplicated_5_n_grams 343건. 즉 반복 계열은 두 필터가 겹침.
FW2 단계별 단독 거절률(각 단계를 전 문서에 적용): fineweb2_ko 0.1/0.1/0.1%; hplt_ko 12.4(rep)/18.0(fwq)/15.5(gq)%. wiki_ko 9.0/29.4/40.3%

## 6. (a) 라벨 100건 대조
| 라벨 클래스 | n | check_ko 거절 | FW2 거절 | 둘 다 | check_ko만 | FW2만 | 둘 다 통과 |
|---|---|---|---|---|---|---|---|
| keep | 67 | 2 | 7 | 1 | 1 | 6 | 59 |
| short(규칙형) | 6 | 6 | 6 | 6 | 0 | 0 | 0 |
| semantic_low_quality | 27 | 3 | 10 | 3 | 0 | 7 | 17 |

재현율(reject 라벨 기준, Wilson 95%) 및 짝지은 비교:
| 클래스 | check_ko | FW2 | check_ko만 포착 | FW2만 포착 | McNemar exact p |
|---|---|---|---|---|---|
| short | 6/6 (0.61–1.00) | 6/6 (0.61–1.00) | 0 | 0 | 1.00 |
| semantic_low_quality | 3/27 (0.04–0.28) | 10/27 (0.22–0.56) | 0 | 7 | 0.016 |

keep 대비 오거절: check_ko 2/67, FW2 7/67. 소스별(100건 내): fineweb2_ko 40건은 두 필터 모두 0건 거절(라벨 reject 9건 모두 둘 다 통과 = 두 필터 모두 놓침); hplt_ko 40건 중 라벨 reject 18, check_ko 5, FW2 14; wiki_ko 20건 중 라벨 reject 6, check_ko 6, FW2 9.
FW2가 잡은 semantic 10건의 사유: line_punct_ratio 4, end_ellipsis 2, 5-gram 중복 2, alpha 임계 1, 10-gram 1. **답: 이 100건에서 FW2 재구성이 의미적 저품질을 더 많이 잡음(10 vs 3). 단 n=27이라 불확실성이 크고, 같은 필터가 keep 문서를 더 많이 오거절함(7 vs 2).** short는 동일.
union의 추가 효과: FW2 단독 대비 TP +0, FP +1.

## 7. 해석 (모두 가설)
- check_ko는 길이·한글 비율·중복 줄/n-gram·alnum 비율 등 고정 임계값. FW2는 kor_Hang 통계에서 도출한 임계값 + 형태소(Kiwi) 기반 단어 수. 정의가 달라 임계값 차이만으로 설명되지 않음: (i) `ngram_rep`: check_ko는 공백 토큰 5-gram 반복 비율 > 0.30, FW2는 Kiwi 토큰 기준 중복 문자 비율 > 0.201 등 n별 다른 임계; (ii) 길이: check_ko 200자 vs Gopher 50 Kiwi 토큰(토큰은 형태소라 훨씬 짧은 글자 수; 라벨 short 6건 중 gopher_short_doc은 2건뿐); (iii) FW2에만 있는 줄 끝 문장부호 비율(line_punct_thr=0.062)·말줄임·불릿 검사가 HPLT/wiki의 거절 대부분을 만듦.
- fineweb2_ko에서 FW2 재구성이 0.3% 거절하는 것은 FW2가 이미 통과시킨 문서이므로 재구성 오차(후보: datatrove/kiwi 버전 차이, step 6 이후 텍스트 변형(`|`->줄바꿈, FTFY, PII)). 원인 검증은 하지 않음.
- check_ko의 short·dup_lines 거절은 FW2 통과 문서에서도 3.7%를 추가로 제거 = 저장소 필터의 순증 효과의 대부분. FW2는 HPLT/wiki 같은 비-FW2 소스에서 훨씬 공격적.

## 8. 한계 / 불확실
- FW2가 실제 쓴 datatrove/kiwipiepy 버전·커밋: 불확실(설치: datatrove 0.10.1, kiwipiepy 0.21.0).
- 0.10.1의 기본값(FineWebQualityFilter `line_punct_exclude_zero`, GopherQuality min_doc_words 등)이 FW2 당시와 같은지 불확실.
- GlotLID/언어점수 필터는 fineweb2_ko 열로만 점검, hplt/wiki/gold는 적용 안 함. MinHash·URL·추출·step 6은 재현 안 함.
- 샘플 (b) 업스트림 입력은 raw, gold는 norm_ws 처리본(비대칭). (b)는 소스당 그룹 20개라 CI가 넓음(특히 hplt, wiki).
- gold 대조는 n=100(semantic 27)로 소규모; 라벨은 이 저장소 검수자 기준이라 check_ko 쪽에 유리할 수 있음(불확실).
- Kiwi 토큰화에서 같은 어절이 중복 토큰으로 나오는 현상(예 '갑','갑니다')은 datatrove 0.10.1/kiwi 0.21.0 동작 그대로이며 FW2 당시와 같은지 불확실.
- row group 크기: 모든 소스 그룹당 1000건(wiki 포함)이었음.

## 9. 재현
```
python -m venv /tmp/claude-0/venv_dt && /tmp/claude-0/venv_dt/bin/pip install datatrove kiwipiepy pyyaml regex
/tmp/claude-0/venv_dt/bin/pip install kiwipiepy==0.21.0 kiwipiepy_model==0.21.0
cd /home/user/korean-dense-llm
/tmp/claude-0/venv/bin/python <SCRATCH>/compare_datatrove/sample_b.py <SCRATCH>/compare_datatrove/sample_b.jsonl
/tmp/claude-0/venv/bin/python <SCRATCH>/compare_datatrove/langscore.py <SCRATCH>/compare_datatrove/fw2_langscore.jsonl
/tmp/claude-0/venv/bin/python <SCRATCH>/compare_datatrove/repo_ko.py IN.jsonl OUT.jsonl
/tmp/claude-0/venv_dt/bin/python <SCRATCH>/compare_datatrove/upstream_fw2.py IN.jsonl OUT.jsonl 4   # 설정은 /tmp/claude-0/x/fw2_kor_Hang.yml (compare_datatrove/fw2_kor_Hang.yml 사본)
python3 <SCRATCH>/compare_datatrove/analyze.py b
```
SCRATCH=/tmp/claude-0/-home-user-korean-dense-llm/ad90d990-9b2e-5dbd-be51-a0bd6620df2e/scratchpad. 전체 수치 출력: compare_datatrove/analysis_out.md.
