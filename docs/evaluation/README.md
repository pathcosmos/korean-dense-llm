# 데이터 처리 소스 가치 평가

평가 기준일: 2026-11-09. 대상: `data/`, `common/data_contracts.py`, `tokenizer/`, `train/data.py`, 데이터 관련 `scripts/`, 관련 `docs/`와 `tests/`.
평가 방식: 계층별 정적 읽기 + CPU 합성 입력 실험 + 비교군 소스 확인. 저장소 코드는 수정하지 않았다.

## 0. 읽는 법과 한계

- 점수는 1~5이며 주관이 들어간다. 계층별 에이전트 평가를 기초로 하고, 아래 "직접 재현" 표시 항목만 평가자가 다시 실행해 확인했다. 나머지는 에이전트 보고이며 `evidence/`에 근거 `file:line`이 있다.
- 앵커: 1=결함/위험, 2=동작하나 중대 갭, 3=적정, 4=견고(테스트·실패경로 근거), 5=업계 수준 이상이며 차별적.
- **측정 불가는 점수에서 제외했다.** 데이터 본문·토큰·가중치가 저장소에 없으므로 (a) 필터의 실제 코퍼스 정밀도/재현율, (b) 1.29억 문서 규모 성능, (c) 정제 결과가 모델 성능에 주는 효과는 이 평가가 말하지 못한다.
- 합성 실험은 "그런 입력이 존재할 때 이렇게 동작한다"는 존재 증명이지 발생 빈도가 아니다.
- 가중치: 기술 품질 0.4 / 오픈소스·포트폴리오 0.3 / 재사용성 0.3.
- **점수는 계층 간 교정되지 않았다.** 계층마다 다른 평가자가 채점했고, 정제 계층의 재사용성(3.7)은 세 하위 항목(결합도 3 / 의존성 4 / 추출 비용 4)의 평균인 반면 다른 계층의 재사용성(2)은 단일 종합 점수다. 계층 간 비교(예: "어느 계층이 가장 재사용 가능한가")는 서수 수준의 참고용이며 소수점 차이에 의미를 두지 않는다.
- 재현 가능성: `evidence/scripts/`의 실험 스크립트 12개는 저장소 루트에서 실행되며(`python -I docs/evaluation/evidence/scripts/<name>.py`, 의존성은 `requirements-cpu.txt` + CPU torch) 이번에 모두 재실행해 정상 종료를 확인했다. 토큰화 보고서(`03_tokenization.md`)의 실험은 스크립트 파일 없이 인라인으로 수행되어 **저장소에서 재현할 수 없다**. 해당 보고서의 "실험 확인" 항목은 미검증 보고로 취급한다(단 `<|eos|>` 주입은 별도로 재현했다).

## 1. 계층별 점수

| 계층 | A 기술 품질 | B 포트폴리오 | C 재사용성 | 가중 합 |
|---|---:|---:|---:|---:|
| 정제·PII (`filters`, `clean*`, `quality_report`) | 2.3 (정확성 2 / 깊이 2 / 견고성 3) | 3.0 | 3.7 | 2.9 |
| 중복 제거·분리 (`dedup`, `split_*`, `probe_*`) | 4 (깊이 3, 견고성 높음) | 4 | 2 | 3.4 |
| 토큰화·로더 (`tokenizer/`, `tokenize_shards`, `train/data`) | 4 | 4 | 2 | 3.4 |
| 수집·출처·계약·배포 (`release_*`, `scripts/*corpus*`, `data_contracts`) | 3.5 (코어 4 / 운영 래퍼 2) | 3 | 2 (프리미티브 3종은 4) | 2.9 |

가중 합 = 0.4A + 0.3B + 0.3C. 계층 간 단순 평균은 약 3.2이나, 코드량 비중이 계약·배포 쪽에 크므로 평균 자체에 큰 의미를 두지 않는 것이 좋다.

### 사전 가설의 검증

가설: "가치는 정제 알고리즘이 아니라 재현성·계약 인프라에 있고 필터는 얇다."

- **지지:** 정제 계층 기술 품질이 가장 낮고(2.3), 알고리즘이 비율·줄 중복·5-gram·정규식이 전부다. 반면 split·토큰화·계약 쪽은 crash-injection, 트랜잭션 원자성, sha 증명 체인이 테스트로 뒷받침된다.
- **부분 반증:** (1) 인프라도 빈틈이 있다 — `clean.py`는 입력/정책 서명이 없고, dedup 입력 변경 감지는 size+mtime뿐이며, split에는 거대 컴포넌트 방지 게이트가 없다. (2) 재사용성은 오히려 얇은 `filters.py`가 높은 편이다(단일 파일, `re`+`xxhash`뿐, 단독 import 확인. 단 위 교정 한계 때문에 서수 비교다). 계약 인프라는 `common.data_contracts`에 강하게 결합되어 있어 단독 추출 비용이 크다. (3) 계약·게이트 코드의 복잡도 대비 방어 가치가 불균등하다(중복 파일, v1/v2 포크 등 아래 참조).

## 2. 주요 발견 (심각도 순)

"재현" = 평가자가 직접 재실행해 확인. "보고" = 에이전트 보고, 근거는 `evidence/` 참조.

| # | 발견 | 상태 |
|---|---|---|
| 1 | **데이터 텍스트 재배포 권리 판단의 최신 기록이 저장소에 없다 (JUN 확인 필요).** `source_registry.json`은 `upload_authorized:false`, 소스별 `not_cleared_for_public_text_release`이고 `DATA_SOURCES_AND_LICENSES.md`는 "자유 라이선스 공개 근거 미확보"라 쓴다. 반면 `finalize_public_corpus*.py`와 `consolidate_public_corpus.py`는 `corpus_text_public=True`를 설정하고, `PUBLIC_GUIDE.md`는 약 242GB 학습 입력이 별도 HF 배포(`pathcosmos/korean-pretrain-training-data` v1.0.1)에 있다고 안내한다. 권리 판단을 갱신한 기록은 `PUBLIC_GUIDE.md`, `PUBLIC_VALIDATION.md`, `release/*.json`, `NOTICE.md`에서 찾지 못했다. **다만 이것을 모순이라고 단정하지 않는다.** 위 문서들은 "Historical-document context banner"가 붙은 시점 스냅샷(2026-11-08)이고, 현행 `README.md`/`NOTICE.md`는 "데이터 권리는 코드 라이선스가 대신하지 않으며 미확정 항목이 있고 모든 원문의 무조건적 상업 재배포 허가를 주장하지 않는다"는 면책을 이미 두고 있다. 코드 라이선스의 "pending"(`prepare_dataset_release.py`)과 `LICENSE`(MIT)의 차이도 시점 차이로 보이며 결함으로 분류하지 않는다. 남는 문제는 둘 중 하나다: (a) 권리 판단을 별도로 마쳤다면 그 기록이 저장소에 없음, (b) 마치지 않았다면 텍스트 공개 상태와 registry 상태가 어긋남. 적법성 판정은 **불확실**이고 이 평가의 범위 밖이다. | 코드·문서 확인(재현 아님). HF 현황·외부 라이선스 사실은 미확인 |
| 2 | **`clean.py`는 입력/정책 서명이 없다.** `.done`이 있으면 입력·임계값·`lang`이 바뀌어도 이전 stats를 그대로 재사용한다. `clean_code.py`의 `POLICY_HASH`와 설계 수준이 다르다. 임계값을 바꿔 재정제하는 모든 실험의 사고 위험이다. | 보고 + 코드(`clean.py:19-25`) 확인 |
| 3 | **PII 마스킹의 미탐과 오탐.** 미탐: `+82-10-1234-5678`, 외국인등록번호 형태(`900101-5234567`), 카드·계좌·여권·주소·IP. 오탐: `0123456789`, `문서번호 02-2024-0001`, `주문번호 2411221234567`이 `<phone>`/`<id>`로 바뀐다. 체크섬·날짜 검증이 없다. README가 "selected PII"라고 밝힌 범위와는 일치한다. | 재현 |
| 4 | **NFD 한글은 거부된다.** `[가-힣]`이 조합형 음절만 세고 `clean.py`는 필터 전에 NFC 정규화를 하지 않는다. 반복 없는 1,849자 한국어 문서가 NFC에서는 `check_ko`를 통과(`None`)하고, 같은 문서를 NFD로 바꾸면 `few_hangul`로 거부된다. 실제 코퍼스의 NFD 비율은 측정 불가. | 재현 (비반복 문서로 확인) |
| 5 | **본문의 `<\|eos\|>` 문자열이 실제 EOS ID로 토큰화된다.** 코드는 `add_special_tokens=False`를 쓰지만 본문 안의 특수 토큰 매칭은 막지 못한다. 인코딩 시점 차단은 없고 사후 EOS 개수 대조(`release_offsets.py`)로만 탐지한다. 토이 WordLevel 토크나이저에서 재현했고 실제 BPE에서는 미확인이다. 업스트림이 해당 문자열을 제거하는지 **불확실**. | 부분 재현 |
| 6 | **split에 거대 연결 컴포넌트 방지 장치가 없다.** 합성 실험에서 코드 repo의 30%가 같은 `MIT License` 파일을 가지면 60개 repo가 한 컴포넌트로 묶여 통째로 train 또는 eval로 간다(eval 비율 5.7~8.7%로 흔들림). `split_distribution*.py`는 보고만 하고 실패 게이트가 아니다. 실제 코드 데이터에서의 규모는 **불확실**. | 재현 (`evidence/scripts/exp_split.py` 재실행: 한 컴포넌트에 240문서·repo 60개, 전부 train). 실제 데이터 규모는 미측정 |
| 7 | **near-duplicate는 측정(probe)만 하고 split 그룹에 반영되지 않는다.** 구현된 것은 NFC SHA256 exact와 raw·NFC 동등·코드 repo 연결까지다. 대소문자·공백만 다른 문서쌍이 모두 생존하고 train/eval 누수 가능성이 남는다. `probe_similarity_recall`은 합성/실제 분모 분리와 양성 0일 때 null 보고라는 좋은 설계를 가지지만 앞 2048자만 비교하고 임계값이 0.8 이상뿐이다. | 보고 |
| 8 | dedup 입력 변경 감지가 size+mtime_ns뿐이다. 크기·mtime이 같고 내용만 바뀌면 조용히 SKIP된다. split 계층은 전체 SHA256을 쓰므로 일관성이 없다. | 보고 |
| 9 | 코드 정제의 secret 탐지는 3계열(RSA/PKCS 개인키, `AKIA`, `ghp_`)뿐이다. 일부 정책(인코딩 비율, 3000자 윈도우 등)은 `POLICY_HASH`에 포함되지 않는다. | 보고 |
| 10 | 필터 정확성을 직접 검증하는 테스트(`check_ko`, `mask_pii`, `norm_ws`)가 없다. 전체 테스트는 257 통과 / 1 실패 / 35 skip이고 실패 1건은 `RAYON_NUM_THREADS` 미설정 환경 의존이다(재현, 변수 설정 시 통과). skip 35건은 별도 파일시스템·원본 학습 산출물을 요구해 이 환경에서 미검증이다. | 재현 |
| 11 | 중복·포크 코드: `split_manifest_candidate.py`는 `split_manifest.py`와 바이트 동일(`cmp` 확인). `finalize_public_corpus.py`와 `_v2`는 사실상 포크. `assert` 기반 검증이 `python -O`에서 무력화된다. `atomic_json`이 `data_contracts`와 `upload_public_corpus`에 따로 있고 전자는 고정 `.tmp` 이름이라 동시 쓰기 시 `FileNotFoundError`가 재현됐다. | 일부 재현(`cmp`) |
| 12 | registry가 가리키는 `evidence/*.md` 등이 저장소에 없어 CLI 일부는 클린 clone에서 실행할 수 없고 해시는 독립 검증 불가(측정 불가). 이메일·`pathcosmos` 계정·`runs/…` 경로가 코드에 박혀 있다. | 보고 |
| 13 | 한국어 토크나이저 품질 측정이 얇다. 카테고리별 chars/bytes per token과 baseline 대비 상대 효율 게이트만 있고 어절당 fertility, NFD 자모 probe, 외부 토크나이저 비교가 없다. f5 결과에 실측 파일이 없어 실제 한국어 수치는 측정 불가. vocab 65536은 uint16에 여유 ID가 0이다(3중 가드는 확인). | 보고 |

**후속 조치 (2026-11-09)**: 위 발견 중 #2(`clean.py` 서명), #3(PII 옵트인 정책), #4(NFD 옵트인), #10(필터 직접 테스트)은 소스를 고쳤고, #4와 #5는 T1 측정(16만 문서)으로 영향이 거의 없음이 확인되어 우선순위를 낮췄다. 상세와 열린 항목은 [IMPROVEMENTS.md](IMPROVEMENTS.md), 측정 표는 [t1/RESULTS.md](t1/RESULTS.md). 가장 큰 새 발견은 의미적 저품질 문서를 규칙 필터가 거의 못 잡는다는 점이다(에이전트 라벨 기준, 검수 전).

강점(근거는 `evidence/`): 크래시 재개와 변조 거부(dedup, 토큰화 shard, 업로드 receipt), sha 결합 리뷰 게이트, 코드 정제의 byte-preserving 정책과 `POLICY_HASH`, 샘플러 결정성(`default_rng([seed,step,rank,micro])`), 합성/실제 분모를 분리 보고하는 probe 철학, 한계를 숨기지 않는 문서 서술.

## 3. 비교군 대조

상세와 URL은 `evidence/05_comparators.md`. 소스 파일을 직접 읽은 범위의 요약이다.

- **FineWeb-2 / datatrove (주 비교군):** 이 저장소의 입력 `fineweb2_ko`는 이미 이 파이프라인을 거쳤다. `kor_Hang.yml`에는 한국어 전용 통계 임계값이 있다(`language_score` 0.842, `dup_line_frac` 0.374, `line_punct_thr` 0.062, `new_line_ratio` 0.196, `max_non_alpha_words_ratio` 0.791, n-gram별 중복 비율, 한국어 불용어, `max_avg_word_length` 6). 이 값은 원본 YAML에서 직접 확인했다. MinHash는 `n_grams=5`, 14 bucket × 8, 언어별 전역 중복 제거이며 클러스터 크기를 메타데이터로 남긴다. 반면 이 저장소의 `check_ko`는 고정 임계값(길이 200, 한글 비율 0.30, 줄 중복 0.30, 5-gram 반복 0.30, 영숫자 0.5)이며 근거 문서가 없다.
  - 해석: 이 저장소의 정제는 FineWeb-2의 대체가 아니라 소스 정규화 계층(다소스 결합, 위키·HPLT 정합, 코드·수학 정책)에 가깝다. 추가 가치는 정제 알고리즘이 아니라 다소스 exact dedup, 코드 byte-preserving, split 누수 방지, 계약 기반 provenance에 있다. HPLT↔FineWeb-2 근사 중복은 LESSONS #7이 스스로 지적한 열린 문제다.
- **NeMo Curator:** exact/fuzzy(MinHash/LSH, GPU)/semantic dedup, 17종 heuristic 필터, 영어 기본값 중심. 한국어 전용 설정과 한국어 PII는 확인한 범위에서 없었다. API가 0.x→1.x(Ray 기반)로 바뀐 이력이 있다.
- **Data-Juicer:** MinHash/SimHash, 다수 operator. 한국어 전용 자산 여부는 불확실(`alibaba/data-juicer`는 프록시 403이라 `modelscope/data-juicer`를 읽었고 동일성 미확인).
- **Dolma:** Bloom filter 기반 exact, MinHash 없음. **text-dedup:** 단어 n-gram 기반이라 한국어는 어절 shingle이 되며 문자 n-gram·형태소 옵션이 없다.
- **한국어 PII(주민번호·휴대폰):** 5개 도구 모두 읽은 범위에서 기본 제공이 없었다. 완전한 부재 보증은 아니다. 이 저장소도 정규식 3개 수준이라 현재는 차별점이라기보다 같이 비어 있는 영역이며, 오탐·미탐을 고치고 검증 가능한 한국어 PII 모듈로 만들면 실제 차별점이 될 수 있다.
- **확인 불가:** arXiv 2510.24139(한국어 CommonCrawl 필터링)와 GECKO(2405.15640) 본문. 이 세션 네트워크에서 arxiv.org에 접근하지 못했다(허용 목록 추가 후에도 연결 거부). 이 두 비교는 **불확실**로 남긴다.

## 4. 격차와 조치 후보 (효과 대비 순)

코드 변경은 이번 범위에 포함하지 않았다. 아래는 승인 후 구현할 후보이며 일정은 두지 않는다.

| 우선 | 조치 | 이유 |
|---|---|---|
| 1 | JUN 확인: 데이터 텍스트 재배포 권리 판단을 별도로 마쳤는지. 마쳤다면 그 기록을 저장소에 반영하고, 아니라면 registry 상태와 공개 상태를 정합 | 비용은 작고 영향은 큼. 결정은 JUN 몫이며 적법성 판단은 이 평가의 범위 밖 |
| 2 | `clean.py`에 입력·정책 서명 추가, 임계값을 설정으로 분리하고 정책 해시에 포함 | 피드백 루프의 선결 조건이자 사고 방지 |
| 3 | `check_ko`/`mask_pii`/`norm_ws` 직접 테스트와 gold set 기반 precision/recall 측정 | 정제 계층의 가장 큰 약점이자 "측정 불가"를 줄이는 길 |
| 4 | PII 정규식 개선(`+82`, 외국인번호, 오탐 체크섬/날짜 검증)과 한국어 PII 모듈 분리 | 비교군에도 없는 영역이라 차별화 후보 |
| 5 | 인코딩 시점 특수 토큰 문자열 차단 또는 정제 단계 제거 | 데이터 오염 방지 |
| 6 | split 거대 컴포넌트 게이트, dedup 입력 SHA 검증, near-dup을 split 그룹에 반영할지 결정 | 평가 누수 방지 |
| 7 | 중복 파일 제거, finalize v1/v2 통합, `assert`→예외, `atomic_json` 단일화 | 유지보수·재사용성 |
| 8 | NFC 정규화 정책 명시(NFD 입력 처리), 한국어 토크나이저 fertility 지표 추가 | 한국어 품질 측정 공백 |

## 5. 피드백 루프

측정→반영→수정 장치는 [FEEDBACK_LOOP_DESIGN.md](FEEDBACK_LOOP_DESIGN.md)에 설계했고, T0 하네스는 [`dataqa/`](../../dataqa/README.md)에 구현했다(신규 모듈, 기존 파일 무수정). 이 세션에서 실제로 실행할 수 있는 범위(T0, T1 일부)와 불가능한 범위(T2, GPU·데이터 필요)를 구분한다.

## 6. 측정 불가 / 불확실 목록

- 실제 코퍼스에서의 필터 정밀도·재현율, NFD 비율, near-dup 양성 수, 거대 컴포넌트 크기
- 1.29억 문서 규모의 dedup/split 성능(LESSONS #19가 외삽 금지를 스스로 경고)
- 한국어 토크나이저의 실제 fertility·효율 수치(결과 파일 부재)
- 정제·dedup 변경이 모델 성능에 미치는 영향(학습 필요)
- 외부 라이선스 사실(Wikimedia, HPLT, ODC-By 등)의 판정, 공개 배포의 적법성
- arXiv 2510.24139, GECKO 본문과의 비교
- 임계값 외부화 시 코드 SHA 재산출이 필요한 스크립트 범위(`audit_retention_recovery.py`, `prepare_public_recipe.py`)
- 건너뛴 테스트 35건이 검증하는 동작

## 7. 근거 파일

`evidence/`의 다섯 보고서는 계층별 평가의 원본이다. 점수마다 `file:line`이 있으나 에이전트가 작성한 것이므로 이 문서의 "재현" 표시 외 항목은 독립 검증 전 상태다.
