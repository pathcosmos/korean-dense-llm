# 평가: 중복 제거 및 train/eval 분리·누수 방지 계층 (읽기 전용)

방법: 대상 파일 전체 정독 + 합성 데이터 직접 실행(`exp_dedup.py`, `exp_split.py`, 같은 디렉터리). 테스트 스위트는 실행하지 않음(요청). 저장소 수정 없음.

## 점수표

| 축 | 점수 | 한 줄 근거 |
|---|---|---|
| A 기술품질 | **4** (규모 성능은 측정 불가) | 정확성/재개/fault injection은 견고. near-dup 제거와 giant-component 방지는 미구현 |
| B 포트폴리오 | **4** | 실패경로·증명 중심 설계와 정직한 scope 문구가 차별점. 알고리즘 깊이(MinHash/LSH 미구현)는 5 미만 |
| C 재사용성 | **2** | 한 줄 스타일(`;` 연결) 코드, 프로젝트 내부 contract 계층(`common.data_contracts`)에 강결합, 중복 파일 존재 |


## 1. exact(NFC SHA256)만으로 부족한 지점

- `data/dedup.py:21-23` 지문은 NFC 정규화 후 SHA256, 대소문자·공백 보존. 실행 결과(합성): `Hello world` / `hello world` / ` Hello world` / `Hello  world` / `Hello world\n` 5개는 **모두 생존**, NFD `café`(e+U+0301)만 NFC 동일로 제거. 즉 대소문자·앞뒤/중복 공백·개행·BOM(`a﻿`)·NBSP 변형은 exact 계층이 못 잡음. HPLT/FW2처럼 같은 CommonCrawl에서 온 문서(LESSONS 7, docs/LESSONS.md:17)는 boilerplate/추출 차이로 바이트가 달라 exact로 안 잡힌다.
- 설계 의도는 정직함: 코드는 공백/대소문자가 의미를 가지므로 보존(`dedup.py:22`, `tests/test_dedup.py:184-188`).
- 긴장 지점: dedup.py는 소스 유형 구분 없이 NFC를 적용한다(`dedup.py:23`). split 계층은 코드에 `raw-utf8-code-v1`(raw SHA)을 쓰고 LESSONS 10(docs/LESSONS.md:22)은 코드 NFC 변환이 literal을 바꾼다고 경고한다. dedup.py에 코드 소스를 넣으면 NFC/NFD literal만 다른 두 코드 파일 중 하나가 제거된다. 실제 파이프라인에서 코드 소스가 dedup.py를 통과하는지는 **불확실**(DATA_CONTRACT 예시는 웹만 보임).

## 2. near-duplicate 정책: 구현 범위 vs 측정만

| 계층 | 상태 | 근거 |
|---|---|---|
| exact NFC 제거(삭제 수행) | **구현** | dedup.py:83 |
| split 그룹: raw 동일본문(소스 간), 웹 NFC 동등, 코드 repo + 동일파일로 연결 | **구현(분리만, 삭제 아님)** | split_manifest.py:283-291, docstring 150-154 |
| 웹 near-dup(5-gram Jaccard, 하위 16 해시 중 2개 공유 후보) | **측정만(probe), 삭제/그룹 연결 없음** | probe_text_similarity.py:1-4, 164-175; 보고서 scope 문구 189 |
| URL 겹침 | **측정만** (텍스트 중복 아님) | probe_url_overlap.py:3, 88 |
| 코드<->웹 whole/fenced block | **측정만**, 샘플/fingerprint 한정 | probe_code_web_overlap.py:145 |
| near-dup을 split 그룹에 반영 | **미구현**: split은 "No near-dup claim"(split_manifest.py:153, 319) | |

즉 near-dup 판정은 어디에서도 train/eval 분리에 반영되지 않는다. near-dup 쌍이 서로 다른 그룹이면 train/eval에 갈라질 수 있다(누수). 합성 실행에서도 `Hello World`/`hello world`/`Hello  World`는 서로 다른 그룹(웹 5행 -> 4그룹; NFC/NFD 쌍만 병합).

### probe_similarity_recall.py의 재현율 정의

- 정의: truth = 비해시(unhashed) NFC+공백정규화 5-char 집합의 **완전 교차-소스 쌍 Jaccard >= t** (t = .8/.9/.95), candidate = 운영 후보 생성기(`probe_text_similarity.run`)가 `_candidate_observer`로 보고한 쌍. `candidate_recall = |truth ∩ candidates| / |truth|`, `scored_recall`은 검증 Jaccard >= t까지 통과한 쌍 (probe_similarity_recall.py:40-46, 74-98).
- 분모 분리: 쌍을 `real_only / synthetic_only / mixed` 3군으로 라벨링(86행)하고 `all` 포함 4군별로 따로 보고(95-97). 양성 0이면 `None`(null) - 100%로 보고하지 않음(2-3, 43-45행; LESSONS 15, docs/LESSONS.md:30과 일치). 합성 control은 시드 8 base, base당 5개 텍스트(원본 1 + 4개 교차-소스 변형: NFC, 공백+접미, 부분 치환, 대문자; probe_similarity_recall.py:26-32, 54). 대문자 변형은 대부분의 5-gram을 바꾸므로 J>=0.8에서는 사실상 음성 대조일 가능성이 있음(미측정, 추정). 모두 `study_origin='synthetic'`으로 구분(25-37). scope에 "full-corpus recall/precision guarantee 아님"을 명시(103).
- 인덱스 정렬 확인: 후보 쌍 키는 운영측 `web` 목록 인덱스, truth 키는 `selected` 인덱스다. `load_samples`는 samples.jsonl을 파일 순서대로 읽고 중복 (file,row)만 거부하며 정렬/필터 없음(data/sample_snapshot.py:24-31), `selected`는 전부 kind=web(62행, controls 34행)이므로 두 인덱스는 일치한다. 키 불일치 결함 없음.
- 한계(코드에서 확인):
  1. truth와 후보 모두 `text[:max_view_chars]`(<=2048) 앞부분만 사용 -> 앞부분이 다르고 뒷부분이 같은 문서는 truth에도 후보에도 없음. 참 near-dup 재현율이 아니라 "bounded view 내 후보생성기 vs 완전비교" 재현율.
  2. 운영 후보는 8바이트 해시 shingle, reference는 비해시 -> 해시 충돌 영향은 이 비교에 섞임(분리 불가).
  3. Jaccard 임계 .8 이상만 측정. bottom-16 중 2개 공유 조건은 J가 높으면 거의 확실히 통과하지만(기대 공유 약 16*J), 낮은 J의 recall은 측정하지 않음(thresholds 하한 .8, 53행).
  4. 실제 표본에서 양성이 없으면(예상됨) 의미 있는 값은 합성 군뿐이고 이는 회귀 증거일 뿐. 실제 분모 규모는 코드만으로는 **불확실**(저장 샘플이 필요, 실행하지 않음).
  5. 후보 postings(`index[h]`) 상한 없음(probe_text_similarity.py:166) -> 흔한 해시에서 시간 상한(`max_seconds`)에만 의존.

## 3. 소스 순서 생존 정책 / 재개 / 입력 변경 감지 (직접 실행 확인)

- 생존: 소스 순서 -> 파일 순서 -> 파일 내 행 순서의 첫 등장 생존(dedup.py:83). 실행: s1 `Hello world` 생존, s3의 동일 문서 제거. 순서를 바꾸면 manifest 불일치로 거부(test_dedup.py:161-162). 단점: 품질 기준 없이 순서만으로 선택(예: 더 짧은/잘린 사본이 먼저면 그것이 남음).
- 원자성: 샤드 단위로 `BEGIN IMMEDIATE`(73) -> seen INSERT + processed INSERT를 같은 트랜잭션에서 commit(91-92), 실패 시 rollback(93-95). 실험: 샤드2의 parquet rename 직전 `os._exit(77)` -> DB는 seen=4, processed=[s1] 유지(샤드2의 seen 삽입은 롤백됨), 재개 후 결과가 무중단 실행과 **동일**(3개 샤드 전부 `resume==clean True`). rename 후 commit 전 크래시는 orphan 보존 후 재실행(67-71, test_dedup.py:164-171). commit 후 `.done` 마커 누락은 복구(62-64, test:157-160).
- 입력 변경: 시작 시 `size+mtime_ns+resolved path+순서+policy`를 manifest와 비교(36-42). 실험: mtime 변경 -> `Inputs/order/policy changed`로 거부. 
  - 갭: **크기·mtime 동일 + 내용 변경은 감지 못함**(실험: 같은 크기로 쓰고 mtime 복원 -> 조용히 SKIP 후 성공). 내용 해시 없음, 파일 처리 도중 변경에도 재확인 없음. split 계층은 반대로 내용 SHA256 전체 검증(split_manifest.py:202-214)이라 일관성이 떨어짐.
  - `synchronous=FULL`이지만 parquet tmp 파일은 rename 전 fsync 안 함(dedup.py:90) -> 전원 장애 시 DB는 commit, 출력은 불완전할 수 있음. 재개 시에는 행 수만 검사(62) - 손상 내용은 못 잡음. 일부는 불확실(OS/FS 의존).
  - `h` 컬럼 누락 입력은 의미 불명 `IndexError`(실험), 검증 메시지 없음. 로직 결함은 아님.
  - 규모: 문서당 SQLite 행 `INSERT OR IGNORE` -> 대규모 성능은 **측정 불가**(코드에 측정 없음; LESSONS 19가 작은 규모 외삽 금지를 스스로 경고).

## 4. 분리 단위와 누수 논리

방식: Union-Find를 SQLite `nodes`에 영속(split_manifest.py:241-251), 본문 키 `text:<sha>` + 코드는 `repo:<name>` 합집합(287), 웹은 NFC 키 합집합(289-291). 컴포넌트 루트당 seed 해시로 한 번 추첨(304-305). 사후 검증 `validate_db`가 content_sha256/group_id/repo 교차 split 0건을 강제(174-183), `SplitSelection`이 DB SHA·per-file 행 커버리지·카운트를 재검증(331-356). 변조 시 거부 테스트 존재(test_split_manifest.py:87-99).

허점(실행으로 확인한 것 포함):

1. **giant component 위험, 방지 장치 없음**: 실험(코드 200 repo, 30%가 동일 `MIT License\n` 파일 보유) -> 60 repo/240문서(코드 문서 660개의 약 36%, 전체 965문서의 약 25%)가 `repo:org/repo0` 한 컴포넌트로 합쳐짐(4개 seed 모두). 한 번의 추첨으로 통째로 train 또는 eval. eval_fraction=.1일 때 eval 총량은 seed별 55~84/965(5.7~8.7%)로 변동. 이 컴포넌트가 eval로 가면(확률 약 10%) 코드 eval의 큰 부분이 단일 클러스터가 되고, 빈 파일·LICENSE·`__init__.py` 류가 많은 실제 GitHub 코드에서는 더 극단적일 수 있음(실제 비율은 **불확실**, 미측정). 완화 장치: 컴포넌트 크기 상한/boilerplate 제외/추첨 후 비율 게이트 코드 없음. `split_distribution*.py`가 largest_components와 `eval_document_fraction`을 보고만 하고 실패 게이트는 아님(split_distribution.py:347-348, verified_split_context.py:206-210). 테스트에도 giant-component 케이스 없음(grep 확인).
2. **repo 합병 의도는 타당**하나(같은 repo는 한 쪽), 반대로 repo 키 때문에 "무관한 repo가 boilerplate로 연결"되는 과대 병합이 구조적 부작용.
3. **웹은 NFC 동등만**: 대소문자/공백/near-dup 변형은 별도 그룹 -> 누수 가능(§2). 문서에도 명시(split_manifest.py:319).
4. **해시 충돌/후보 누락**: 그룹 키는 전체 SHA256 hex라 충돌은 사실상 무시 가능. 누락은 "후보 생성" 문제가 아니라 규칙이 exact/NFC/repo뿐이라 생기는 구조적 누락. 코드 부분 조각·fenced block 교차는 split 그룹에 미반영(probe만).
5. 코드 repo는 `strip().lower()` 정규화(286-287) -> `org/Repo`와 `org/repo` 병합(타당). 소스 간 repo 이름 충돌 가능성(다른 플랫폼 같은 이름)은 **불확실**.
6. `text_hash`는 `encode('utf8')`(171): 단독 surrogate 포함 문자열은 예외로 ingest 중단(실패 안전하나 미테스트; 실제 발생은 불확실).
7. 두 개념 분리: dedup.py(삭제)와 split(그룹 격리)가 서로 독립 -> exact 중복은 dedup 후 split에서 다시 처리되며 이중 방어. 좋은 설계지만 dedup 출력이 split 입력이 되는 경로는 DATA_CONTRACT에 따름(확인 범위 밖).

## 5. 재개/원자성(split)

LESSONS 12(docs/LESSONS.md:25)와 코드 일치: ingest/assign 진행 지점을 데이터와 같은 트랜잭션(`put('row')` 후 `commit`, split_manifest.py:293), finalize는 단일 UPDATE+phase 트랜잭션(309-313). `tests/test_split_recovery.py:16-39`는 7개 크래시 지점(`os._exit`)마다 재개 후 결과가 무중단 baseline과 동일함을 검증. 잔여 비용: 재개마다 전체 입력 SHA 재검증, 큰 finalize 트랜잭션(DATA_CONTRACT:60 자체 인정). LESSONS 19가 지적한 대로 100만 행 측정 외삽 금지 - 실제 1.29억 건 성능은 **측정 불가**.

## 6. 기타 관찰

- `data/split_manifest_candidate.py`는 `split_manifest.py`와 **바이트 동일**(cmp 확인) - 중복 파일. benchmark/테스트(`scripts/benchmark_split_hotpath.py:12`, `tests/test_split_hotpath_candidate.py:7`)만 참조 -> 유지보수 부담/혼동.
- `release_split.py`: 읽기전용 immutable 연결, 배치별 논리 digest를 인코딩해 Parquet 재독 후 재계산 비교(70-75), RSS 예산, fsync, 분리 마커. 단 `assert` 사용(python -O에서 무효화), `contract.json` 비교에 `database_stat` inode/mtime 포함(재배포·복사에 취약), `fresh_global_leakage_check: False`를 정직하게 기록(137).
- `verified_split_context.py`/`local_split_reader.py`: 호출 단위 검증 컨텍스트, "영속 신뢰 캐시 없음", 로컬 복사본은 SHA 일치 접두 재개(`local_split_reader.py:249-257`) - 견고.
- 스타일: 세미콜론 연결 한 줄 코드(split_*, probe_*)로 가독성·재사용성 낮음. dedup.py만 일반 스타일.
- 테스트: test_dedup.py 4개 테스트가 재개/orphan/롤백/정규화 의미를 커버. 입력 내용 변경(동일 size/mtime)·h 누락·다중 샤드 크래시는 미커버(본 평가에서 실험으로 확인).

## 점수 근거 요약

- A=4(알고리즘 깊이 감점 반영): exact+Union-Find+bottom-k 후보+완전 reference 비교는 적정 수준이나 MinHash/LSH 기반 실제 near-dup 제거가 없어 '알고리즘 깊이'는 3 수준, 견고성/재현성은 4~5 수준으로 평균 4. 트랜잭션 원자성, fault-injection(7 지점 + orphan), 전체 SHA 기반 증명, 교차 검증(통합 DB 불변식)이 file:line으로 확인됨. 미달: dedup.py 내용 변경 미감지, giant component 게이트 없음, near-dup 미반영, 규모 성능 측정 불가.
- B=4: "0건의 분모/합성 vs 실제 분모 분리/후보 재현율 null" 같은 측정 윤리와 크래시 복구 증명은 차별적. 대규모 실측이 없어 5는 아님.
- C=2: 프로젝트 contract 계층(`common.data_contracts`, `source_inventory`)에 강결합, 한 줄 스타일, 중복 파일, 경로 하드코딩(`release_split.py:148`). dedup.py 자체는 독립적이라 일부 추출 가능.
