# 정제(cleaning) 계층 평가 — /home/user/korean-dense-llm

평가 방식: 읽기 전용. 저장소 추적 파일은 수정하지 않았다(`git status --short` 출력 없음). `__pycache__`는 gitignore 대상이며 생성 시점이 이 세션과 겹친다. 내가 import하지 않은 `build_mix` 바이트코드도 있어 생성 주체는 불확실하다.
실행 스크립트는 모두 `evidence/scripts/exp1.py ~ exp8.py`이며 `python -I`로 실행했다.
- 합성 입력은 시드 고정 난수 한글 음절 단어로 만들었다. 실제 웹 코퍼스 분포는 측정하지 않았으므로, 아래 오탐/미탐은 "이런 입력에서 이렇게 동작한다"는 존재 증명이다. 실제 발생 빈도는 **측정 불가**다.
- 실행해 통과를 확인한 테스트: `tests.test_clean_streaming`, `test_code_clean`, `test_math_filter` (6개 OK), `tests.test_data_readiness` (4개 OK).
- exp1에는 `KO*N` 반복 입력이 `ngram_rep`을 건드려 결과가 오염된 사례가 있다. 이 보고서는 exp2~exp8의 결과만 인용한다.

## 1. 점수표

| 축 | 항목 | 점수 | 핵심 근거 |
|---|---|---|---|
| A 기술품질 | 정확성(오탐/미탐) | **2** | `filters.py:47-54`의 임계값 필터는 동작한다. 그러나 NFD 한글 전량 거부, 코드 섞인 글 거부, 템플릿 스팸 통과, `+82-10-…` 미마스킹, 코드 secret 3패턴 등 중대 갭이 있다(§2~§5). `check_ko`/`mask_pii`/`norm_ws`를 직접 검증하는 테스트가 없어 4점 근거가 없다(§8). |
| | 알고리즘 깊이 | **2** | 문자 비율, 줄 중복, 5-gram 반복, 정규식 PII가 전부다(`filters.py:23-55`). 언어식별(LID), perplexity/분류기, near-dup(MinHash)은 정제 계층에 없다. 이는 `LESSONS.md:17`이 인지한 한계다. 코드 쪽은 정책 해시와 provenance 설계가 한 단계 낫다(`clean_code.py:29-35,82-91`). |
| | 견고성/재현성 | **3** (clean.py 3, clean_code 4, quality_report 4) | `clean.py`는 `.tmp`→`os.replace`→`.done` 원자적 커밋(`:53-57`)과 덮어쓰기 거부(`:26-27`)를 갖췄다. 그러나 `.done`에 입력/정책 서명이 없어, 입력이나 lang이 바뀌어도 이전 stats를 조용히 재사용한다(§6). `clean_code.py`는 입력 size/mtime과 `POLICY_HASH` 서명(`:82-87`)과 행 수 검증(`:89`)이 있다. 다만 해시가 일부 규칙을 덮지 못한다(§7). |
| B 포트폴리오 | 문서 | **3** | `README.md:163`, `DATA_SOURCES_AND_LICENSES.md:137`, `LESSONS.md`는 한계를 정직하게 공개한다("완전 제거 보장 안 함"). 반면 임계값과 필터별 근거를 설명하는 문서는 없다. `LESSONS.md:11`은 clean.py 구조에 대해 낡았다(§9). |
| | 차별성 | **3** | 한국어 특화 로직은 `[가-힣]` 비율과 한국 PII 정규식 정도다. 차별점은 오히려 재개 가능한 원자적 커밋, 코드 byte-preserving 정책과 NFC 충돌 사례 문서화(`LESSONS.md:22`), 수학 완화 필터(`:76-87`)와 receipt/policy 계약(`quality_report.py`)이다. |
| C 재사용성 | 결합도 | **3** | `filters.py`는 독립적이다(5점 수준). `clean.py`는 `common.paths`(`:12`), 모듈 전역 `ROOT`, `sys.path` 조작, `from filters import *`에 묶여 있다. `quality_report.py`는 `common.data_contracts`, `common.source_inventory`, 인벤토리 스키마와 강결합이다(2점 수준). |
| | 의존성 | **4** | `filters.py`는 `re` + `xxhash`뿐이다(`:2-3`). 파이프라인은 `pyarrow`가 추가로 필요하다. |
| | 단독 추출 비용 | **4** | `filters.py` 한 파일만 복사해 `-I`로 import 성공을 확인했다(`extract/` 디렉터리). `clean.py`는 `common/paths.py`(4줄) + `filters.py`면 충분하다. `clean_code.py`는 여기에 `data.dedup.atomic_json`(`:16`) 때문에 `dedup.py` 전체와 sqlite를 끌고 온다. `quality_report.py`는 `common/data_contracts.py`(80줄) + `common/source_inventory.py`(73줄)가 필요하다. |

대상 파일 8개를 모두 읽었고 실행 검증을 마쳤다. 점수 중 측정 불가인 것은 없다. 다만 실제 코퍼스 대비 정밀도/재현율은 **측정 불가**다(레이블된 데이터와 raw shard가 환경에 없음).

## 2. check_ko 경계 사례 (exp2.py)

| 입력(합성) | 결과 | 평가 |
|---|---|---|
| 한글 299자/비한글 701자 (h/ns=0.299) | `low_hangul_ratio` | 경계 정상. 임계 `<0.30`(`filters.py:48`) |
| 한글 300자/비한글 700자 (h/ns=0.300) | 통과 | 경계 정상 |
| 한글 49자 + 숫자 패딩 → | `few_hangul` | `:47` 임계 `<50` 정상 |
| 한글 50자 + 영어 → | `low_hangul_ratio` | 정상(두 필터 중 비율이 우선 발현) |
| 199자 / 200자 | `short` / 통과 | 경계 정상(`:42`). 길이에 공백/개행 포함 |
| 100000자 / 100001자 | 통과 / `long` | `:43` 정상 |
| 줄 10개 중 4줄 동일 (정확히 30%) | `dup_lines` | **부동소수 오차**: `1-7/10 = 0.30000000000000004 > 0.30`이라 "30% 초과"라는 의도와 달리 정확히 30%도 거부(`:50`, `:23-26`). 영향은 경미 |
| **산문 120어 + 코드 25줄** (h/ns=0.288) | `low_hangul_ratio` | **오탐(FP) 후보**: 한국어 기술 블로그/튜토리얼처럼 코드가 섞인 글이 거부된다. 산문 60어 + 코드 25줄(0.176)도 거부 |
| 영어 300어 + 한글 요약 40어 (h/ns=0.067) | `low_hangul_ratio` | 정책상 의도된 거부로 보임 |
| **NFD 정규화된 정상 한국어 문서** (NFC에서는 통과) | `few_hangul` (h=0) | **오탐(FP)**: `HANGUL = [가-힣]`(`:8`)이 조합형 음절만 센다. `clean.py`는 필터 전에 NFC 정규화를 하지 않는다(`:39-40`). macOS 유래 등 NFD 소스는 전량 탈락한다. 코퍼스 내 NFD 비율은 **측정 불가** |
| 한글 20줄 고유 시 | 통과 | 정상 |
| 가사: 후렴 4줄×5 반복 | `dup_lines` | 의도된 거부일 수 있으나 정당한 가사/시도 같이 거부됨(반복 구조를 구분 못 함) |
| 한글 마크다운 표 40행(고유) | `low_alnum` (0.455) | 표/구조화 문서 거부(`:52-53`). 의도 여부 불확실 |
| 한글 + 숫자 표 | `low_hangul_ratio` | 위와 같음 |
| **템플릿 목록**("- 항목 {i}: 한국어 설명 {고유어}입니다" ×40, 번호만 고유) | **통과** | **미탐(FN)**: 줄이 번호로 달라 `dup_lines`가 안 걸리고, 5-gram에 번호가 섞여 `ngram_rep`도 안 걸린다 |
| 템플릿 스팸(번호+동일 문구+난수) 30줄 | `ngram_rep` | 정상 탐지 |
| **공백 없는 반복 문자열** `"가나다라마바사아자차카타파하"*30` | **통과** | **미탐(FN)**: `ngram_rep`은 공백 분리 단어가 20개 미만이면 0을 반환한다(`filters.py:30-31`). 단어 19개 반복도 통과하고 20개부터 `ngram_rep`이다 |
| 한자어 비중 20/50/70% 합성 문서 | 모두 통과 (h/ns 0.827 / 0.625 / 0.373) | 한자는 비한글로 계산되지만 이 비중에서는 통과. 더 높은 비중의 거부 지점은 미측정 |
| U+FFFD 1~4개(~460자 문서) | `bad_encoding` | 정상(`:54`). 비율 임계 0.001 |
| `check_en`에 한국어 산문 / 일본어 / 악센트 라틴 무작위 | 모두 통과(None) | `check_en`에는 언어 필터가 없다(`:58-66`). `clean.py` 영어 경로는 `lang` 인자를 믿는다. 코퍼스 영향은 **측정 불가**(LESSONS #6은 FW2 kor 라벨에 영어가 섞인다고 기록) |

비고: 호환 자모(`ㅋㅋ`)에 대한 별도 검증은 하지 않았다(exp2의 해당 케이스가 입력 구성 오류로 무효였음). `[가-힣]`이 호환 자모를 세지 않는다는 점은 코드상 자명하다(`:8`).

처리량: `check_ko+norm_ws` 약 1,750 docs/s, 4.9 Mchar/s, `mask_pii` 10.7 Mchar/s, `exact_hash` 45 Mchar/s(평균 2.8k자 합성 문서, 단일 프로세스). `LESSONS.md:15`의 "순수 Python 필터가 병목"과 방향이 일치한다.

## 3. mask_pii (exp3.py, exp4.py)

### 주민번호 (`filters.py:7` `RRN = (?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)`)

전제 정정: 2000년 이후 출생 내국인은 7번째 자리가 3/4이므로 `[1-4]`에 **포함**된다(`050101-4234567 → <id>` 확인). 놓치는 것은 아래다.

| 입력 | 결과 |
|---|---|
| `900101-5234567` (외국인 1900년대 남) | 미마스킹 |
| `900101-6234567` | 미마스킹 |
| `050101-7234567` (외국인 2000년대) / `050101-8234567` | 미마스킹 |
| `900101-9234567` / `-0234567` (1800년대) | 미마스킹 |
| `9001015234567`(하이픈 없음, 5) | 미마스킹 |
| 전각 하이픈 `900101－1234567`, 점 `900101.1234567` | 미마스킹 |
| `900101-1234567`, `9001011234567`, `900101 1234567` | 마스킹 |

외국인등록번호(5~8)를 놓치는 것이 실제 누출이다. 새 체계(2020년 이후 부여분)의 뒷자리 세부 규칙은 **불확실**이라 단정하지 않는다.

오탐: 날짜 유효성과 체크섬을 보지 않는다. `991301-1234567`, `000000-1234567`, `123456-1234567` 모두 `<id>`로 마스킹됐다. 더 중요한 것은 13자리 숫자다.
- `8801234567893`(한국 GS1 바코드 880 접두) → `<id>`. 7번째 자리가 1~4일 때만 걸린다(균등 가정 시 40%). 쇼핑 페이지에서의 실제 빈도는 **측정 불가**.
- `주문번호 2411221234567` → `주문번호 <id>`.

### 전화번호 (`:6` `(?<!\d)(?:\+?82[- ]?)?0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}(?!\d)`)

| 입력 | 결과 |
|---|---|
| `010-1234-5678`, `010 1234 5678`, `01012345678`, `02-123-4567`, `031-123-4567`, `070-…`, `080-…` | 마스킹 |
| **`+82-10-1234-5678`, `+82 10 1234 5678`, `821012345678`** | **미마스킹** (정규식이 `82` 뒤에 `0`을 요구하므로 표준 국제표기 미처리. `+82-010-…`만 마스킹) |
| `(02) 123-4567`, `010.1234.5678`, `010/1234/5678`, `1588-1234`, `0507-1234-5678`, 전각 숫자 | 미마스킹 |
| `전화010-1234-5678` | 마스킹(앞에 한글 붙어도 처리) |

오탐:
- `0123456789`(0으로 시작하는 10자리) → `<phone>`
- `SN 01234567890` → `SN <phone>`
- `문서번호 02-2024-0001` → `문서번호 <phone>`
- `05 100 2000` → `<phone>`, `매출 011 2345 6789 원` → `매출 <phone> 원`

날짜(`2024-01-15`, `20240115`, `2024.01.15`), 국민/농협/신한식 계좌 하이픈 표기, 카드 16자리, ISBN, 우편번호 조합은 오탐되지 않았다(전화 정규식에 걸리지 않음).

### 이메일 (`:5`)
- 한글 로컬/도메인(`홍길동@example.com`, `user@한국.kr`)은 미마스킹. 난독화(`abc[at]x.com`, `abc at x dot com`)도 미마스킹.
- 오탐: `git@github.com:foo/bar.git` → `<email>:foo/bar.git`.
- 정규식 시간 복잡도: `"a."*20000` 1.0s, `"a"*50000+"@"` 1.6s, `local+@+도메인 점 반복` 2.4s로 입력 길이에 대해 초선형이다. `check_ko`가 100k자로 상한을 두므로 최악도 수 초 규모다. 실제 웹에서의 영향은 **불확실**.

### 미처리(코드로 명시적으로 확인)
카드번호, 계좌번호(`국민은행 123456-04-123456`), 여권(`M12345678`), 주소(`서울특별시 강남구 테헤란로 152 …`), IPv4/IPv6, MAC, 사업자번호, 운전면허, URL 토큰, 이름 모두 변하지 않는다. README는 "selected PII patterns"와 "완전 제거 보장 안 함"이라고 명시하므로(`README.md:118,163`) 문서와 동작은 일치한다.

## 4. norm_ws / exact_hash vs dedup.py (exp8.py)

실제 파이프라인 합성으로 비교했다. within-shard는 `exact_hash(norm_ws(x))`(`clean.py:39-42`), cross-shard는 `fingerprint(mask_pii(norm_ws(x)))`(`dedup.py:21-23`, 입력이 이미 정제본이므로)다.

| 문서쌍 | within-shard(clean.py) | cross-shard(dedup.py) | 결과 |
|---|---|---|---|
| `Hello World` / `hello world` | 동일 | 다름 | within만 병합 |
| `a  b` / `a b`, 탭/공백, NBSP/공백, 줄바꿈/공백 | 동일 | 다름 | within만 병합 |
| `a\n\nb` / `a\nb` | 동일 | 다름 | within만 병합 |
| `İ` / `i̇` (Turkish) | 동일 | 다름 | within만 병합 |
| 3개 이상 개행 vs 2개, CRLF vs LF, 줄머리 들여쓰기, 줄끝 공백 | 동일 | 동일 | 일치(둘 다 `norm_ws`가 선행 정리) |
| 전화번호만 다름 / 이메일만 다름 | **다름** | **동일** | dedup만 병합 |
| `café` NFC vs NFD (Latin) | **다름** | **동일** | dedup만 병합 |
| 한국어 NFC vs NFD | 다름 | 동일 | dedup만 병합. 단 NFD 한국어는 `check_ko`에서 먼저 탈락(§2)해 실제로 dedup에 도달하지 않음 |

해석:
- 두 계층의 동일성 기준이 다르다. `exact_hash`(`filters.py:69-70`)는 `lower()` + 공백 축약 후 xxh64이고, dedup 지문은 NFC + SHA256에 대소문자와 공백을 보존한다(`dedup.py:21-23`). 문서도 이를 인정한다(`DATA_SOURCES_AND_LICENSES.md:135`, `README.md:118`).
- 샤드 내 `h`는 PII 치환 **전** 텍스트의 해시이고(`clean.py:42` vs `:47`), 저장되는 text는 치환 **후**다. 그래서 전화번호만 다른 두 문서는 같은 샤드에서 서로 다른 `h`로 둘 다 남고(exp5: 최종 text가 동일한 두 행이 `h`만 다르게 출력됨), 이후 cross-shard dedup에서 합쳐진다. `tests/test_release_provenance.py:12-15`가 "masked-equal bodies keep distinct raw origins"를 의도로 명시하므로 의도된 설계이며 결함이 아니다.
- 대소문자 접기에 대한 인지는 `scripts/audit_frankenstallm.py:187-193`에 있다. 웹 정규화가 코드에는 부적절하다는 취지로 코드 경로를 분리했다(`CODE_DATA_PLAN.md:27`).
- 해시 폭: xxh64는 64비트. 샤드당 수백만 문서 규모에서 충돌 기대값은 무시 가능하다(2.8억 문서 가정 시 약 0.002쌍, 단순 생일 근사). 하지만 `h`만으로 문서를 식별하면 안 된다는 것을 문서가 이미 경고한다.

## 5. 코드 정제 (clean_code.py, exp6.py)

### 비밀정보(`:28`)
정규식은 `-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----`, `AKIA[0-9A-Z]{16}`, `ghp_[A-Za-z0-9]{36}` 3계열뿐이다. 탐지되면 파일 전체를 버린다(`:73-74`, 마스킹 아님).

| 입력 | 결과 |
|---|---|
| `AKIA`+16자, `ghp_`+36자, `RSA/PKCS8 PRIVATE KEY` | 거부(`secret_pattern`) |
| `ASIA…`(STS), AWS secret access key 40자, `github_pat_…`, `gho_…`, Slack `xoxb-…`, Stripe `sk_live_…`, OpenAI `sk-…`, Google `AIza…`, **`ENCRYPTED PRIVATE KEY`, `PGP PRIVATE KEY BLOCK`**, `password = "…"`, DB URL 자격증명, JWT | **전부 통과(미탐)** |
| `-----BEGIN PUBLIC KEY-----` | 통과(정상) |
| 문서 예시 키 `AKIA…EXAMPLE (AWS 문서 예시 키)`, 정규식 설명 주석 속 `-----BEGIN RSA PRIVATE KEY-----` | 거부(오탐 성격. 파일 단위 삭제라 영향은 보수적) |

업계 관행(예: StarCoder 계열의 detect-secrets + 엔트로피 기반 탐지)에 비하면 범위가 좁다. 이 비교는 내 사전 지식이며 이번 세션에서 검증하지 않았다. README/`CODE_DATA_PLAN.md:51`도 "실제 secret/PII 탐지 범위 검증은 후속"이라고 인정한다.

### 생성 코드(`:27,68`)
`@generated|automatically generated|generated by|do not edit`(대소문자 무시), 앞 3000자만 검사.
- 오탐: `# please do not edit this value without review` → `generated`, `# This list is generated by the user at runtime` → `generated`.
- 미탐: `auto-generated`, `autogenerated`, `This file was generated from schema`, 3000자 이후에 나오는 헤더.
- 확장자 기반 제외 없음: `x_pb.go`, `foo_pb2.py`, `x.pb.go`, `x.d.ts`, `x.bundle.js`, `a.min.js`는 모두 통과(헤더가 없으면 길이 필터에 의존).

### 라이선스(`:20,53`)
메타데이터 소문자 완전일치로 `mit/apache-2.0/bsd-2/3-clause/isc/cc0-1.0/unlicense`만 통과. `MIT`/`Apache-2.0`은 통과(소문자화). `gpl-3.0`, `lgpl`, `mpl-2.0`, `''`, `None`, `0bsd`, `zlib`, `mit-0`, `'MIT License'` 문자열은 거부(보수적). **본문의 GPL 헤더나 "All rights reserved. Proprietary" 헤더가 있어도 메타가 mit이면 통과**한다. 본문-메타 불일치율은 **측정 불가**.

### 경로/벤치마크
- `vendor`, `build`, `dist`, `node_modules`는 경로 세그먼트 완전일치(`src/builder`, `src/distribution`은 통과 — 정상). 대문자 `Vendor`는 소문자화되어 거부. Windows 구분자도 처리.
- 벤치마크 누출은 repo 이름 또는 경로 세그먼트의 정확한 문자열 일치만 본다(`:26,59-60`): `openai/human-eval`은 거부, 그러나 `org/HumanEval`(소문자 `humaneval`≠`human-eval`), `human-eval-x`, `mbpp-solutions`, `data/humaneval/…`, `humaneval.py`, `mbpp.py`, `swe-bench/…`, `livecodebench`는 통과한다. HumanEval/0 문제 본문 자체를 담은 파일도 통과한다. 내용 기반 오염 검사는 이 단계에 없다(`DATA_CONTRACT.md:41`이 후속 검사로 남겨둠).

### 길이/인코딩/줄
`min_chars=20`, `max_chars=200000`, `max_line=2000`, `max_mean_line=200`(`:31`)은 POLICY에 포함된다. 단일 줄 2001자 파일, 평균 줄 길이 250자 파일, 2100자 문자열 리터럴이 있는 정상 파일이 모두 `minified_or_long_lines`로 거부된다(후자는 오탐 후보). `\x00`과 U+FFFD 0.2%는 `encoding`으로 거부. lone surrogate는 `select_code`는 통과하지만 `work()`의 `.encode()`(`:103`)에서 예외가 난다. 그러나 pyarrow가 lone surrogate 문자열을 만드는 단계에서 `UnicodeEncodeError`로 거부하므로(exp8) 유효한 parquet 입력으로는 도달 불가로 판단한다.

### 중복
SHA256 완전일치만(`:103`). CRLF/LF가 다른 파일은 서로 다른 문서이며, `test_code_clean.py:16-28`(D03)이 이 의미 보존 정책을 검증한다. 샤드 내 중복만 제거하고 교차는 별도 단계다(`:34`).

## 6. clean.py 재현성/견고성 실험 (exp5.py)

- `[확인]` 입력을 바꾼 뒤 `.done`이 있는 상태로 재실행하면 새 입력을 처리하지 않고 이전 stats를 그대로 반환한다(`:19-25`). `lang`을 `ko`에서 `en`으로 바꿔도 동일하다. `.done`은 빈 파일(`:57`)이고 `stats.json`에는 `counts`/`chars_kept`만 있다. 입력 크기/mtime, 필터 정책이나 임계값의 서명이 없어, 임계값을 바꾼 뒤 기존 출력이 그대로 재사용될 수 있다. `clean_code.py:82-91`과 대비된다.
- `[확인]` 이전 실패로 남은 `.tmp`는 다음 실행에서 정상 복구(`:32` 덮어쓰기).
- `[확인]` `text` 컬럼이 없는 입력은 `IndexError`(`.tmp`만 남고 `.done` 없음). 재실행도 동일 오류로 실패해 부분 커밋 상태가 생기지 않는다(fail-fast). 오류 메시지는 불친절하다.
- `[확인]` 빈 문자열/공백만 있는 행은 `short`로 집계, `None`은 `null`로 집계.
- `[확인]` 배치 4096행 경계를 넘는 중복 제거와 재개가 `tests/test_clean_streaming.py:12-29`로 검증되어 있다.
- `[불확실]` ProcessPoolExecutor에서 워커가 OOM으로 죽으면 `job.result()`가 `BrokenProcessPool`을 던지는 것이 일반적인 문서화된 동작이다. 이 평가에서는 실제로 워커를 강제 종료해 검증하지 않았다. `LESSONS.md:11`은 과거 `multiprocessing.Pool`에서의 영구 정지를 기록한다.

`probe_filters.py`(코드 읽기 기준, 일부만 실행):
- `sys.path.insert(0, "data")`가 실행 위치(cwd)에 의존한다(`:3`).
- 앞 N행만 읽는다(`:7-15`). 무작위 표본이 아니라 row-group 순서 편향이 있다.
- `check_ko`만 호출한다(`en`/`math` 불가). 결측 `text`(None)는 `norm_ws(None)`에서 예외가 날 코드 구조다(`:10`, 코드 읽기만, 미실행).
- 예시 출력에서 입력 원문 200자를 그대로 출력한다. PII 마스킹 전의 raw에서 읽으므로 로그에 PII가 남을 수 있다(코드 읽기).

`quality_report.py`는 이름과 달리 필터 품질(오탐/미탐, reject rate)을 평가하지 않는다. receipt, footer 행 수, 정책 선언의 정합성을 검증하는 준비도(readiness) 점검이다(docstring `:1-4`과 `:159` note로 스스로 명시). 중복 JSON 키 거부(`:15-22`), 증거 파일 SHA 고정(`:122`), 보고서가 자기 코드 SHA를 계약에 포함(`:156`)하는 설계는 견고하다. 반면 한 줄에 여러 문장을 `;`로 이은 압축 스타일(`:31-35`, `:47-100` 등)이 가독성을 크게 해친다. `POLICY` 문자열이 `dedup.py:129`와 `quality_report.py:11`에 각각 하드코딩되어 있어 어긋날 수 있다.

## 7. 임계값/정책 하드코딩 전수 목록

"해시 포함"은 `clean_code.py`의 `POLICY_HASH`(`:35`) 포함 여부다. `clean.py`에는 해당 메커니즘이 아예 없다.

### data/filters.py (설정 외부화 시 영향: clean.py, probe_filters.py, tests/test_math_filter.py, tests/test_release_provenance.py, scripts/audit_frankenstallm.py:187을 직접 호출)

| 위치 | 값 | 의미 |
|---|---|---|
| `:5`,`:6`,`:7` | EMAIL/PHONE/RRN 정규식 | PII 패턴 |
| `:8` | `[가-힣]` | 한글 판정(NFD, 자모 제외) |
| `:9` | `\n{3,}` | 개행 축약 |
| `:29` | `n=5` | n-gram 길이 |
| `:30-31` | `len(words) < n*4` (=20단어) | 반복 검사 면제 |
| `:39,58,76` | `min_chars=200`, `max_chars=100_000` | 세 함수에 중복 기본값(인자로만 덮어쓰기 가능하나 `clean.py`는 전달 안 함) |
| `:47` | `< 50` | 최소 한글 수 |
| `:48` | `< 0.30` | 한글 비율 |
| `:50`,`:63` | `len(lines) > 5`, `> 0.30` | 줄 중복 |
| `:51`,`:64` | `> 0.30` | n-gram 반복 |
| `:53`,`:65` | `< 0.5` | 영숫자 비율(분모: 전체 길이) |
| `:54` | `"�"`, `> 0.001` | 깨진 인코딩 (`check_en`에는 없음) |
| `:73` | `MATH_MARKER` | 수학 표지 정규식 |
| `:83,85` | `0.50→0.25`, `>= 2`개 | 수학 완화 규칙 |

참고: `check_ko`와 `check_en`은 길이/줄중복/n-gram/영숫자 검사 코드가 거의 복사본이다(`:42-53` vs `:59-65`). 임계값을 분리하면 두 곳을 동시에 바꿔야 한다. 언어별 제외 사유(깨진 인코딩 검사 차이)가 의도인지는 **불확실**하다.

### data/clean.py
`:33` batch_size=4096, `:62` 기본 workers=10, `:12-13,17-18,63` 경로(`DATA`, `clean/<src>`, `raw/<src>/**/*.parquet`), `:28` lang 디스패치(`ko/en/math`), `:30` 스키마(`text`,`h`)와 `zstd`, `:7` `NUMEXPR_NUM_THREADS=8`(정제와 무관). 정책/임계값은 서명되지 않는다.

### data/clean_code.py
| 위치 | 값 | 해시 포함? |
|---|---|---|
| `:20` LICENSES, `:21-24` LANGUAGES, `:25` EXCLUDED_PARTS, `:26` BENCH_REPOS | 목록 | 포함 |
| `:27` GENERATED, `:28` SECRET 정규식 | 패턴 | 포함(`pattern` 문자열) |
| `:31` min_chars=20, max_chars=200000, max_line=2000, max_mean_line=200 | 길이 | 포함 |
| `:30` `version='code-v1'` | 수동 버전 | 포함(수동 관리) |
| `:66` `'\x00'`, FFFD 비율 `> .001` | 인코딩 | **미포함** |
| `:68` `text[:3000]` 생성 코드 검사 범위 | 윈도우 | **미포함** |
| `:51-52` `.C → C++` 특례 | 언어 판정 | **미포함** |
| `select_code`의 판정 순서와 로직 자체 | 로직 | **미포함** |
| `:98` batch_size=1024, `:127` source choices, `:131` workers 기본 2 | 운영 | 미포함(결과 무관) |

따라서 미포함 항목을 바꿔도 `POLICY_HASH`가 변하지 않아 재개 시 변경이 감지되지 않는다(`:82-87`).

### 영향 범위 요약
- 임계값을 설정으로 빼면 `filters.py` 호출부 5곳(위 표 제목 참고), 임계값을 인용하는 문서(없음 — 문서는 값을 명시하지 않음), `scripts/audit_retention_recovery.py:92-93`와 `scripts/prepare_public_recipe.py:30`이 `filters.py`/`clean*.py` 파일 목록을 참조하므로 코드 식별자(SHA) 재산출이 필요할 가능성이 있다(이 스크립트들은 실행하지 않았음 — **불확실**).
- `experiments/code/` 아래 해시 이름의 스냅샷 사본들은 역사적 기록이므로 건드릴 필요가 없다.
- 설정 도입 시 `clean.py`에도 `clean_code.py`와 같은 정책 해시 서명을 `.done`에 넣어야 §6의 재사용 위험이 해소된다.

## 8. 테스트 커버리지 (대상 3개 테스트)

| 테스트 | 검증하는 것 | 검증하지 않는 것 |
|---|---|---|
| `test_clean_streaming.py:12-29` | 4096 경계 넘는 중복 제거, `.done` 재개, 미표시 출력 덮어쓰기 거부, `null`/`short` 집계 | `short` 외 어떤 거부 사유도 실제로 발동시키지 않음. PII, 한글 비율, 실패 경로 |
| `test_code_clean.py:16-57` | D03 의미 보존, 라이선스/vendor/벤치마크/generated/AKIA 거부, 원문 보존, 출처 컬럼, 재개, 정책 변경 거부 | 위의 secret 미탐과 generated 오탐, 라이선스 변형, 길이/인코딩 사유, 벤치마크 이름 변형 |
| `test_math_filter.py:5-15` | 수학 완화와 다른 필터 유지 | 임계값 경계 |

`check_ko`, `mask_pii`, `norm_ws`를 직접 검증하는 테스트는 대상 3개에는 없다. 리포 전체에서도 `tests/test_release_provenance.py`가 `mask_pii`와 `exact_hash`를 간접 호출하는 것 외에는 찾지 못했다(`grep` 결과). 이 때문에 정확성은 4점이 될 수 없다.

## 9. 문서 평가

- 좋은 점: 한계 공개가 명확하다("완전 제거 보장 안 함", `README.md:163`). `DATA_SOURCES_AND_LICENSES.md:135`가 `h`의 정의(PII 치환 전, 소문자/공백 축약, xxh64 − 2^63)를 정확히 기술한다. `LESSONS.md:22`가 NFC와 코드 원문 보존 문제를 분리해 기록했고, 이 기록이 `clean_code.py` 설계(byte-preserving)로 이어진 흔적이 보인다. `#6`과 `#8`이 각각 한글 비율 필터와 `check_math` 도입 근거다.
- 미흡한 점: 각 임계값(0.30, 50, 0.5 등)을 왜 그 값으로 정했는지, 어떤 실측으로 정했는지 적힌 문서가 없다. 정제 사유별 제거율 자료도 번들에 없다("Missing run artifacts are not bundled data", `LESSONS.md:1`). 그래서 필터 품질을 외부에서 재현/검증할 수 없다.
- 낡은 기술: `LESSONS.md:11`의 "현재 clean.py는 샤드 전체를 메모리에 모으는 구조"는 지금 코드(`clean.py:32-51` 배치 스트리밍 + ParquetWriter)와 다르다. 문서 머리말이 "historical snapshot"이라고 밝히므로 심각한 오류는 아니나 읽는 사람에게 혼란을 줄 수 있다.
- `clean.py`의 docstring(`:1-4`)에는 사용법이 있으나 필터 사유 코드 목록과 출력 stats 스키마 설명이 없다.

## 10. 가장 중요한 발견 5개

1. **`clean.py`는 입력/정책 서명이 없어 변경 후에도 낡은 결과를 재사용한다.** `.done`이 존재하면 새 입력이나 다른 `lang`에도 이전 stats를 반환한다(`clean.py:19-25,57`, exp5 확인). `clean_code.py`의 `POLICY_HASH`(`:35,82-87`)와 설계 수준이 다르다. 임계값 조정 후 재정제 시 사고 위험이 있다.
2. **NFD 한국어 문서는 전량 `few_hangul`로 탈락한다.** `[가-힣]`(`filters.py:8`)이 조합형만 세고 NFC 정규화가 필터 앞에 없다(`clean.py:39-40`). 정상 문서를 NFD로 바꾸면 h=0이 되어 거부됨을 확인했다. 코퍼스 내 비율은 측정 불가.
3. **PII 마스킹의 구멍과 오탐.** `+82-10-1234-5678`(국제표기)과 외국인등록번호(7번째 자리 5~8), 하이픈 변형이 미처리다. 카드/계좌/여권/주소/IP도 처리하지 않는다. 반대로 `0123456789`, `문서번호 02-2024-0001`, 7번째 자리가 1~4인 13자리 숫자(`8801234567893`, 주문번호)가 `<phone>`/`<id>`로 오마스킹된다. 체크섬/날짜 검증이 없다.
4. **코드 정제의 secret 탐지는 3패턴 계열뿐이고 정책 해시가 일부 규칙을 덮지 못한다.** ASIA/AWS secret/github_pat/Slack/Stripe/OpenAI/Google/ENCRYPTED·PGP 키/JWT/DB URL이 모두 통과한다(§5). `generated` 정규식은 일반 주석(`do not edit`)을 오탐하고 `auto-generated`는 놓친다. 인코딩 비율, 3000자 윈도우, `.C` 특례, 로직 자체는 `POLICY_HASH` 밖이다(§7).
5. **필터 정확성을 보증하는 테스트가 없고, 일부 필터 설계에 구조적 미탐/오탐이 있다.** `check_ko`/`mask_pii`/`norm_ws` 직접 테스트가 없다(§8). 공백 없는 반복 문자열과 단어 19개 이하 반복은 `ngram_rep`을 통과하고(`filters.py:30-31`), 번호만 다른 템플릿 목록도 통과한다. 한국어 기술 글(산문+코드)은 `low_hangul_ratio`로 거부된다. 줄 중복 임계는 부동소수 오차로 정확히 30%에서도 거부한다.

보조 관찰: 샤드 내(`exact_hash`: 소문자/공백 축약, PII 치환 전)와 샤드 간(`fingerprint`: NFC SHA256, 대소문자/공백 보존, 치환 후) 중복 기준이 다르다. 이는 의도된 설계로 문서화되어 있으므로(`test_release_provenance.py:12-15`, `DATA_SOURCES_AND_LICENSES.md:135`) 결함으로 분류하지 않았다.
