> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 코드 자료 별도 확보 계획

현재 실행 중인 5개 소스의 정확중복 제거는 그대로 유지한다. 이 문서는 코드 후보 확보 계획이며 본학습용 코드 구성 확정 보고가 아니다.

## 빠진 자료 구분

- 사전학습 범주: 한국어 웹·위키, 영어, 수학은 확보·정제 완료. 코드 소스만 미확정.
- 후속 단계: 최종 한국어 벤치마크·독립 holdout 검증, SFT instruction, 필요 시 preference 자료는 별도 준비 대상. 코드 확보만으로 전체 파이프라인 준비 완료가 되지 않음.
- 과거 모델의 원 checkpoint/optimizer 복원은 비교 검증용 별도 작업이며 새 모델 학습의 선행 조건이 아님.

## 현재 조사·시험 확보

- 선택 후보: codeparrot/github-code, revision b5661e6b17396364b2bcf8e68977b0d28e1ebd19. Hub API상 gated=false, private=false. 기존 HF 인증으로 파일 읽기 성공.
- 시험 경로: data/raw/github_code_pilot/. 1 parquet 285,840,384 bytes, 102,209파일. SHA256·parquet metadata 검증 통과.
- 실제 스키마: repo_name, path, size, content, license. 카드 예제의 code/language 필드를 그대로 가정하지 않는다. 언어는 확장자로 추정하며 repo/path/license 정보를 보존한다.
- scripts/probe_code_pilot.py 조사 결과: 원문 UTF-8 829,316,551 bytes. 초기 후보 정책 MIT/Apache-2.0/BSD-2/3/ISC/CC0/Unlicense 메타데이터와 길이 20~200,000자를 적용하면 67,807파일, 404,518,545 bytes. 아직 실제 정제·중복제거 산출물을 만든 것은 아님.
- 한 개 초기 샤드의 결과는 전체 분포를 대표하지 않음. 단순 라이선스 라벨 필터를 개별 파일의 권리 검증 완료로 표시하지 않음.
- 코드 파일뿐 아니라 HTML/Markdown/기타 확장자가 있으므로 최종 언어 구성은 Python/JS/TS/C/C++/Java/Go/Rust/SQL/Shell 등을 구분해 재집계한다.
- StarCoderData는 gated=auto이며 접근 약관 확인이 별도로 필요하므로 현재 기본 후보에서 보류. SmolLM python-edu는 코드 본문이 아니라 Software Heritage blob 식별자 중심이며 S3 본문 확보 단계가 추가되어 즉시 사용할 대체 본문으로 간주하지 않음.

## 병행 순서

1. 원본 본수집 전에 언어·라이선스·출처·생성 코드·vendor/minified 파일·내용 길이·오염/비밀키 처리 정책을 작은 표본으로 정한다.
2. pilot과 별도 manifest/경로에 제한된 크기로 확장 확보한다. 기존 pilot의 max_files=1 manifest나 기존 5개 소스 manifest를 변경하지 않는다. 전체 수백 GB를 처음부터 받지 않는다.
3. 코드 전용 정제에서 대소문자·들여쓰기·개행을 보존한다. 웹용 norm_ws/exact_hash와 PII 문자열 치환을 코드에 무조건 적용하지 않는다. REF-DATA D03 입력을 실제 코드 정제/중복 함수에 연결한다.
4. 코드 내부 정확/유사중복·generated/vendor·언어 비중을 점검한다. repo/path/source revision/license를 provenance manifest에 보존한다.
5. 이미 완성된 정제본(예: 코드↔OWM)의 읽기 전용 교차검사는 exact_v1과 병행할 수 있다. 기존 exact_v1 결과와의 통합/삭제 판정은 완료 후 별도 처리 버전에서 수행한다. 실행 중 exact_v1 SQLite/manifest/입력 순서를 수정하거나 같은 output으로 코드를 추가하지 않는다. 교차 검사 결과는 별도 버전·manifest로 기록한다.
6. 검증한 기존 데이터+코드 경로를 최종 tokenizer 표본에 연결한 뒤 tokenizer를 학습한다. 코드 10% 목표는 100~120B 학습 중 10~12B 노출 토큰을 뜻하며, 필요한 고유량·반복 횟수는 실제 토큰화 결과로 산정한다.

기존 데이터 준비 흐름과 별도의 파일 확보이며 모델 서브에이전트 위임이 아니다. 현재 권장 개발 모델/effort는 Sol / High.

## 근거

- https://huggingface.co/datasets/codeparrot/github-code/blob/b5661e6b17396364b2bcf8e68977b0d28e1ebd19/README.md
- https://huggingface.co/datasets/codeparrot/github-code-clean/blob/main/README.md
- https://huggingface.co/datasets/bigcode/starcoderdata/blob/main/README.md
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/main/README.md
- 로컬 조사: logs/github_code_pilot_audit.json, logs/download_github_code_pilot.log.

## 64샤드 별도 수집 실행

- github_code_v1: 같은 고정 revision에서 전체 1126개 중 양 끝을 포함한 균등 간격 64개 샤드 선택, 원본 합계 18.44GB. 이는 샤드 분산 표본이며 언어/저장소 비율을 보장하는 층화 표본은 아님.
- 실행: `python data/download.py github_code_v1 64 --spread --revision b5661e6b17396364b2bcf8e68977b0d28e1ebd19 --workers 1`. manifest는 pilot과 분리, 재실행 시 선택 변경 거부.
- `data/clean_code.py`: 원문 그대로 보존, 주요 언어 및 명시적 license metadata 선택, vendor/build/generated/minified/고정 평가 저장소/일부 credential 패턴 제외. 정책·source manifest·입력 size/mtime 기록, repo/path/license/language/source row/content SHA256 보존. metadata label을 개별 코드 권리 검증 완료로 해석하지 않음.
- 샤드 내부 dedup은 raw UTF-8 SHA256 사용. Unicode 정규화로 문자열 리터럴 의미를 바꾸지 않으며, 기존 웹용 NFC dedup 정책과 구분한다.
- 시험 정제: 102,209개 중 33,963개(210,745,816 UTF-8 bytes). 보존 파일 전부 원본 text/repo/path/license/content SHA256 대조 성공. `logs/code_pilot_content_validation.json`.
- 시험 데이터는 검증용이며 최종 혼합에 따로 더하지 않는다. 실제 후보는 github_code_v1 출력으로 한정.
- 다운로드 완료 후 `scripts/preprocess_followup.py code`가 code cleaner 2 workers 실행. 아직 샤드 간/기존 5소스와의 코드 교차중복 제거, 유사중복·언어 비율·실제 토큰량 검증은 미완료.
- 고정 평가 저장소 제외는 오염 검사의 출발점이다. 최종 평가 문항과의 내용 대조, 실제 secret/PII 탐지 범위 검증 및 출처별 분할은 후속 단계에서 확인한다.
- 5개 소스의 실행 중 exact_v1 index/manifest/프로세스에는 코드 후보를 추가하지 않는다. 기존 결과와 연결은 완료 후 별도 처리 버전과 검증을 거친다.

## 코드↔수학 병행 교차 검사

`data/probe_code_math_overlap.py`는 완성된 코드 64개·OWM 114개 정제본만 읽고 기존 exact_v1 DB/출력에는 접근하지 않는다.

- 코드 content SHA256 재검증 및 코드 내부 정확중복 수 집계.
- 수학 문서 전체 raw UTF-8 SHA256이 코드 파일과 같은 경우 집계.
- 수학 문서의 fenced 코드 블록(백틱/물결표, 80자 이상)이 코드 파일과 같은 경우 별도 집계. 바깥 CR/LF만 제외하고 들여쓰기·대소문자·Unicode는 유지.
- 블록 교집합은 진단 후보이며 해당 수학 문서 전체 삭제 근거로 사용하지 않음.
- 일반 inline 코드, fence가 없는 코드, 부분 변형·유사중복 탐지는 이번 범위 밖. 중복률 0이라도 모든 코드 중복이 없다고 판단하지 않음.
- 샤드별 보고서와 인덱스를 별도로 캐시해 재개. 입력 snapshot/정책 변경 시 새 output 경로 사용.
- 로그 `logs/probe_code_math_overlap.log`, 최종 결과 `data/samples/code_math_overlap_v1/summary.json`, 완료 `.done`.

### 코드·수학 검사 결과

코드 2,148,459파일의 SHA256 재검증 및 수학 5,449,160문서 스캔 완료.
문서 전체 exact match 0건, 80자 이상 fenced block 40,970개의 코드 파일 일치 0건.
코드 내부 raw UTF8 exact duplicate도 0건. 검증된 범위의 결과이며 inline/변형 유사중복·부분 코드 조각 비교나 다른 소스 교차검사를 대신하지 않는다.
결과 `data/samples/code_math_overlap_v1/summary.json` 및 `.done`.
