# korean-dense-llm 클린 클론 재현성 시험 (smoke)

## 1. 기준
- 클론: `git clone --branch claude/gracious-wozniak-lrksyz https://github.com/pathcosmos/korean-dense-llm /tmp/claude-0/clean_clone/repo` 성공 (프록시 문제 없음, 로컬 클론 대체 안 함).
- HEAD `2e6103f4951713e26fd421f3f6c1b2a4b91f2eff` == 원본 작업 트리 HEAD (`git rev-parse HEAD` 동일). 시험 후 추적 파일 변경 없음(`git status`는 ignored `__pycache__`/`.pytest_cache`만).
- 환경: Linux 6.18, 4 vCPU, 15GiB RAM, Python 3.13.18(CI는 3.12), 기존 venv `/tmp/claude-0/venv`(torch 2.7.0+cpu, numpy 2.5.4, pyarrow 19.0.1, tokenizers 0.23.3, pytest 9.2.0). 새 venv/pip 설치 미시험 -> 한계.
- 환경변수: OMP/MKL/OPENBLAS/RAYON_NUM_THREADS=2, CUDA_VISIBLE_DEVICES=(빈값), PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. 이 설정으로 깨진 테스트 없음.

## 2. 문서 검증 재현
| 명령 | rc | 시간 | 결과 |
|---|---|---|---|
| `python examples/quickstart_cpu.py` | 0 | 4s | 통과, `resume_model_optimizer_exact: true` |
| `python -m pytest -q tests -rs` | 0 | 36s | **336 passed, 36 skipped, 0 failed** (372 collected) |

- 문서 수치와 불일치: PUBLIC_VALIDATION은 284 passed/9 skipped(총 293), README/CI 기록은 258/35(총 293). 현재 HEAD는 372개 수집 -> 문서의 검증 수치는 이 HEAD 기준이 아니다(확정). 이후 커밋으로 테스트가 79개 늘었으나 문서 미갱신(추정).
- skip 36건 분해(전부 skip, 실패 아님): 두 파일시스템 필요 26건(8+5+13; /tmp 단일 FS라서), 원본 운영기록 필요 integration 9건(문서의 "9건"과 일치), `tests/test_filters_oracle.py:37` "local-only pilot texts (not committed) are absent" 1건. `dataqa/gold/seed_v1.jsonl`, `dataqa/real/pilot_*_v1.jsonl`은 커밋되어 있어 dataqa 테스트는 skip 없이 통과. `dataqa/review*`·로컬 gold에 의존하는 테스트는 실패하지 않고(별도 skip 표시 없음 - 해당 파일은 테스트가 요구하지 않음) 통과.

## 3. 데이터 처리 CLI 사슬 (PROJECT_ROOT 오버라이드)
입력: `data/download.py wiki_ko --list-only` -> 3 shards 0.78GB, revision b04c8d1c... ; `data/download.py wiki_ko 1` -> 1 shard 0.40GB, 215,966 rows, 16s, rc 0, SHA256 검증 통과. 이후 전체 shard로 단계 2~3(work), 6개 row group(6,000행)로 줄인 같은 이름 parquet으로 단계 4 이후(work3)를 실행(속도 목적; pyarrow `read_row_group`으로 잘라 같은 파일명 규칙으로 저장).

| # | 단계/명령 | rc | 시간 | 핵심 출력 | 판정 |
|---|---|---|---|---|---|
| 1 | download.py wiki_ko 1 | 0 | 16s | `OK ... rows=215966` | 통과 |
| 2a | clean.py wiki_ko ko 1 (전체 shard) | 0 | 102s | kept 174,342 / 215,966 | 통과 |
| 2b | 같은 명령 재실행 | 0 | 1s | 출력 parquet mtime 불변(재사용) | 통과 |
| 2c | `.stats.json` signature | - | - | 키 `counts, chars_kept, signature`; signature에 입력 sha256·bytes, ko_policy, pii_policy, filters_sha256, logic_version | 통과 |
| 2d | lang만 변경(`wiki_ko en`) | 1 | - | `RuntimeError: Output signature differs from current input/policy/code; use a new output tree` | 거부 확인 |
| 2e | 입력 parquet 교체(같은 이름, 1000행) + 기존 clean 복사 | 1 | - | 동일 RuntimeError | 거부 확인 |
| 3a | dedup.py --sources wiki_ko --output dedup/exact_v1 | 0 | 9s | input 174,342 kept 174,342 dup 0 | 통과 |
| 3b | dedup 재실행 | 0 | 1s | `SKIP wiki_ko/...` | 재사용 확인 |
| 4a | prepare_source_manifest.py (6개 소스 하드코딩) | - | - | 5개 소스의 raw download_manifest가 없어 그대로는 불가 | **우회**: SOURCES를 `['wiki_ko']`로 몽키패치한 호출 -> draft 생성 rc 0 |
| 4b | scripts/audit_source_inventory.py (PROJECT_ROOT=임시) | 1 | 0s | `FileNotFoundError: .../work3/data/source_audit.py` | **실패(확정)**: source_audit.py:61-62가 자기 코드 파일 해시를 `resolve()`=PROJECT_ROOT 기준으로 읽음 |
| 4b' | 우회: PROJECT_ROOT 안에 repo의 common/ data/*.py tokenizer/ 등을 symlink 후 재실행 | 0 | 1s | 5188행 검증 | 통과(우회) |
| 4c | 정책 없이 `assess()` | - | - | readiness `pending` | 게이트 동작 확인 |
| 4d | 정책 | - | - | 모든 check를 passed로 쓴 **smoke 전용 정책**(quickstart_cpu.py 방식 복제, 실제 승인 아님; 가이드가 금지한 것을 시험 목적으로만 수행) -> assess `passed` | **우회(비승인)** |
| 5 | scripts/finalize_source_manifest.py | 0 | 0s | `status: ready` | 통과(위 smoke 정책) |
| 6 | data/split_manifest.py --eval-fraction 0.02 --seed 7 | 0 | 1s | train 5083 / eval 105 | 통과 |
| 7a | tokenizer/train_tokenizer.py (--mix wiki_ko:1 --mb 10 --vocab 4096 --split-manifest ...) 심볼릭 없이 | 1 | 1s | `FileNotFoundError: .../work3/tokenizer/train_tokenizer.py` | **실패(확정)**: 4b와 같은 부류(PROJECT_ROOT 기준 코드 해시) |
| 7b | symlink 우회 후 재실행 | 0 | 6s | `SPLIT-VERIFIED CANDIDATE TOKENIZER`, 1018 docs | 통과(우회) |
| 8a | data/tokenize_shards.py wiki_ko --sources-manifest ... | 0 | 12s | 13,372,760 tokens, holdout 233,336 | 통과 |
| 8b | 재실행 | 0 | 0s | `REUSED ...bin` | 재사용 확인 |
| 8c | data/parallel_tokenize.py --experimental --workers 2 | 0 | 11s | `experimental_complete`, 13,606,096 tokens | 통과(`--experimental`) |
| 9 | data/build_mix.py --token-root data/tokens wiki_ko:1 --seq-len 256 | 0 | 2s | mix.json 생성 | 통과 |

(`data/split_inputs.py`, `data/source_audit.py`는 `__main__`이 없는 라이브러리 모듈이라 PUBLIC_GUIDE 표의 "진입점"이지만 `python data/split_inputs.py --help`는 `ModuleNotFoundError: No module named 'common'`(rc 1). `PYTHONPATH=. python -m data.split_inputs --help`는 rc 0이나 출력 없음. tokenizer 준비는 CLI 없이 `train_tokenizer.py`가 내부 호출.)

## 재현 불가 / 증거 파일 부재
- `docs/data_provenance_v1/source_registry.json`: 경로형 참조 22개 중 evidence 6개(`docs/data_provenance_v1/evidence/{fineweb2_ko,fineweb_edu,github_code_v1,hplt_ko,owm,wiki_ko}_pinned_README.md`)가 클론에 없음 (`evidence/` 디렉터리 자체 없음). 또한 `docs/data_provenance_v1/file_inventory.json`을 코드가 참조하나 클론에 없음(`ls`로 확인: 디렉터리에는 source_registry.json만 존재). 영향: 출처 증빙 재검증 불가(확정).
- `release/files.json`: 참조 548개 모두 존재(0 missing).
- `release/source_inventory.json`: 참조 403개 중 147개 부재 - 대부분 `data/clean/...`, `data/MANIFEST.json` 등 원본 운영 데이터 경로(배포 제외물; 설계상 부재).
- `runs/`는 `.gitkeep`뿐. 코드가 읽는 `runs/…` 경로(예: `scripts/run_f2_checks.py`의 `runs/source_audit_full_v1/report.json`, `scripts/measure_split.py`의 `logs/source_audit_job.json`)는 원본 실행 기록이라 부재 -> 그 운영 스크립트들은 재현 불가(PUBLIC_GUIDE 4번 항목이 명시).
- `scripts/make_manifest.py --verify`(RUNBOOK 0단계): `python scripts/make_manifest.py --verify` -> rc 1 `ModuleNotFoundError: No module named 'common'`; `PYTHONPATH=.`를 주어도 rc 1 `FileNotFoundError: .../data/MANIFEST.json`(원본 데이터 manifest 부재).
- 실제 데이터 승인(ready) 단계: 정책의 `decision`/`evidence`는 사람의 검토 산출물이라 클린 클론에서 재현할 수 없음. 위 5~9단계는 smoke 정책으로 기능만 확인한 것이며 승인된 실제 데이터 사슬의 재현이 아님.
- 시도하지 않은 단계: fineweb2_ko/hplt_ko/fineweb_edu/owm/github_code_v1 다운로드·정제(용량), `clean_code.py`, 토크나이저 대규모 학습, GPU 학습/평가, `data/probe_*.py`.

## 4. 하드코딩·이식성 점검 (읽기 전용, 데이터 처리 경로)
| 위치 | 내용 | 분류 |
|---|---|---|
| data/source_audit.py:61-62 (`code_hashes`, `resolve(p)`) | 자기 코드 파일을 PROJECT_ROOT 기준으로 해시 -> PROJECT_ROOT 오버라이드 시 FileNotFound | **실제 오류 발생(확정)** |
| tokenizer/train_tokenizer.py (코드 해시 경로, 오류 메시지로 확인; 정확한 줄은 미확인) | 동일 부류 | **실제 오류 발생(확정)** |
| common/paths.py:3, common/data_contracts.py:12-26 | PROJECT_ROOT가 데이터 루트와 코드 루트를 겸함 -> 위 두 오류의 근본 원인 | 설계상 원인(확정) |
| scripts/make_manifest.py:5 및 split_inputs/source_audit 직접 실행 | `sys.path` 보정 없음 -> `ModuleNotFoundError: common` | **실제 오류 발생(확정)**; PYTHONPATH=. 필요 |
| data/clean.py:13, data/clean_code.py:13 | `NUMEXPR_NUM_THREADS` 기본 "8" (setdefault) | 이론상(오류 없음; 4코어에서도 실행됨) |
| data/parallel_tokenize.py:171 | RAYON/OMP/NUMEXPR/TOKENIZERS_PARALLELISM을 기록만 함 | 이론상(RAYON 미설정이어도 동작) |
| data/parallel_tokenize.py:6,100; data/source_audit.py:6,68,168 | `fcntl.flock` (Windows 불가) | 이론상(Linux에서 오류 없음) |
| data/release_split.py:98; data/release_provenance.py:92,166 | `os.sched_setaffinity/getaffinity` (Linux 전용, macOS 불가), CPU 4개 미만이면 슬라이스 동작 변화 | 이론상(이번에 실행하지 않음) |
| scripts/resource_admission.py:25-27; scripts/process_accounting.py:15,37 | `/sys/fs/cgroup`, `/proc/meminfo`, `/proc/<pid>/smaps_rollup` | 이론상(데이터 사슬 CLI에서는 호출 안 함; 이번에 실행하지 않음) |
| scripts/make_smoke_data.py:3, train/configs/smoke.json:3 | `/tmp/smoke_tokens` 절대경로 | 이론상 |
| scripts/finalize_public_corpus_v2.py:95 | `assert api.whoami()['name']=='pathcosmos'` | 이론상(배포 전용 스크립트) |
| scripts/prepare_public_recipe.py:58-122, prepare_corpus_upload.py:65-154, finalize_public_corpus_v2.py:171 | 계정명 `pathcosmos`와 연락 이메일 하드코딩(문서·카드 생성용) | 이론상(개인정보성 연락처가 코드에 포함; 처리 경로 아님) |
| scripts/prepare_dataset_release.py:174 | `REBUILD_ROOT="/your/local/scratch/..."` 자리표시자 | 이론상 |
| data/prepare_source_manifest(scripts/prepare_source_manifest.py:11,19-30) | 6개 소스·`dedup/exact_v1`·`raw` 구조 하드코딩 | **단일 소스 실행 시 실제 차단(확정)** |
| 코드 내 `/home/`, `/mnt/`, `/data/`, `/nfs` 절대경로 | `common data tokenizer scripts model train eval examples`에서 `.py/.sh/.json` grep 결과 위 `/tmp` 항목 외 없음 | - |

## 한계
- 기존 venv 사용(새 venv·pip 설치 시간 미측정; requirements-cpu.txt 해소 재현성은 이번에 시험하지 않음). Python 3.13(CI는 3.12).
- 단일 파일시스템이라 two-FS retention 26건 미실행. GPU/학습/평가/분산 체크포인트 미시험.
- 4단계 이후는 6,000행 부분집합과 smoke 정책, 그리고 symlink 우회를 사용 -> 승인된 실제 사슬이 아님. 시간은 1 shard 기준이며 전체 규모 성능 아님.
- 원문 텍스트는 보고서에 인용하지 않음.

## 재현 명령 (요약)
```bash
git clone --branch claude/gracious-wozniak-lrksyz https://github.com/pathcosmos/korean-dense-llm /tmp/claude-0/clean_clone/repo && cd /tmp/claude-0/clean_clone/repo
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/tmp/claude-0/venv/bin/python
$PY examples/quickstart_cpu.py && $PY -m pytest -q tests -rs
export PROJECT_ROOT=/tmp/claude-0/clean_clone/work
$PY data/download.py wiki_ko --list-only; $PY data/download.py wiki_ko 1
$PY data/clean.py wiki_ko ko 1; $PY data/dedup.py --sources wiki_ko --output dedup/exact_v1
# 4단계 이후는 PROJECT_ROOT 안에 repo의 common/ data/*.py tokenizer/ scripts/ 를 symlink 해야 audit/tokenizer가 동작
```
