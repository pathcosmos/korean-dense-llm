> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 학습 데이터 보존·삭제 후보·재다운로드 절차

작성 범위: 2026-11-07 대화의 **원본과 중복제거 전 정제본 약668GB 모두 검토** 요청. 실제 검사 시각은 보고서의 UTC를 따른다. 검토 후 사용자의 **“추가 후보까지 포함 삭제하자”** 승인에 따라 아래 두 목록의 payload 삭제를 완료했다. 검토 증거와 실제 실행 증거를 구분해 보존한다.

## 승인된 삭제 실행 완료 (2026-11-07 대화 / 실제 UTC 2026-11-06)

- 주 대상 **1,389개/667.728GB**와 추가 후보 **5개/6.131GB**, 합계 **1,394개/673.859GB**를 파일별로 삭제했다. 대상 경로 잔존0이며 디렉터리·manifest·정제 marker·복구 증거는 보존했다.
- 삭제 직전 1,394개 전수 SHA·크기·mtime·inode와 실행 중 FD/mmap을 다시 검사했다. 접근 가능한 프로세스의 대상 열린 파일0; 접근 제한 시스템 daemon2개는 검사 한계로 기록했다. 삭제 의도/완료를 파일별 fsync journal에 남겼다.
- 최종 tokens/tokenizer·dedup 원문·코드 정제본·split/reader와 복구 자료 **5,823개 stat 불변**, 기존 ready source receipt/footer 소비자 검사 결과 전후 동일·통과. 새로운 전체 corpus 감사 완료를 뜻하지 않는다. 리허설은 삭제 직후920 → 후속940step으로 진행했다.
- 예상 할당량 **629.933GiB**, 후속 NFS 여유 변화 실측 **+629.931GiB**: **482.897 → 1,112.827GiB**. 삭제 직후 statvfs에는 반영되지 않았고 23:02:32 UTC 후속 관측에서 반영됐다. 실행 receipt의 직후 수치는 덮어쓰지 않고 별도 관측으로 남긴다. 동시 로그 쓰기 등으로 할당량과 여유 변화에는 작은 차이가 있다.
- 실행 **74.82초 / peak RSS1.574GiB**. 내용 변경·symlink·경로 이탈·파일 교체 거부 자체검사와 완료 후 재실행의 무삭제 동작을 확인했다. 원본 삭제를 과거 download 프로세스 종료 오류로 오인한 감시는 내용 SHA로 연결된 완료 receipt가 있는 정확한 경로만 역사적 완료/`payload_state=retired_after_verified_completion`로 표시하도록 수정했다. 감시 집중검사4개 통과, 30초 로컬 watcher 유지.
- [삭제 receipt](recovery/data_retention_v1/eviction_receipt.json), [공간 회수 후속 관측](recovery/data_retention_v1/space_observation.json), `eviction_execution_evidence.tar.gz`와 `EVICTION_CHECKSUMS.json`을 복구 디렉터리에 추가 보존했다. 실행 archive에는 승인 목록·파일별 journal·보호 대상 snapshot·전후 검사·실행 코드가 들어 있다. 기존 `CHECKSUMS.json`과 복구 묶음은 변경하지 않았다.

## 결론과 범위

현재 고정된 학습·후속 native 평가 경로는 최종 토큰, 최종 중복제거본, 코드 정제본, split DB와 증거를 읽는다. 아래 원본·중간 정제 payload는 직접 입력이 아니므로 해당 최종 산출물을 보존하는 조건에서 삭제 후보로 분리할 수 있다.

| 분류 | 파일 수 | 논리 크기(GB) | 실행 결과 |
|---|---:|---:|---|
| 다운로드 원본 `data/raw/*/**/*.parquet` 중 고정 목록 | 727 | 394.609 | 고정 revision·원격 크기/SHA256와 전수 일치, 승인 후 삭제 완료 |
| 웹·수학 중복제거 전 `data/clean/{wiki_ko,fineweb2_ko,hplt_ko,fineweb_edu,owm}/*.parquet` | 662 | 273.118 | 최종 입력과 분리, 대표 입력 재현 통과 후 삭제 완료 |
| 합계 | **1,389** | **667.728** | 주 대상 할당량 **624.33GiB**; 추가 후보와 함께 삭제 완료 |

**668GB 전부를 다시 다운로드하는 것이 아니다.** 다운로드 원본 약395GB를 고정 revision에서 받고, 중간 정제본 약273GB는 보존한 정제 코드로 다시 만든다. 현재 최종 토큰·원문이 보존돼 있으면 본학습 시작/재개를 위해 이 원본과 중간본을 다시 만들 필요는 없다. 정제 정책 변경이나 upstream 단계 재실행 때 필요하다.

모든 미래 작업에서 불필요하다는 뜻은 아니다. 원본 URL·필터 탈락 문서·정제 전 provenance를 다시 조사하거나 새 정제 정책을 적용하는 작업은 원본을 다시 요구할 수 있다. 이번 판단은 현재 고정된 본학습·평가 파이프라인에 대한 것이다.

## 반드시 보존할 항목

- `data/tokens/final_v1/` 전체, mix·train/holdout·내용 증명, `tokenizer/candidate_v1/`.
- `data/dedup/exact_v1/clean/` 전체: 최종 웹·수학 원문. 토큰 완성 후에도 native 평가와 source/split fresh 검증이 읽는다.
- `data/clean/github_code_v1/` 전체: 코드의 최종 원문·provenance·정제 통계·정책. **다른 clean 디렉터리와 함께 지우면 안 된다.**
- `data/datasets/`의 ready/source/split manifest·완료 증명, canonical `final_v2.split` DB와 사용 중인 로컬 reader DB 및 reader manifest.
- 모든 `data/raw/*/download_manifest.json`, `data/recovery/*.verified.json`, source별 license/revision·정제 통계·`.done` 등 작은 증거. 원본 payload와 디렉터리째 묶어 지우지 않는다.
- `data/dedup/exact_v1/manifest.json`, `summary.json`, `.done`, `index.sqlite`. index는 학습 직접 입력이 아니어도 이번 삭제 범위에는 넣지 않는다.
- 진행 중 리허설, 학습/진단 체크포인트, 평가 표본/프롬프트, source 감사·split/tokenizer/token 내용 증명, 코드·실행 계약·로그. 학습 이외 파일이라는 이유만으로 증거를 삭제하지 않는다.
- 이 문서와 `docs/recovery/data_retention_v1/`의 복구 묶음. `runs/`는 git ignore되므로 복구 핵심 자료는 문서 디렉터리에도 별도 보존한다.

## 실제 확인한 내용과 한계

`runs/data_retention_review_v1/report.json` 및 연결 증거:

1. 7개 다운로드 manifest의 고정 revision에서 **727개 원격 파일의 존재·크기·LFS SHA256**를 조회했다. 현재 원본727개 전수 SHA가 일치했다. 중간본662개도 전수 SHA를 기록했다. 후보1,389개/667.728GB 해시는 **36.415초**; 같은 서버의 기존32worker 측정 근거를 재사용했다.
2. 6개 실제 소스에서 가장 작은 shard 하나씩, **합계4,672,306,160bytes**를 cache 재사용 없이 새 로컬 경로로 다운로드했다. 전체 파일 크기/SHA256가 원격 및 기존 파일과 일치했다. 모든727개를 다시 다운로드한 것은 아니다.
3. 각 다운로드 파일의 처음4096행을 실제 `data/clean.py` 또는 `data/clean_code.py` 함수로 정제했다. 6소스 모두 원래 정제본의 대응 prefix와 **전체 열의 논리 값이 정확히 일치**했다. 총24,576개 원본 행/19,701개 유지 행 검사다. 668GB 전체 정제·중복제거를 다시 수행하거나 재생성 Parquet의 전역 byte 동일성을 증명한 것은 아니다.
4. 실제 `data/download.py --list-only`를 7가지 복구 명령으로 실행해 파일 선택·revision·기존 manifest 일치를 확인했다. 영어56개와 코드64개 spread를 잘못 선택하는 경로를 피한다.
5. 학습·native 평가·monitored3B·mix/source/split/reader의 현재 계약9개를 대조해 후보 payload 직접 참조0을 확인했다. 후보 전체의 stat/open/ParquetFile 접근을 실패시키는 별도 프로세스에서 기존 ready 판단용 receipt/footer 검사가 통과했고, 후보 접근 시도0·최종726파일을 확인했다. 새 full source/split 물리 검증이나 F9 완료를 대신하는 검사는 아니다.
6. 접근 가능한28개 프로세스의 FD/mmap에서 후보 열린 파일0이었다. 시스템 프로세스 PID1(`docker-init`)/192(`ssh-agent`)는 일부 `/proc` 접근이 제한돼 완전한 전 프로세스 검사를 주장하지 않는다. 삭제 직전 재검사 결과는 위 실행 완료 절과 receipt에 기록했다.
7. 독립 감사는 CPU affinity64..95/해시32worker/네트워크2worker/Arrow2/RSS상한32GiB/로컬출력상한16GiB/1200초 예산이었다. 실제 **99.55초, peak RSS2.053GiB**, OOM0. 30초 로컬 watcher에 등록했고 GPU 작업·진행 중 producer 변경·삭제는 없었다.

고정 commit은 내용 선택을 고정하지만 외부 배포자가 영구 보관을 보장하는 것은 아니다. **현재 재다운로드 가능함을 확인**했으며, 향후 저장소 삭제·접근 변경까지 보장하지 않는다. upstream 유실 시에도 반드시 원본을 복구해야 한다면 별도 원본 백업이 필요하다. 현재 최종 학습 입력은 보존한다.

## 삭제 목록과 추가 정리 후보

정확한 파일별 경로·SHA·크기·mtime·inode는 `docs/recovery/data_retention_v1/candidate_manifest.json`에 고정한다. 이 목록에는 데이터 payload만 있다. `data/raw/` 또는 `data/clean/` 전체를 재귀 삭제하는 명령으로 대체하지 않는다.

추가로 확인한 **5파일/6.131GB(할당5.607GiB)**는 별도의 `additional_candidates_verified.json`에 기록한다. 복구 완료 후 남은 HPLT/OWM `.incomplete` 2개, 손상 FW2 원본과 중단된 `.tmp`, 최종 코드 파일과 SHA256가 동일한 pilot 정제 payload다. 정상 대체 원본의 SHA는 이번 원격 대조에서 검증했다. 주 대상과 합치면 **673.859GB/할당629.933GiB**지만, 두 목록을 분리해 고정했고 추가 범위까지 사용자 승인 후 삭제했다. 과거 실패/진단 run 전체와 SQLite DB를 일괄 정리하지 않는다.

삭제 실행 시에는 승인된 manifest의 SHA를 고정하고 각 파일의 크기/mtime/inode/내용 및 실행 중 소비자 상태를 다시 검사한다. 변경되었거나 사용 중인 파일은 중단하고 목록을 재검토한다. 삭제 후에는 파일별 결과·실제 free 변화와 최종 입력/후속 작업 상태를 `eviction_receipt.json` 및 STATUS에 기록한다. 실제 실행은 `scripts/execute_reviewed_eviction.py --execute`로 완료했으며, 위 실행 절과 receipt를 따른다.

중간 payload 삭제 후 `.done`은 역사적 증거로만 남는다. 기존 `clean.py`는 `.done`이 있는데 payload가 없으면 실패하므로, 이를 완료 데이터로 오인해 기존 경로에서 재생성하지 않는다. 아래처럼 **새 복구 루트**를 사용한다. 오래된 `preflight.py`의 raw/clean 누락 경고는 현 최종 입력의 실패와 구별한다.

## 복구 묶음

`docs/recovery/data_retention_v1/`에 다음을 보존한다.

- `recovery_bundle.tar.gz`: 실제 사용한 download/clean/filter/code-clean/dedup/path 코드, 원본 manifest7개, 정제 marker/통계·정책, 복구 증명, dedup 순서/집계, Python/주요 패키지 버전. 데이터 payload·토큰·체크포인트·비밀키는 넣지 않는다.
- `raw_remote_manifest.json`: 원본727개 각각의 repo/revision/원격 경로/크기/SHA256. 일시적 signed URL 대신 고정 Hub resolve URL을 기록한다.
- `candidate_manifest.json`: 주 삭제 후보1,389개의 현재 내용 SHA와 stat.
- `recovery_cli_verified.json`, `candidate_unavailable_check.json`, `dependency_scan.json`, `review_report.json`: 실제 검증 결과와 한계.
- `CHECKSUMS.json`: 복구 묶음 파일별 SHA256. 원격 업로드나 git commit/push를 수행한 것은 아니다. 별도 기기로 옮길 때 코드 저장소와 함께 이 디렉터리도 보존한다.

## 원본 재다운로드

아래는 복구가 필요한 시점에 실행할 명령이며 이번에 전체 다운로드를 실행한 기록은 아니다. 먼저 `scripts/hardware_profile.py`로 복구 대상 기기의 quota/affinity·RAM·GPU·파일시스템을 확인한다. 기존 데이터/학습과 별도의 새 로컬 복구 루트를 정한다. 원본+중간 정제본만 다시 만들면 최소 약668GB에 작업 여유가 추가되고, dedup까지 만들면 약1TB 수준과 여유가 필요하다. 전체 split/token까지 다시 만드는 용량은 별도로 산정한다.

```bash
# 원 프로젝트 루트에서 시작한다. RECOVERY_ROOT는 새롭고 충분히 큰 경로로 바꾼다.
RECOVERY_PACKAGE="$PWD/docs/recovery/data_retention_v1"
RECOVERY_CODE="$(mktemp -d /tmp/llm-recovery-code.XXXXXX)"
RECOVERY_ROOT="/충분한_로컬공간/새_복구루트"
test ! -e "$RECOVERY_ROOT" || exit 1
mkdir -p "$RECOVERY_ROOT/data/raw"
tar -xzf "$RECOVERY_PACKAGE/recovery_bundle.tar.gz" -C "$RECOVERY_CODE"
# 원본 선택 manifest만 복사한다. 과거 clean/.done은 복구 출력에 복사하지 않는다.
cp -a "$RECOVERY_CODE/data/raw/." "$RECOVERY_ROOT/data/raw/"
```

압축 해제 전 `CHECKSUMS.json`과 archive SHA를 대조하고, 해제 후 `bundle_manifest.json`의 코드/메타데이터 SHA를 검사한다. 현재 검사 환경은 묶음의 `environment.json`에 기록했다. 다른 환경에서는 별도 가상환경에 해당 pyarrow/xxhash/huggingface_hub 등 버전을 맞추고 대표 파일부터 확인한다. 기존 학습 환경의 패키지를 교체하지 않는다.

아래 다운로드는 각 source마다 manifest를 고정하고, 신규 파일의 전체 LFS SHA를 검증한다. `--verify-sha256`는 이미 존재하는 파일도 검증하게 한다. 파일 선택을 먼저 확인하려면 각 명령 끝에 `--list-only`를 붙인다. 이 7가지 선택은 이번에 실제 통과했다. workers8은 이 기기의 전체 복구 출발 후보이며 새 기기/회선에서는2/4/8의 총시간·메모리를 비교해 결정한다.

```bash
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" wiki_ko --revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa --workers 8 --verify-sha256
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" fineweb2_ko --revision af9c13333eb981300149d5ca60a8e9d659b276b9 --workers 8 --verify-sha256
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" hplt_ko --revision d1324a5283f762ee62c2a5c81de08fc6450ea540 --workers 8 --verify-sha256
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" fineweb_edu 56 --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 --workers 8 --verify-sha256
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" owm --revision fde8ef8de2300f5e778f56261843dab89f230815 --workers 8 --verify-sha256
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" github_code_v1 64 --spread --revision b5661e6b17396364b2bcf8e68977b0d28e1ebd19 --workers 8 --verify-sha256
# pilot은 현재 본학습에 불필요하다. 이력까지 전체727파일을 복구할 때만 받는다.
PROJECT_ROOT="$RECOVERY_ROOT" python "$RECOVERY_CODE/data/download.py" github_code_pilot 1 --revision b5661e6b17396364b2bcf8e68977b0d28e1ebd19 --workers 8 --verify-sha256
```

pilot은 현재 본학습에 필요하지 않아, 현 학습용 원본만 재확보하면 마지막 명령을 생략한다. `main`이나 최신 commit으로 바꾸거나 영어56개·코드 spread64개 선택을 바꾸지 않는다. [Hugging Face 공식 다운로드 문서](https://huggingface.co/docs/huggingface_hub/guides/download)의 `revision`/`local_dir` 동작을 따른다.

고정 목록과 다시 대조하는 예시는 다음과 같다. 현재 기기의 CPU 여유에서 최대32개의 병렬 읽기/해시를 사용하며, 경합 시 줄인다. 다운로드 신규 파일 SHA와 같은 실행의 신뢰할 수 있는 증명이 있으면 불필요한 두 번째 전체 pass는 생략하고 그 증명을 연결할 수 있다.

```bash
PROJECT_ROOT="$RECOVERY_ROOT" python - "$RECOVERY_PACKAGE/raw_remote_manifest.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root = Path(os.environ['PROJECT_ROOT'])
items = json.loads(Path(sys.argv[1]).read_text()).values()
def check(r):
    p = root / r['path']
    assert p.stat().st_size == r['bytes'], r['path']
    h = hashlib.sha256()
    with p.open('rb') as f:
        while block := f.read(16 * 1024**2): h.update(block)
    assert h.hexdigest() == r['sha256'], r['path']
with ThreadPoolExecutor(max_workers=min(32, len(os.sched_getaffinity(0)))) as pool:
    list(pool.map(check, items))
print('727 pinned raw payloads verified')
PY
```

위 검사는 pilot을 포함한727개 전체 복구용이다. pilot을 생략했다면 검증 목록에서도 해당 source만 명시적으로 제외하고 검사 수를726개로 기록한다.

## 중간 정제본·중복제거본 재생성

정제 전 다운로드 파일의 크기·내용 SHA를 검증한다. footer 검사만으로 복구 성공을 판단하지 않는다. 기존 손상 사례는 같은 크기·정상 footer에서도 실제 decode가 실패했으며, 재생성 중 오류가 나면 해당 shard를 보존하고 원인을 확인한다.

이 기기에서는 아래의16worker/Arrow1/OMP1을 출발 후보로 두고, 대표 shard의 총시간·worker별 RSS를 확인한 뒤8/16/32 중 선택한다. host128GiB·디스크200GiB reserve와 진행 중 작업의 향후 peak를 먼저 확보한다. 아래 wrapper는 현재 기록된 Python의 fork 방식 worker에 Arrow 설정을 전달한다. 다른 multiprocessing 시작 방식을 쓰는 환경에서는 자식의 실제 설정을 먼저 확인한다.

```bash
run_recovery_stage() {
  PROJECT_ROOT="$RECOVERY_ROOT" OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TMPDIR="$RECOVERY_ROOT" \
    python - "$@" <<'PY'
import runpy, sys
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
PY
}
run_recovery_stage "$RECOVERY_CODE/data/clean.py" wiki_ko ko 16
run_recovery_stage "$RECOVERY_CODE/data/clean.py" fineweb2_ko ko 16
run_recovery_stage "$RECOVERY_CODE/data/clean.py" hplt_ko ko 16
run_recovery_stage "$RECOVERY_CODE/data/clean.py" fineweb_edu en 16
run_recovery_stage "$RECOVERY_CODE/data/clean.py" owm math 16
# 최종 코드 정제본까지 재생성이 필요한 경우에만 추가한다.
run_recovery_stage "$RECOVERY_CODE/data/clean_code.py" github_code_v1 16
```

파일별 유지 수·탈락 사유·문자 수를 묶음의 이전 `.stats.json`과 대조한다. 원문 전체 열·행 순서·문서 수가 맞는지 논리적으로 검사하고, 동일 serializer 환경이면 후보 manifest의 Parquet SHA도 비교한다. 포맷 bytes만 다른 경우 이를 동일 내용으로 임의 승격하지 않고 논리 대조와 새 증명을 만든다. 이번 표본 검증을 전수 재생성 완료로 재사용하지 않는다.

중복제거를 다시 수행하는 경우 source 순서는 반드시 다음과 같다. 먼저 등장한 중복을 유지하므로 순서를 바꾸면 결과가 달라진다. 새 복구 루트의 로컬 저장소에서 실행하고, 완료 후 최종 발행·검증은 별도로 수행한다.

```bash
run_recovery_stage "$RECOVERY_CODE/data/dedup.py" \
  --sources wiki_ko fineweb2_ko hplt_ko fineweb_edu owm \
  --output dedup/rebuilt_v1
```

기존 참고 집계는 input141,588,242 → kept126,707,799 / exact_duplicates14,880,443이다. 코드 정제본은 이 웹 dedup에 넣지 않는다. 현재 dedup manifest는 입력 절대경로·mtime도 묶으므로 새 루트의 결과를 기존 `exact_v1`에 덮어쓰거나 이전 `.done`을 복사해 재개시키지 않는다. 최종 source/split/tokenizer/tokens를 교체하려면 새 감사·내용 증명·단계별 진입 검증과 별도 확인이 필요하다. 보존 중인 현재 학습 입력은 계속 사용한다.
