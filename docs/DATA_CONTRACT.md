> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 최종 데이터 입력 계약과 남은 확정 조건

공개 재배포 조건과 출처 감사는 [최종 정제 데이터 출처와 재배포 조건](DATA_SOURCES_AND_LICENSES.md)을 따른다. 이 문서의 학습용 `ready` 판정은 외부 공개 권리 검토의 통과를 뜻하지 않는다.

병행 작업 B/D의 공통 schema v1이다. 실제 최종 manifest가 생성·검증됐다는 보고가 아니다.

## 소스 입력 manifest

`common/data_contracts.py`가 tokenizer 학습과 토큰화의 입력을 공통으로 해석한다. 다음은 **draft 예시**이며 아직 학습에 사용할 수 없다.

```json
{
  "schema_version": 1,
  "status": "draft",
  "sources": {
    "fineweb2_ko": {
      "text_policy": "nfc-web-exact-v1",
      "completion_marker": "data/dedup/exact_v1/.done",
      "files": ["data/dedup/exact_v1/clean/fineweb2_ko/000_00000.parquet"]
    },
    "github_code_v1": {
      "text_policy": "raw-utf8-code-v1",
      "completion_marker": "data/clean/github_code_v1/.done",
      "files": ["data/clean/github_code_v1/train-00000-of-01126.parquet"]
    }
  }
}
```

위 예시는 두 소스의 일부 경로만 보여 준다. 실제 manifest에는 확정한 여섯 소스의 전체 파일 목록과 source revision·정제/중복 정책·품질 근거를 추가한다. 파일·완료 마커 존재, 중복 경로·중복 basename, schema/status를 검사한다. `status=ready`인 manifest만 사용한다.

`ready`는 사용자에게 추가 승인을 요구하는 상태명이 아니라 아래 데이터 품질·완료 조건이 검증됐다는 실행 기록이다. 검증 없이 값을 바꾸어 진행하지 않는다.

## ready 이전에 확정할 것

- exact_v1 전체 완료·입출력 집계와 소스별 제거량 검증.
- HPLT/FW2 유사중복의 표본·임계값·추가 처리 또는 비중 조정 판단. URL 중복률을 그대로 삭제율로 사용하지 않음.
- 코드↔수학의 완료된 정확 교차검사 재사용. 코드↔한국어/영어, 부분 변형과 평가 오염의 추가 검사 범위 결정.
- 코드 언어 비중 및 실제 토큰량 추정 방법. tokenizer 표본 byte 비율과 학습 token 비율을 구분.
- 중복 그룹을 train/eval 양쪽에 넣지 않는 문서 단위 분리. 코드 repo 단위 분할 정책 검토. 분리 근거를 source row·내용 식별자와 함께 보존.

현재 exact_v1 index·입력 순서·DB는 변경하지 않는다. 원본을 다시 복사하지 않고 완료 결과와 코드 경로를 manifest로 연결한다.

## 토큰 산출물 계약 v2

`data/tokenize_shards.py`는 실제 tokenizer JSON SHA256·입력 parquet SHA256·snapshot ID·EOS·split·format으로 contract_id를 만든다. 같은 vocab 크기만으로 재사용하지 않는다.

소스 디렉터리에는 manifest.json, shard.bin, shard.bin.done, 필요 시 hold.bin, summary.json 및 .done이 생긴다. cast 전 0..65535 정수 범위를 검사하고, 문서마다 실제 EOS ID를 추가한다. holdout 문서는 학습 파일에서 제외한다.

완료 증명은 바이너리의 바이트 수·SHA256을 포함한다. pending 증명을 먼저 남겨 data rename과 완료 마커 사이에서 중단돼도 검증 후 재개한다. 증명이 없거나 정책이 바뀐 기존 결과는 덮어쓰지 않는다. holdout_docs를 목표 토큰 수와 동일시하지 않으며 실제 토큰량을 확인한다.

`data/build_mix.py`는 완료된 source contract만 합친다. 다른 tokenizer/EOS/format 혼합, 짧은 shard, 잘못된 가중치, 실제 파일 크기·내용 불일치는 거부한다. source weight는 정규화하고 선택적으로 목표 학습량에 따른 반복 노출량을 계산한다. 결과는 sampler_version=2를 기록한다.

## 그룹 분리 산출물과 토큰 형식 v3

`data/split_manifest.py`는 ready 입력의 본문을 읽어 별도 SQLite에 최종 파일/row/본문 SHA256 및 그룹을 기록한다. raw exact 본문은 소스 간 연결하고, 웹에는 NFC 동등 그룹을 추가하며 코드에는 실제 repo_name 그룹을 연결한다. 여러 저장소에 같은 코드가 있으면 연결된 저장소까지 한 분할에 배치한다. seed와 안정된 그룹 대표값으로 train/eval을 결정하며, 코드 일부 조각이나 유사중복까지 해결한 것으로 보지 않는다.

완료 산출물은 manifest.json·split.sqlite·.done이다. manifest에는 입력 내용 hash/행 수·source snapshot·그룹 정책·seed·비율·파일별 split 수, DB 내용 hash를 기록한다. 토큰화 전에 DB 무결성·split/그룹 교집합·파일별 row coverage·내용 증명을 검사한다. N7 builder_version=2는 build.json 및 SQLite progress에 동일 입력 내용/정책을 기록하고 `.building`의 commit된 배치부터 재개한다. 입력 hash/seed/비율 변경은 거부한다. 이전 builder의 미확인 partial은 보존하고 새 output-dir로 실행한다. ingest/assign 배치와 progress는 같은 transaction이며 finalize는 단일 transaction으로 rollback/재개한다. DB rename·manifest 발행 뒤의 중단도 복구한다. 7개 강제 종료 지점과 합성 1만/10만 건 검증은 완료했으나 실제 전체 규모 성능/용량을 보증하지 않는다. 재개 시 입력 전체 hash를 다시 읽는 비용과 큰 finalize transaction 비용은 남아 있다.

`tokenize_shards.py --split-manifest`를 쓰면 마지막 N문서 방식과 함께 사용할 수 없다. 소스와 snapshot이 일치한 split을 읽어 **uint16-eos-split-v3**로 쓰며 shard별 `name.hold.bin`, 완료 기록의 holdout_file, source plan의 split_contract_id를 기록한다. 학습 문서가 0개인 shard는 train mix에서 제외하고 평가 토큰은 유지한다. v2와 v3는 같은 mix에 섞지 않는다. 짧은 비어 있지 않은 토큰 shard는 기존 seq_len 검사를 통과해야 한다.

`scripts/validate_source_manifest.py`는 ready manifest의 중복 경로·완료 증거·text 스키마·parquet 읽기/행 수·내용 hash를 검사한다. 성공은 구조 검사 결과이며 데이터 품질이나 정규화 정책을 자동으로 승인하거나 status를 ready로 바꾸지 않는다.

## sampler와 기존 checkpoint

새 기본값은 v2다. 유효 window 수 `n-(seq_len+1)+1`로 샤드를 선택하고 마지막 시작점을 포함한다. version=1을 명시하면 이전 길이 가중·마지막 시작점 제외 규칙을 유지한다. 버전 정보가 없는 과거 checkpoint는 v1로 간주해 v2로 조용히 재개하지 않는다.

새 v2 mix를 v1으로 읽는 것은 거부한다. C2 보완 이후에는 sampler 버전만으로 과거 checkpoint 재개를 허용하지 않는다. `training_contract`와 랭크별 RNG 상태가 없는 이전 smoke checkpoint는 새 실행 디렉터리에서 시작해야 한다. 버전이 있는 checkpoint도 데이터·tokenizer/mix·학습 설정·world size·관련 코드 식별자가 일치해야 재개한다.

`train/runtime.py`는 시작 시 rank 0에서 입력 토큰/holdout 전체 내용 SHA256을 계산하고 다른 랭크에 계약을 전달한다. 큰 실제 데이터에서는 시작 I/O 비용이 발생하며 아직 그 비용을 측정하지 않았다. 현재 계약에는 mix에 기록된 tokenizer SHA256, 소스 순서, 파일 내용, 학습률 일정·배치·seed 등의 설정이 들어간다. 재개할 때는 입력 파일을 수정하지 않는다.

중간 저장·재개 비교는 `max_steps`를 바꾸지 않고 `stop_after`로 일찍 종료한다. `stop_after`는 계약 비교에서 제외되지만 학습률 일정을 결정하는 `max_steps`는 포함된다. 작은 8GPU 합성 검증은 `runs/c2_validation_v2/summary.json`에서 통과했으며 실제 데이터·본학습 크기·compile 재개는 별도 검증한다.

## 다음 실행 예시 — 준비 완료 후에만

```bash
# 최종 source ready·품질/그룹 정책 확정 후 실제 split 생성
python data/split_manifest.py --sources-manifest data/datasets/final_v1.json --output-dir data/datasets/split_v1 --eval-fraction 0.01 --seed 1234
# 비율은 예시. 실제 분포·용량 확인 후 후보 tokenizer 생성
python tokenizer/train_tokenizer.py --sources-manifest data/datasets/final_v1.json --split-manifest data/datasets/split_v1/manifest.json --mix fineweb2_ko:0.35 hplt_ko:0.10 wiki_ko:0.05 fineweb_edu:0.35 owm:0.075 github_code_v1:0.075 --mb 3000
# 위 비율은 사용법 예시이며 확정된 표본 구성은 아니다.
# 후보 검증·내용 해시 고정 후 각 소스 토큰화
python data/tokenize_shards.py fineweb2_ko --sources-manifest data/datasets/final_v1.json --tokenizer tokenizer/candidate/tokenizer.json --output-dir data/tokens/v2/fineweb2_ko
# 모든 소스 토큰량/holdout 확인 후 mix 작성
python data/build_mix.py fineweb2_ko:0.35 hplt_ko:0.10 wiki_ko:0.05 fineweb_edu:0.30 github_code_v1:0.10 owm:0.10 --token-root data/tokens/v2 --output data/tokens/v2/mix.json
```

mix도 사용법 예시다. 실제 실행에서는 사용할 소스만 양수 가중치로 나열하고 전체 비율을 실제 양에 맞춰 확정한다. 이 문서 작성 중 실제 tokenizer/토큰화/학습은 실행하지 않는다.

## Tokenizer 정규화 결정 (N5, 2026-11-06)

새 후보 학습 기본은 `--normalization raw`: 코드뿐 아니라 모든 입력의 원문 Unicode를 보존한다. 소스에 따라 tokenizer 동작을 바꾸지 않는다. 기존 웹 NFC 중복/그룹 판정과 구분하며 기존 clean을 다시 쓰지 않는다. checker는 호환상 기본 NFC이므로 새 후보에는 반드시 `--normalization raw`를 명시한다. normalizer 설정 자체와 roundtrip을 함께 검사하고 정책을 보고서 계약에 포함한다. NFC 비교의 needs_review 결과를 raw 합격으로 다시 표시하지 않으며 별도 출력 경로를 사용한다.

## N9 품질/완료 근거와 candidate 감사

`data/quality_report.py`와 `scripts/check_data_readiness.py`는 candidate inventory에서 source/revision·완료 shard receipt·footer 행 수·정책·집계를 확인한다. active dedup SQLite를 열지 않는다. origin raw manifest를 지정한 경우 repo/revision 및 선택 파일 이름도 대조한다. `.done`이 실행 도중 생성되는 경우는 시작 시 완료 상태를 기준으로 처리하고 다음 조회에서 재평가한다.

반환 `run_status=completed`는 보고서 작성 종료다. `readiness=passed|pending|failed`와 구분한다. `.done` 증명도 보고서 내용/실행 종료만 증명한다. 웹 aggregate와 코드 receipt/footer reconcile 외에 실제 내용 감사와 near_duplicate_policy/code_cross_policy/holdout_policy/tokenizer_byte_mix 근거가 필요하다. 정책 pass에는 이유·범위·행동과 내용 hash로 고정한 증거를 요구하며, 이를 전체 자료의 무중복 증명으로 표시하지 않는다. holdout 정책에는 seed/비율/repo grouping, tokenizer에는 전체 source의 양수 byte weights/raw 정규화를 요구한다.

`validate_source_manifest.py --candidate`는 완료된 draft의 text 전체 읽기/hash 감사를 허용하지만 manifest를 바꾸지 않는다. 기본 ready 감사와 학습·토큰화의 ready-only 입력 계약은 유지한다. 실제 전체 감사는 dedup 완료 후 I/O 비용을 따져 실행한다. candidate/quality 입력이나 증거가 바뀌면 새 보고서 경로를 사용한다. runs/n9_readiness_*의 현재 pending은 최종 source 준비 완료가 아니다.

## N10 분리 입력과 tokenizer 학습 계약

`data/split_inputs.py`는 ready source 및 검증한 split SQLite를 열어 전체 source 파일 내용 hash를 비교한다. 실제 대량 적용 시 이 검증 비용이 있으므로 I/O 여유를 따져 실행한다. file/row/body SHA256/group/repo로 문서를 추적하며, seed로 file/row group 순서를 고정한 뒤 해당 split의 row만 batch로 읽는다. 이는 corpus 전체의 균일 무작위 표본이 아니다.

최종 tokenizer CLI는 `--split-manifest`를 요구한다. 명시 `--experimental-unsplit`은 예전 실험 경로이며 검증한 eval 제외로 표시하지 않는다. train 표본의 byte 예산은 사용 텍스트 기준이며 문서 하나까지 초과할 수 있고 실제 값/overshoot를 기록한다. 잘림·원문 hash·사용 텍스트 hash를 구분한다. 부족 표본은 실패하고 조용히 source를 빼지 않는다. samples.jsonl·manifest.json·.done을 보존하며 tokenizer 계약에 split/표본 SHA/설정/코드를 포함한다.

평가 입력은 `eval/prepare_holdout.py`가 eval row에서만 ko/en mapping으로 추출한다. 원문 잘림·EOS 삽입 없이 저장하며 기존 PPL 경로가 EOS를 1회 붙인다. `eval/run.py --holdout-manifest`는 데이터 증명/JSONL hash뿐 아니라 SQLite row membership·본문 hash·group/repo를 확인한다. 수동 JSONL 입력도 사용할 수 있지만 보고서 provenance는 검증되지 않은 입력으로 구분한다. 같은 본문이라도 source/split/설정이 바뀐 산출물을 조용히 재사용하지 않는다.

## N11 제한 진단 입력

sample frame은 읽을 완료 파일 목록을 고정한다. 선택 row group의 사용 column uncompressed 크기를 읽기 전에 예산에 계상하고, row/저장 byte/row group/file/시간 상한을 기록한다. source 전체 hash 검증 여부는 false로 표시하고 실제 선택 body SHA와 size/mtime/완료 증명을 남긴다. `sample_status=partial`과 `complete_for_defined_frame`은 전체 corpus 완료를 뜻하지 않는다. 실행 완료 증명과 sampling/진단 상태를 함께 확인한다.

웹은 NFC+공백 정리된 제한 view의 5-gram hash 집합을 사용한다. 낮은 16개 hash 중 2개를 공유하는 후보만 실제 Jaccard로 검사하므로 후보 재현율이 보장되지 않는다. `source_pairs`는 실제 검증 후보가 있었던 source pair의 집계이며, 키 부재는 전체 비교/무중복을 뜻하지 않는다. 코드↔웹은 whole-document raw SHA와 잘리지 않은 코드 전체↔선별 fenced block만 검사한다. 잘린 코드 view는 block index에서 제외하고, block 자체가 없으면 블록 교차 여부를 판단하지 않는다. 실제 전체 자료의 중복률·삭제 정책과 별도로 품질 판단 근거를 남긴다.

## F0/F1 재개 가능한 전체 감사와 ready 바인딩

새 대량 감사는 `scripts/audit_source_inventory.py`와 `data/source_audit.py`를 사용한다. 기존 validate_source_manifest는 작은 단발 구조 감사/legacy API로 유지한다. 새 감사는 source completion/shard receipt와 footer count를 확인하고 웹 text/h, 코드 사용 provenance와 body SHA/source_row를 decode/검증한다. 파일 단위 원자적 receipt와 raw body/fenced-block fingerprint Parquet를 별도 출력에 저장한다. 원문은 다시 저장하지 않는다.

완료 파일 재개는 원본 전체 SHA와 지문 artifact SHA/행수까지 확인한다. size/mtime만으로 reuse하지 않는다. 미완료 파일은 처음부터 재검사하며 fingerprint rename→receipt→최종report 발행의 강제 종료 복구를 검증했다. job budget pause는 재개 가능한 상태이고 신규 source/code/설정 계약은 새 경로를 요구한다. `.done`은 final report SHA를 증명한다. pilot의 passed_selected_files는 전체 content audit로 사용할 수 없다.

`common/source_inventory.py`는 source 정의와 감사한 file path/bytes/rows/SHA 기반 content inventory ID를 top-level status/created time에서 구분한다. `finalize_source_manifest.py`는 내용이 같은 candidate의 **전체 감사 + 모든 quality 조건 passed**를 확인하고 실제 파일 SHA를 재검증한 뒤 별도 ready와 완료 증명을 만든다. 실제 production policy는 아직 pending이다. source_files는 바인딩된 ready의 audit/content ID/완료 증명과 실제 파일 SHA를 검증한다. 기존 테스트/legacy unbound ready는 호환상 남아 있으나 새 최종 source의 audited-ready 증명으로 표시하지 않는다.

감사 지문은 raw 전체 본문 SHA와 웹 closed fence(80자 이상)의 CR/LF 경계만 제외한 SHA를 기록하며, 코드에는 whole-code block key를 기록한다. 후속 코드↔웹 검사에서 지문 파일 SHA를 확인하고 사용할 수 있다. 파일 크기 대비 checksum 산출·decode·지문 쓰기를 포함한 실제 I/O 비용은 pilot으로 측정한다. 이 지문만으로 arbitrary inline/변형 코드 또는 웹 유사중복이 해결됐다고 하지 않는다.

## 감사 지문 소비 / 후보 누락 검증 / split 확장 측정

`data/probe_audit_overlap.py`는 report 완료 증명과 지문 artifact SHA/schema/모든 row coverage를 검증한다. 기본 full audit만 허용하고 명시적 pilot은 `full_source_coverage=false`로 보고한다. full audit여도 자원 상한으로 일부만 소비하면 전체 coverage를 통과시키지 않는다. whole raw match pair, matched web document, eligible fence, matched fence/pair를 별개 집계하고 원문 삭제 없이 source/file/row 예시를 남긴다. 2026-11-06 pilot0쌍은 해당6파일 범위의 결과다.

`data/probe_similarity_recall.py`는 SHA로 식별한 저장 표본에서만 제한 view를 만들고 실제 후보 생성기를 observer로 관찰한다. 독립 unhashed NFC+공백 정규화5-gram 집합의 source간 완전 비교를 기준으로 threshold별 양성/탐지/누락/오탐을 비교한다. 실제/합성/혼합 지표를 분리하며 양성0쌍이면 recall=null이다. candidate cap/time에 걸리면 부분 상태를 보존하고 전체 corpus 재현율을 주장하지 않는다.

`scripts/measure_split.py`는 synthetic fixture용이다. 입력 크기100~3,000,000행, 단계 heartbeat/별도 worker/time·RSS·case bytes/free-space 감시와 최종 HWM 확인, SHA 검증 완료 cache 및 같은 계약 중단 재개를 제공한다. SQLite cache1~1024MiB(기본64)는 성능 설정이고 논리 분리 계약을 바꾸지 않는다. runner는 cache/코드/환경/입력/출력까지 묶어 과거 측정을 구분한다. 완료 cache 재사용은 새 측정이 아니며 부분 시도의 누적시간과 현재 build시간을 구분한다. 실제 production source ready와 별도이고 활성 full audit 중10만 초과 측정은 차단한다. sampled journal/출력 peak를 정확한 최대값으로 표시하지 않는다.
