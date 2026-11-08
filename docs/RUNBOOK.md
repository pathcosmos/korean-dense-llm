> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# RUNBOOK (모두 프로젝트 루트에서 실행)

2026-11-07 본학습 보존 확대 준비: [CHECKPOINT_RETENTION](CHECKPOINT_RETENTION.md)의 정확한 새 run/NFS·로컬 namespace와 `--prepare --keep-checkpoints 10 --local-history 15 --retention-plan ...`을 따른다. 현재 실제 최종 보고서가 없어 준비 명령도 의도대로 거부되며, 본학습/운영 정리 승인은 별도다. 예전 `keep_ckpts=2/3` 또는 native10을 그대로 실행하지 않는다. 검증된 새 구성의 native 값은57001이고 외부 감독기가10+15 보존을 담당한다.

2026-11-07 원본/중간본 복구 안내: [DATA_RETENTION_AND_RECOVERY](DATA_RETENTION_AND_RECOVERY.md)에 삭제 후보1,389파일·현재 소비자 보호·고정 revision 재다운로드7명령·새 루트 정제/중복제거 절차를 기록했다. `docs/recovery/data_retention_v1/`에 코드/메타데이터 복구 묶음과 파일별 SHA를 보존한다. **추가 후보5개를 포함한1,394개/673.859GB 삭제 완료, NFS 여유629.931GiB 회수 확인.** 실행 receipt·후속 공간 관측·journal archive도 복구 디렉터리에 보존했다. 보존된 최종 입력으로 학습/평가를 계속할 수 있으면 raw를 다시 받을 필요가 없다.

2026-11-06 입력 계약 변경: 단계 3~4는 [DATA_CONTRACT](DATA_CONTRACT.md)의 ready manifest·명시적 소스 비율·별도 후보/출력 경로를 사용한다. 현재 `data/datasets/final_v1.draft.json`은 미완료 inventory이며 사용할 수 없다. 실제 진행 상태는 STATUS를 우선한다.

```
# 0 환경
bash scripts/setup_env.sh ; python scripts/make_manifest.py --verify
# 1 다운로드 (재시작 가능; names: fineweb2_ko hplt_ko wiki_ko fineweb_edu owm ; 두번째 인자=최대 파일 수)
python data/download.py hplt_ko ; python data/download.py fineweb_edu 56 ; python data/download.py owm
# 2 정제  (src lang workers) ; 샤드당 메모리와 cgroup memory.max로 workers 결정
python data/clean.py hplt_ko ko 16 ; python data/clean.py fineweb_edu en 16
python data/probe_filters.py data/raw/<src>/.../x.parquet 20000     # 필터 제거율 사전 점검
# 3~4: DATA_CONTRACT.md의 ready/품질 조건 통과 후 명령 사용
# train_tokenizer.py 최종 경로에는 --sources-manifest/--split-manifest/--mix 필수, 기본 출력 tokenizer/candidate/tokenizer.json
# check_tokenizer.py는 --tokenizer/--samples/--output 명시: 아래 N1~N4 절 참조
# 5 학습  (1GPU/8GPU 동일 코드; --set key=value 로 덮어쓰기)
torchrun --standalone --nproc_per_node=8 train/train.py --config train/configs/1b_rehearsal.json
# 재개: 같은 명령 재실행 (out_dir의 마지막 완전한 ckpt 자동 로드)
```
- 로그: `runs/<name>/log.jsonl` (loss, gnorm, tok_s, mfu, holdout_loss). 체크포인트: DCP 디렉터리 `ckpt_XXXXXXX/` (.complete 있는 것만 유효).
- 샘플 단위 스키마: FW2 컬럼 text,url,...; HPLT text,u,...; FineWeb-Edu text,...; OWM text,url,...; clean 결과 [text, h(int64 xxh64)].

## 데이터 준비 중 GPU 성능 측정

`python scripts/benchmark_gpu.py --run-dir runs/gpu_sweep_<새이름>`

- 8GPU가 비어 있을 때 실행. tiny compile 확인 후 1B 배치 2, 3B 배치 2·4의 compile off/on을 순차 비교한다. 실제 모델은 길이 4096, global batch 131,072 tokens, 각 40 steps이며 최초 10 steps를 성능 집계에서 제외한다.
- 합성 데이터·설정·로그·결과는 지정한 runs 경로에 저장한다. 본학습 데이터와 tokenizer는 필요하지 않다. 평가와 체크포인트 저장은 비활성화하며 일반 학습의 기본 저장 동작은 유지된다.
- 각 조건은 최대 15분 후 종료하고, 오류나 비유한 수치가 있으면 다음 조건을 실행하지 않는다. 성공한 조건의 `.done`과 `result.json`, 전체 `summary.json`을 확인한다. 같은 명령으로 재실행하면 성공한 조건을 건너뛴다.
- `max_allocated_gib`/`max_reserved_gib`는 전체 랭크 중 최대 PyTorch 메모리이며 nvidia-smi 전체 사용량과 다르다. compile 초기 비용은 첫 로그 처리량과 전체 실행 시간에 포함되고, 안정 구간 처리량에서는 제외된다. `mfu`는 코드상 추정이며 GPU 사용률이 아니다.
- 모든 조건 완료 후 종료하는 검증 작업이다. 실제 mix·평가·재개 검증 및 본학습 승인 절차는 별도로 필요하다.

## 추가 데이터 품질 점검

```bash
# URL 컬럼만 전체 스캔하고 같은 URL 해시 표본으로 겹침 추정 (텍스트 dedup 대체 아님)
python data/probe_url_overlap.py --one-in 100
# OWM 다운로드/검증 완료 및 필터 표본 확인 후 수학 전용 모드로 정제
python data/clean.py owm math 4
```

URL 측정 결과: `data/samples/url_overlap_v1/summary.json` (`.done` 생성 시 완료).
OWM 필터 사전 비교: `logs/owm_filter_comparison.json`. 단계별 완료 여부는 STATUS.md 참조.

## 정제 후 정확중복 제거

```bash
# 원본 clean은 보존. 모든 소스의 raw manifest 대상 정제가 끝난 후 실행.
python data/dedup.py --sources wiki_ko fineweb2_ko hplt_ko fineweb_edu owm --output dedup/exact_v1
```

- 결과: `data/dedup/exact_v1/clean/<source>/*.parquet`, 집계 `summary.json` 및 `.done`.
- 먼저 지정한 소스의 문서를 보존. NFC 정규화 외 대소문자·공백은 유지, 최종 정제/마스킹 텍스트 SHA256 사용.
- 입력 파일/순서/정책이 달라지면 새 output 경로 필요. 정확중복 전용이며 MinHash 유사중복을 대체하지 않음.
- 후속 대기 프로세스: `logs/clean_fineweb_edu_followup.log`, `logs/dedup_followup.log`. 각각 같은 이름의 `.pid`로 상태 확인. 이미 실행 중이면 중복 실행하지 않음.
- 최종 tokenizer·토큰화 입력은 검증된 dedup 출력과 코드 별도 경로를 ready source manifest로 연결한다. 토큰화의 기존 input-dir 경로는 시험/명시적 입력 용도이며 최종 품질 조건을 대신하지 않는다.

## 상태 변화 확인

사용자 선택: 대화 중 보고 + 서버 이벤트 기록. 외부 푸시는 연결하지 않음.

```bash
cat logs/pipeline_state.json
tail -20 logs/pipeline_events.jsonl
# 감시 프로세스 상태: logs/pipeline_watch.pid 참조
# 실행 중이면 새 watcher를 중복 실행하지 않는다.
python -u scripts/watch_data_pipeline.py --interval 30
```

30초 간격으로 오류·예상 밖 종료·단계 완료·정체와 주요 진행 이정표를 기록한다. 현재 설정 대상은 데이터 정제·코드 확보·exact dedup·코드/수학 교차검사다. 컨테이너 재시작 후에는 감시도 다시 실행해야 한다. 기록은 자동 대화 메시지가 아니다.

## 별도 코드 후보 확보·정제

```bash
python data/download.py github_code_v1 64 --spread --revision b5661e6b17396364b2bcf8e68977b0d28e1ebd19 --workers 1
# 다운로드 완료 후 (후속 프로세스 실행 중이면 중복 실행하지 않음)
python data/clean_code.py github_code_v1 2
```

본수집: `logs/download_github_code_v1.log`, 후속 정제: `logs/clean_github_code_v1_followup.log`.
`data/clean/github_code_v1/policy.json`, `summary.json`, `.done` 및 파일별 source row로 출처/검증 확인.
웹용 cleaner 대신 코드 전용 cleaner를 사용. 원문/들여쓰기/대소문자/Unicode를 보존하며 provenance 열을 유지한다. 현재는 샤드 내부 dedup만 적용; cross-source 코드 처리 전 기존 exact_v1에 입력을 추가하지 않는다.

## 코드·수학 읽기 전용 교차검사

```bash
python -u data/probe_code_math_overlap.py
```

완성된 코드/OWM clean snapshot을 읽는다. 기존 exact_v1 DB·입출력은 수정하지 않는다.
문서 전체 raw UTF-8 일치와 fenced code block(기본 80자 이상) 일치를 별도 기록한다.
결과 `data/samples/code_math_overlap_v1/summary.json` 및 `.done`, 진행 `logs/probe_code_math_overlap.log`.
블록 일치만으로 문서 전체를 삭제하지 않으며 inline/변형 유사중복 검사는 포함하지 않는다.

## 병행 개발 검증 및 재개 호환성

```bash
# 임시 입력과 작은 CPU 모델만 사용
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
# 입력 완료량만 inventory에 반영하며 ready로 변경하지 않음
python scripts/prepare_source_manifest.py
```

2026-11-06: 39개 통과(`logs/tests_parallel_final.log`). A sampler D01/D02, B token/cache T01/T02, C1 PPL E01~E03 포함. `eval.perplexity.evaluate_sequences`와 `eval.generation`은 자체 모델용 단일 프로세스 API이며 아직 실제 한/영 결과 저장·8GPU 평가 경로에 연결하지 않았다.

새 sampler 기본은 v2다. v2 mix를 v1 sampler로 사용하는 것은 거부한다. C2 이후에는 training_contract·랭크별 RNG가 없는 과거 checkpoint를 sampler_version=1만 지정해 재개할 수 없다. 과거 산출물은 보존하고 새 out_dir에서 검증한다. 계약이 있는 checkpoint는 데이터·설정·world size·코드가 동일할 때 재개한다.

## C2 짧은 분산 회귀 검증

```bash
# GPU 8장이 비어 있을 때, 존재하지 않는 새 runs 경로 사용
OMP_NUM_THREADS=2 python scripts/validate_distributed.py --run-dir runs/c2_<새이름>
```

- 기존 GPU benchmark와 같은 잠금 사용. micro 모델, vocab 128, seq 32, 4 steps, 8GPU, compile=false, 작은 합성 데이터로만 검증한다. 각 torchrun은 240초 제한이다. 실행 경로가 있으면 거부하여 이전 증거를 보존한다. 중단 시 failure/log를 확인하고 새 이름으로 전체 suite를 재검증한다.
- 연속 실행 ↔ 2 step 저장 후 4 step까지 재개, 가중치·optimizer·랭크 RNG·LR/loss·sampler batch 비교. `stop_after=2`로 중단하며 max_steps=4는 유지한다. `.complete`가 없는 더 최신 checkpoint는 선택하지 않는다.
- holdout 1/7/9 windows의 분산 점수를 전 문서 순차 계산 기준과 비교한다. NaN loss와 gradient를 각각 rank 3에만 주입하고 모든 랭크에서 optimizer 갱신 전 거부 및 오류 checkpoint 미생성을 확인한다. 오류 주입은 검증 harness 안에서만 수행한다.
- 결과 `summary.json`·`.done`, 사례별 console·config·checkpoint·평가/오류 증거. 상태 변화는 logs/pipeline_events.jsonl에도 기록한다. tokenizer 변경 거부 사례의 nonzero exit는 기대한 결과이며 suite 자체 실패와 구분한다.
- 2026-11-06 `runs/c2_validation_v2`: 6개 실행 사례 통과, 비교 텐서 84개 최대 절대차 0, 전체 CPU 테스트 43개 통과(`logs/tests_c2_cpu.log`). 실제 한/영 평가·본학습 크기·compile=true 재개는 미검증이다.

## N1~N4 도구 사용

아래 입력 경로는 실제 준비 후 사용한다. 현재 final source manifest는 draft이고 final tokenizer는 미확정이다. 2026-11-06 N1~N4의 작은 입력 검증과 실제 데이터 품질 검증을 구분한다.

```bash
# N1: JSONL 각 행 {"category":"ko|en|code|math","text":"..."}; 네 범주 모두 필수
python tokenizer/check_tokenizer.py --tokenizer tokenizer/candidate/tokenizer.json --samples data/eval/tokenizer_samples.jsonl --output logs/tokenizer_candidate_check.json --normalization raw
# N2: 새 runs 디렉터리, GPU가 비어 있을 때. 현재 n2_compile_v1은 이미 완료됐으므로 반복 불필요
OMP_NUM_THREADS=2 python scripts/validate_distributed.py --compile --case-timeout 900 --run-dir runs/n2_compile_<새이름>
# N3: samples는 ko/en text JSONL, prompts는 추가로 고유 id 필수. 기본 CPU
OMP_NUM_THREADS=2 python eval/run.py --checkpoint runs/<실행>/ckpt_<step> --tokenizer tokenizer/candidate/tokenizer.json --samples data/eval/holdout.jsonl --prompts data/eval/prompts.jsonl --output logs/native_eval_<새이름>.json --seq-len 4096 --stride 2048
# N4: 전체 ready/품질 조건 확인 후에만 실제 스캔·분리 실행
python scripts/validate_source_manifest.py --manifest data/datasets/final_v1.json --output logs/final_sources_audit.json
python data/split_manifest.py --sources-manifest data/datasets/final_v1.json --output-dir data/datasets/split_v1 --eval-fraction 0.01 --seed 1234
python data/tokenize_shards.py <src> --sources-manifest data/datasets/final_v1.json --split-manifest data/datasets/split_v1/manifest.json --tokenizer tokenizer/candidate/tokenizer.json --output-dir data/tokens/v3/<src>
```

eval-fraction=0.01은 사용법 예시이며 최종 한/영 holdout 토큰량을 보장하지 않는다. 실제 그룹 크기/소스 분포와 토큰량으로 비율을 판단한다.

N1 종료 코드: 0 passed, 1 failed, 2 needs_review. 코드 literal의 NFC 원문 변화는 일반 NFC roundtrip과 별도로 보고하며 exit 2에서 합격 .done을 만들지 않는다. 필요한 정책 검토를 건너뛰어 상태만 바꾸지 않는다. 외부 비교 tokenizer 다운로드는 하지 않는다.

N3는 checkpoint .complete·모델 키/shape·tokenizer 식별을 검사한다. ko/en 문서에 EOS를 추가한 PPL, 토큰 가중 전체 값, Base 고정 생성 및 입력/프롬프트/checkpoint/설정 식별자를 저장한다. 같은 경로의 다른 입력 보고서를 덮어쓰지 않는다. 실제 품질 점수는 실제 holdout snapshot으로 평가한 뒤 보고한다.

N4 그룹 v3는 각 shard의 .hold.bin을 별도로 기록한다. SQLite 분리 결과의 증명/입력 내용과 snapshot을 확인하고 train/eval을 선택한다. 평가 전용 shard의 빈 train 파일은 mix에서 제외한다. v2와 v3 형식 혼합은 거부한다. 구조 감사 성공은 품질 ready 판정과 별개다. N7 builder_version=2는 같은 명령/입력 hash로 부분 SQLite build를 재개한다. 입력·seed·비율 변경 또는 예전 builder의 partial은 거부하므로 보존 후 새 output-dir을 사용한다.

최종 회귀: logs/tests_n1_n4_final.log **52개 통과**. GPU compile=true: runs/n2_compile_v1/summary.json **6개 사례 통과**, 연속/재개 텐서 84개 최대차 0. 실제 모델 크기·데이터 품질 검증은 남아 있다.

## N5~N8 추가 독립 검증 (2026-11-06)

- N5: `scripts/compare_normalization.py --source ko:<parquet> --source en:<parquet> --source code:<parquet> --source math:<parquet> --output runs/<새이름>`은 파일당 최대 256행을 읽어 128개 고정 표본(학습 96/평가 32)을 선택한다. 입력 행·본문 hash·파일 size/mtime, 잘림 한도 4096자를 남긴다. 실제 전체 source hash/대표 표본/최종 후보 학습이 아니다. 기존 runs/normalization_v1은 완료, 반복 불필요. 새 tokenizer 학습 기본 raw, checker에는 raw 옵션을 명시한다.
- N6: `OMP_NUM_THREADS=2 python -m unittest discover -s tests -p test_model_reference.py -v`. CPU FP32 독립 per-head causal attention/RoPE/norm/SwiGLU 계산 및 전체 gradient, tied/untied·KV 1/2/4·길이 1/7의 12조합. GPU bf16/본학습 크기의 수학 검증으로 확대 해석하지 않는다.
- N7: `OMP_NUM_THREADS=2 python scripts/measure_split.py --output runs/<새이름> --sizes 10000 100000`. 최대 10만/케이스로 제한한 짧은 합성 데이터 측정이다. 기존 runs/split_scale_v1 완료. 7개 실제 프로세스 강제 종료 지점은 tests/test_split_recovery.py에서 검증했다. 실행 중인 exact_v1 DB와는 별도다.
- N8: `OMP_NUM_THREADS=2 python eval/multiple_choice.py --checkpoint runs/<실행>/ckpt_<step> --tokenizer <검증한 tokenizer> --task <task.json> --examples <examples.jsonl> --output logs/<새보고서>.json`. 기본 CPU. 명시적 버전·템플릿·정답 인덱스·점수 방식과 입력/checkpoint/tokenizer/code hash를 저장한다.

객관식 task.json 예시(공식 벤치마크 설정 아님):
```json
{"schema_version":1,"name":"local-example","revision":"fixture-v1","template":"{question}\n{choices}\nAnswer:\n","target":"label","scoring":"sum"}
```
각 JSONL 행은 `{"id":"q1","question":"문제","choices":["답1","답2"],"answer":0}`. 정답은 0부터 시작하며 `target=label|choice_text`, `scoring=sum|mean`을 명시한다. label 방식은 template에 choices를 표시해야 한다. 프롬프트+선택지를 함께 encode했을 때 prompt token prefix가 바뀌면 오류로 거부한다. context 초과를 자동 자르지 않는다. EOS를 추가하지 않고 선택지 토큰만 채점하며 동점은 첫 인덱스/동점 목록을 기록한다. 실제 KMMLU·KoBEST·HAE-RAE의 원본 스키마 변환/버전/공식 템플릿은 아직 연결하지 않았다.

전체 CPU 회귀: logs/tests_n5_n8_final.log **62개 통과**. 이번 작업은 모델/학습 런타임을 수정하지 않았고 기존 GPU suite를 다시 돌리지 않았다. 최종 데이터·tokenizer·실제 benchmark 평가는 남아 있다.

## N9→N10→N11 사용 및 현재 증거

현재 도구/작은 입력 검증은 완료했으며 실제 readiness는 pending이다. 기존 n9/n10/n11 결과 경로는 보존한다. 아래 `<...>`는 준비 후 대체하며 지금 전체 데이터에 실행하라는 명령이 아니다.

```bash
# N9: 새 시점 보고서. exit 2=pending은 실행 실패와 구분
python scripts/check_data_readiness.py --sources-manifest data/datasets/final_v1.draft.json --dedup-root data/dedup/exact_v1 --policy runs/n9_readiness_v2/policy.json --output runs/<새보고서>/report.json
# 모든 source 완료 후 candidate full text 읽기/hash 감사. ready로 변경하지 않음
python scripts/validate_source_manifest.py --candidate --manifest <candidate.json> --output logs/<새감사>.json
# N10: 실제 ready+split 및 byte 구성 확인 후 최종 후보 (source/weight는 확정값)
python tokenizer/train_tokenizer.py --sources-manifest <ready.json> --split-manifest <split/manifest.json> --mix <src:weight> --mb 3000 --sample-dir data/samples/<새train표본> --output tokenizer/candidate/<새후보>.json
# 검증된 eval row만, 한/영 source mapping과 UTF8 예산 명시
python eval/prepare_holdout.py --sources-manifest <ready.json> --split-manifest <split/manifest.json> --source fineweb2_ko:ko:10000000 --source fineweb_edu:en:10000000 --output-dir data/samples/<새holdout>
python eval/run.py --checkpoint <checkpoint> --tokenizer <검증후보.json> --samples data/samples/<새holdout>/samples.jsonl --holdout-manifest data/samples/<새holdout>/manifest.json --prompts <prompts.jsonl> --output logs/<새평가>.json --seq-len 4096 --stride 2048
# N11: 이미 존재하는 고정 frame으로 새 표본을 만들 때만 실행
python data/sample_snapshot.py --frame runs/n11_overlap_v2/frame.json --output-dir data/samples/<새표본> --per-source 64
python data/probe_text_similarity.py --samples-manifest data/samples/<새표본>/manifest.json --output runs/<새진단>/web.json --threshold 0.8
python data/probe_code_web_overlap.py --samples-manifest data/samples/<새표본>/manifest.json --output runs/<새진단>/code_web.json
```

N11 CLI는 기본 상한을 사용한다. 실제 v2는 Python API에서 max_read_bytes=256MiB/max_row_groups=8/max_files_per_source=2/row_groups_per_file=1/max_seconds=180 및 source당64건을 명시한 별도 실행이며 logs/n11_overlap_v2.log와 manifest contract에 보존했다. CLI 예시는 그 실행의 동일 재현 명령이 아니다. 임의의 전체 corpus frame으로 확장하지 않는다.

보존된 증거: N9 시점별 report/.done와 policy, N10 fixture sources/split/train/eval/vocab272 tokenizer, N11 v1 partial 및 v2 작은 균형 frame. N11 v2 256문서·약93.2MiB 읽기 계상·0.83MiB 본문 저장, 웹 후보22쌍에서 Jaccard>=0.8 없음·코드 whole exact0·비교 eligible fence0이다. 전수 중복 제거 완료와 구분한다.

현재 남은 작업은 전체 dedup 완료·실제 내용 감사, 교차/유사중복 범위 결정, holdout 비율과 tokenizer byte 구성 확정이다. 본학습 크기 GPU 검증/공식 benchmark 연결은 후속이며 이번 작업에서 GPU를 사용하지 않았다.

최종 전체 CPU 회귀: logs/tests_n9_n11_final_v3.log **74개 통과**. 최신 실제 readiness 보고서: runs/n9_readiness_v3/report.json(pending), 근거 policy는 runs/n9_readiness_v2/policy.json.

## F0/F1 실제 전체 감사

현재 실제 source는 새 `final_v1.candidate.json`(draft)로 고정했다. `source_audit_pilot_v1`은 실제6파일 pilot이고 `f0_binding_validation_v1`은 4문서 fixture다. 과거 결과를 덮어쓰지 않는다.

```bash
# 소스별 median 크기1파일씩. 아래 경로는 실제 새 이름으로 지정
OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 python -u scripts/audit_source_inventory.py --manifest data/datasets/final_v1.candidate.json --output-dir runs/<새pilot> --pilot --max-seconds 1800 --max-rss-gib 16 --min-free-gib 200
# pilot 결과 검토 후 전체726파일. 같은 계약/출력 경로로 실행하면 파일 단위 재개
OMP_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 python -u scripts/audit_source_inventory.py --manifest data/datasets/final_v1.candidate.json --output-dir runs/<새full> --max-seconds <측정후초단위예산> --max-rss-gib 16 --min-free-gib 200
# 모든 정책+full 감사 통과 뒤에만, 새 ready 파일로 생성
python scripts/finalize_source_manifest.py --candidate data/datasets/final_v1.candidate.json --audit runs/<완료full>/report.json --policy <해당candidate정책.json> --dedup-root data/dedup/exact_v1 --output data/datasets/<새ready>.json
```

감사 job의 state.json은5초 이상 간격으로 행 heartbeat를 갱신하고 파일 시작/완료/오류/pause는 pipeline_events.jsonl에 기록한다. CLI exit2는 time/free-space budget pause이며 확인 후 동일 입력 계약으로 재개한다. 원본/코드/정책 변경은 거부한다. 실행 중 audit/source_inventory/data_contracts/code-math fence helper를 변경하면 마지막 code SHA 검사에서 실패하므로 고정한다.

watcher의 logs/source_audit_job.json 등록은 state_file/pid_file/log_file/run_id/total을 지정한다. 등록 후30초 watcher를 기동하며 모든 대상 complete면 정상 종료한다. 다음 job 등록/재기동 시 현재 PID를 확인한다. 외부 push나 대화 종료 후 메시지 발송 기능은 아니다.

F0 전체 CPU 회귀: logs/tests_f0_final_v2.log **81개 통과**. source content/ready 연결의 동작 증거는 runs/f0_binding_validation_v1에 보존. 실제 전체 source의 품질 합격은 별도다.

2026-11-06 실제 실행 갱신: pilot6/6(203.47초/peak259MiB) 통과 후 source_audit_full_v1을21600초 예산·외부timeout21900초로 시작. 예상2~4시간, 실제 progress는 state.json에 기록. wrapper124685/python124686/watcher124687이며 PID는 생존 여부를 매번 확인한다. 입력/감사 dependency는 완료 전 고정한다.

## F1 병행 준비 결과 / 감사 종료 후 실행

아래 명령은 후속 실행 안내다. 현재 full 감사는 실행 중이며 전체 join/100만 측정을 아직 기동하지 않았다. 기존 증거 경로를 덮어쓰지 않고 새 경로를 지정한다.

```bash
# 전체 감사 완료 뒤: 원문 대신 검증된 감사 지문으로 whole/fence 교차 집계
python data/probe_audit_overlap.py --audit runs/source_audit_full_v1/report.json --output runs/<새full_join>/report.json --max-code-rows 3000000 --max-seconds 1800
# pilot을 진단할 때만 --allow-pilot 명시. 결과 full_source_coverage=false
# 보존된 작은 sample_snapshot report에서 후보 누락 완전 비교(대량 원문 읽기 없음)
python data/probe_similarity_recall.py --samples-manifest runs/<고정표본>/report.json --output-dir runs/<새recall> --max-real-per-source 32 --max-view-chars 1024 --max-candidates 20000
# 감사 종료 뒤에만 100만 합성 split 측정. 원본/실제 ready 입력이 아님
python scripts/measure_split.py --output runs/<새scale> --sizes 1000000 --sqlite-cache-mib 64 --max-case-seconds 1800 --max-rss-gib 4 --max-case-gib 8 --min-free-gib 200
```

split runner는 활성 감사 중10만 초과 case를 차단한다. budget pause는 exit2이며 미완료 DB를 보존한다. 같은 코드/입력/size/cache 계약으로 같은 출력에서 재개할 수 있고, budget 변경은 attempt에 남긴다. 완료 결과는 SHA 검증 후 재사용하므로 새 성능 측정으로 보고하지 않는다. 최대300만 설정을 지원하지만100만 결과를 먼저 검토한다. 시간/메모리/디스크는 worker 감시와 종료 검증으로 제한하며 OS disk quota는 아니다. 출력/journal peak는 관측값이다.

완료 증거: parallel_audit_overlap_pilot_v2/report.json(.json.done), parallel_similarity_recall_v1/summary.json(.json.done), parallel_split_scale_ready_v3/summary.json(디렉터리 .done), CPU89개 logs/tests_parallel_p1_p3_final_v3.log. 첫 두 결과는 pilot/작은 표본, split은1,000건뿐이다. 100만 계획은 parallel_split_million_plan_v3.json이며 미실행이다. 공식 benchmark와 별도 GPU 성능 benchmark는 사용자 별도 작업으로 유지하고 F2/F7에 시점을 알린다.

## GPU 합성 benchmark v3 (2026-11-06 사용자 실행 지시)

`python scripts/benchmark_gpu.py --run-dir runs/<새이름>`은 GPU lock을 잡고 bf16 attention/gradient·4096 RoPE 확인 후 tiny,1B/3B compile off/on 및3B micro batch2/4를 순차 실행한다. 각 case900초 상한, seq4096의 global_tokens131072·40steps·첫10steps 제외. 토큰 처리량은 기록 구간의 시간 합계에 해당하는 조화평균으로 비교한다. 전체 모델 bf16 동치나 공식 언어능력 평가 완료를 뜻하지 않는다.

`contract.json`은 code/torch/CUDA/driver/GPU UUID/input hash/설정을, case `.done`은 result/config/log SHA를 묶는다. 이전 빈 marker/변경 결과는 거부하며 새 코드/환경은 새 run을 사용한다. 실패한 기존 run을 덮어쓰지 않는다. state.json 및 pipeline_events에 시작/각 case 완료/실패/최종완료를 기록하고 case 로그로 진행을 확인한다. 실행 중 model/train/benchmark helper는 수정하지 않는다.

현재 실행: runs/gpu_benchmark_v3, logs/gpu_benchmark_v3.log/.pid. 최초v2는 GPU 기동 전 이벤트 경로 타입 오류로 종료했고 수정 회귀 검증 후v3로 실행했다. CPU92개 통과(logs/tests_benchmark_v3.log). 결과와 최적화 판단은 STATUS 및 해당 run의 최종 분석을 따른다.

## compile 수치 대조·단계별 시간 측정

`python scripts/validate_compile_pair.py --output runs/<새이름>`은 GPU lock 아래 micro/1B/3B 단일GPU→3B8GPU를 순차 수행한다. case900초 상한, 기존 경로는 거부한다. 초기 parameter local shard를 CPU에 보존·매번 복원하고 동일 입력/RNG·새 AdamW로 eager 반복/compile을 비교한다. logits/loss/모든 gradient/실제 update 차이를 집계한다. 보고값의 update 상대L2는 가중치 전체 대비 변화율이 아니다.

사전 기준: logits 상대L2 3%, gradient5%, update10%, loss절대0.03. 초과는 needs_review이며 기준을 사후 확대하지 않는다. micro에서 bf16 vs FP32 기준도 보고한다. 본 비교는 seq32의 제한된 합성 한step, AdamW의 동일 uniform weight_decay0.1 비교이며 production의 norm 무감쇠 parameter group·실제seq4096·장기학습 동치를 대신하지 않는다.

3B8GPU에서 별도로 seq4096/batch2/accum2, warmup2+측정3steps의 단계별 CUDA 동기화 wall time을 기록한다. H2D/forward/loss검사/backward/gradient검사/clip/norm검사/optimizer 비용이며 각 구간의 collective를 포함한다. 동기화에 따른 overlap 변화/관측 overhead가 있어 기존 비계측 처리량과 직접 같다고 보지 않는다. trace파일은 생성하지 않는다.

결과 state/summary/rank별JSON과 소스 계약·최종.done은 run 아래 저장한다. needs_review라도 측정 완료 증명은 생성할 수 있으므로 .done만으로 수치 통과를 판단하지 않는다. 실패하면 기존 결과를 보존하고 새 경로에서 수정 검증한다. 메인 감사의 코드/입력은 고정한다.

## 추가 공간 예산과 독립 job 감시

```bash
# 현재source별 receipt/pilot 기반 추정. 새출력만사용
python scripts/storage_budget.py --audit runs/source_audit_full_v1 --pilot runs/source_audit_pilot_v1 --policy runs/storage_budget_v1/policy.json --output runs/<새예산>/report.json
# 실제 producer를 시작하기 직전, 해당작업의 추가peak만 fresh free와 비교. 부족하면exit2
python scripts/storage_budget.py --required-new-gib <추가peak_GiB> --reserve-gib 200
# 소량state/log/pid/report 완료증명으로 독립 job 등록; 기존watcher없으면30초기동
python scripts/register_pipeline_job.py --spec runs/<새job>/monitor_spec.json
```

예산은가정별민감도분석이다. 기존raw/clean을free에다시더하지않고,여러mount출력은별도예산을요구한다. final tokenizer효율/100만split용량은현재미실측. 기존생산명령을자동감싸지는않으므로 allocation check 성공을확인한후producer를별도로실행한다. 보고서과거free를실행시점free로재사용하지않는다. 자료삭제/덮어쓰기를공간확보수단으로자동실행하지않는다.

등록spec 예시는 runs/job_registration_v1/compile_pair_v1.json. 필수run_id/state_file/pid_file/log_file/certificate/report_file/proofs의path+field목록을쓴다. proofs는certificate의SHA필드에report를연결해야한다. benchmark처럼중첩필드는["artifacts","summary.json"],다른진단은["summary_sha256"] 또는["report_sha256"]. requires_review=true는진단측정완료를검토필요로유지한다. 새실행시새run_id를쓰고producer가pid/state/log와atomic완료증명을쓴다. 완료marker만으로validation통과를추정하지않는다.

watcher는 registry를매30초읽고기존source감사와독립job을 pipeline_state/events에함께기록한다. 같은상태반복알림은없고state/run/case전환만이벤트를만든다. error/paused는계속감시,모든대상완료/검토필요등종료상태면정상종료하며새등록은필요시다시기동한다. 외부메시지전송/모델호출없음. 현재main감사PID는유지했고watcher만163709로교체했다.

## 기기별 자원 실행

향후 새 장시간 작업은 [HARDWARE_RESOURCES](HARDWARE_RESOURCES.md)에 따라 `python scripts/hardware_profile.py --output logs/hardware_profile.json`으로 자원을 읽고 `python scripts/run_with_resources.py --stage <cpu-prep|tokenization|sqlite|gpu> -- <실행명령>`으로 thread/로컬temp/컴파일cache를 적용한다. 실제 worker/RSS/disk/GPU개수는 해당producer CLI에서 명시한다. GPU dry-run은 실행승인이 아니다. 현재preset·자식적용검증은resource_policy_validation_v1,result와로그의resource_launches참조.

로컬scratch는profile의storage.scratch_dir로설정된다. synthetic DB는절대출력경로로실측해논리결과를검증했지만생산DB를로컬에만두는전환은아직구현하지않았다. 실제큰DB는원자적영구발행/내용검증/재개·유실대응을검증한뒤사용한다. 완료한로컬측정metadata는NFS에보관하고활성감시에서archive하며,원문/DB를삭제하지않는다.


## F4 로컬 생성 → 영구 발행 (2026-11-06)

`python scripts/run_local_split.py --sources-manifest <ready.json> --output-dir <새 영구 split 디렉터리> --run-dir <영구 실행 기록> --scratch-dir <전용 로컬 디렉터리> --sqlite-cache-mib 131072 --hash-workers 32 --seed 1234 --eval-fraction .01 --local-budget-gib 300 --durable-budget-gib 160 --max-rss-gib 384 --reserve-gib 200 --max-seconds 259200`

위 예산은 실제 전체 split 계획값이다. 실행 직전 hardware_profile 및 여유 공간을 확인하며 실제 실행 여부는 STATUS를 따른다. run/output/scratch는 서로 포함하지 않는 별도 디렉터리로 지정한다. supervisor가 resource launcher를 통해 단일 SQLite writer·Arrow4/OMP2·로컬 temp를 적용하고 2초 간격으로 예산을 감독한다. 사용자 상태 watcher는 별도30초다. 72시간은 attempt 상한이며 예상 완료시간이 아니다.

- 완료: run/.done의 report/contract SHA, run/report.json status=passed, output/manifest.json·.done 및 SplitSelection 검증. output만 보거나 프로세스 종료만으로 전체 완료를 선언하지 않는다.
- 한도 초과: state.json phase=paused/이유, 해당 attempt 결과를 보존한다. 기존 명령을 동일 source/seed/fraction/cache/code로 재실행한다. 시간/RSS/공간 예산은 새 attempt 기록으로 남는다. 코드/입력 변경은 새 run/output을 사용한다.
- 프로세스 종료: 로컬 SQLite의 커밋된 cursor에서 재개한다. 발행 중 .publishing의 닫힌 DB가 남으면 SHA/논리검증 뒤 원자적 rename을 재개한다.
- 로컬 유실: 같은 영구 run/output으로 새 scratch-dir을 지정한다. 검증 가능한 영구 staging/final이 있으면 재사용하고, 그전 단계라면 source에서 다시 생성한다. 미완료 로컬 DB의 원격 중간 백업은 없으므로 그 시점까지의 작업시간을 잃을 수 있다. 영구 source가 필수다.
- 예산은 DB+저널과 양쪽 파일시스템을 따로 계산한다. 대형 단일 transaction의 순간 peak는 2초 관측 사이에 늘 수 있으므로 200GiB 여유를 유지한다. 자동으로 원자료를 삭제하거나 full 학습을 시작하지 않는다.

최신v2는생성검증과닫힌DB의복사SHA동일성을전달하고최종독립consumer검증을수행한다. 동일DB를local/staging각경로에서반복SQLscan하지않는다. --max-rss-gib/--sqlite-cache-mib/--hash-workers를생략하면fresh hardware의sqlite preset을사용하며현재기기probe에서384GiB/131072MiB/32확인. 입력/코드/설정이변경된기존run은다른출력으로실행한다. 실제메인경로는runs/f4_full_v2 및data/datasets/final_v2.split이다.
