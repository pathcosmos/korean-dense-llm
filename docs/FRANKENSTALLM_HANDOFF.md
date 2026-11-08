> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 새 Dense 모델 개발에 적용할 FRANKENSTALLM 참고 지침

2026-11-06T22:45:03.018005+00:00 REF-TRAIN/EVAL/운영: process_accounting/supervise_main_training/test_main_supervision 신규4CPU검사·실제proc메모리계수확인. 최종보고서뒤미승인bundle준비→별도허가후감독실행/인증prefix복구경로구현,현재bundle미생성/3Bmain미실행. 로컬정리만지원하며remoteACK정리는미구현·거부. activeproducer/데이터보존,적용예산·한계·남은actual실증은STATUS·main_supervision_preparation_verified 참조.

2026-11-06T22:31:14.211252+00:00 REF-TRAIN/EVAL/운영: 실제500step1.048576B source/rank/trace·정규평가/저장인증증명추가. assemble_pretrain_review/test_pretrain_review_assembly의2CPU검사/실제부분plot·시각확인후최종증명대기조립기를기동했다. 전체5000/native/3B실제gate유지,조립은최종검토/Goal완료/본학습승인과구별. 적용파일·한계·자원/근거는STATUS·f8_step500_verified·review_assembler_preparation_verified 참조.

2026-11-06T22:18:59.007255+00:00 REF-TRAIN/EVAL/운영: observe_rehearsal/test_rehearsal_observer 새3CPU검사 및실제메타관측기기동. actual단계/평가/저장인증/진척정체 이벤트,재개·평가오버헤드속도구간분리. producer SHA/fsync영수증재사용·source/범주가중NLL메타확인이고최종모델/물리검증대체아님. 적용예산/근거/남은단계는STATUS·rehearsal_observer_verified 참조.

2026-11-06T22:12:12.672000+00:00 REF-TRAIN/EVAL/DATA: monitored3B별도entry와2→4실제검증대기관리기 구현,실제tinyDCP범위/내용및허가/8rank생성3CPU검사통과. actualGPU는F8/최종native뒤대기. 스토리지현재소비자보호/backup검증/복원/ACK후정리·비용대안을읽기전용계획에반영. 기존producer pin/자료보존,적용파일·결과·남은검증은STATUS·monitored_training_preparation_verified·storage_cost_and_retention_proposal 참조.

2026-11-06T21:50:49.518206+00:00 REF-DATA/TRAIN/EVAL: `scripts/evaluate_final_rehearsal.py`/`tests/test_final_rehearsal_evaluation.py` 신규적용. 실제 작은 raw split→local reader→DCP/native 평가 포함4검사 통과, 실제 코드/입력 prereq 확인 뒤 final1B 평가 관리기만 기동했다. 후반 CPU fresh 검증/동일livecontext 재사용→최종5000 인증·GPU 해제 후 기존native를 실행한다. 현재 final평가 미실행·F8/F9미완료. 예산/검증/적용근거는 STATUS·final_evaluation_runner_verified 참조.

2026-11-06 REF-DATA/TRAIN/EVAL 실제 갱신: F6 report/.done·현재 producer·전수726파일·mix 내용증명 검토 완료. `train/rehearsal.py`의 DCP zero-LR optimizer 초기화와 실제 update 구분, `scripts/run_rehearsal.py`의 `--module train.rehearsal` 진입을 적용한 v3에서 실제 full-mix early4/source·생성·ckpt2/4 인증 후 step4 재개·120step 진행을 확인했다. 과거 실패 run/코드/산출물은 보존했다. 현재 producer pin 불변. 전체5000·학습된 최종 native 평가는 미완료다. 다음 평가의 CPU 검증 선행/같은 live context 재사용 계획은 `runs/pretrain_readiness_v1/final_native_eval_lookahead_plan.json`이며 아직 실행 전이다. 적용 파일·실제 근거는 STATUS와 f6_completion_verified/f8_early_and_resume_verified를 따른다.

2026-11-06T21:17:47.745009+00:00 REF-DATA/TRAIN/EVAL: native한영평가 actualtarget663905/705634·6prompt/model 확인(F7완료); fullnative121.17Btrain/1.228Beval·실제mix정책적용(원F6최종검증대기). startupSHA->native출력digest연결3CPU·준입2CPU·mix관리3CPU 및 actual1B/FSDP8 source/gen/journal2update 검증통과. 자동1B리허설 연결은실제증명/자원gate유지. STATUS참조.

2026-11-06T20:17:33.172691+00:00 REF-TRAIN actual1B/3B14case 완료: finaltokenizer/선택batch/lr·재개 tensors657/1020·8rankRNG exact·전역NaN 중단·영구 SHA/fsync 저장 검증. REF-EVAL native 실제평가 실행중,진단4step 품질완료 아님. STATUS·f7_model_validation_verified 참조.

2026-11-06T20:14:10.655969+00:00 REF-TRAIN/EVAL/DATA: 실제 deterministic1B/3B 반복 차이0·선택설정 continuous/resume tensors657/1020 exact(최종 suite증명대기). rehearsal/source-ledger/rank0 EOS broadcast/전체입력 fresh parallel contract 구현·새10CPU 검증(2rank Gloo포함)통과. F8실제/최종PPL은 pending. STATUS·f8_preflight 참조.

2026-11-06T19:35:50.616724+00:00 REF-TRAIN: 원actual batch8case 측정 완료·반복gradient차이 발견. deterministic_train 및 별도 benchmark/validator에 실행flags/code를 resume identity로 연결,2CPU 검사 통과·실제 결정성 비교 기동. 원code pin 보존. exact재개/리허설은 pending. STATUS 참조.

2026-11-06T19:31:57.566167+00:00 REF-EVAL/운영: run_native_evaluation + test_native_evaluation_pipeline의2CPU 검사 통과; 실제 source/split 준비를 CPU에서 기동하고 actual model certificate/GPU lock 뒤 native 한영 PPL/6생성으로 연결. 실행은 진행 중이며 actual PPL/리허설 완료 아님. STATUS 참조.

2026-11-06T19:20:01.731718+00:00 REF-TOK/DATA: 최종6파일 native 파일럿 완료·SHA-bound 용량/실행 검토 후 full24×2 시작. REF-TRAIN/EVAL: 실제 pilot GPU batch 탐색 기동, source probe 구현·2CPU 및 synthetic8GPU1/7/9 직렬동등 검증 통과. 실제1B/3B 검증·리허설은 pending. 적용파일/근거는 STATUS 최상단 참조.

2026-11-06T18:37:48.751155+00:00 REF-TOK/DATA → 실제3GB/raw65536최종후보·독립6144문서검사/효율통과(F5완료),F6fresh로컬검증진입. final-tokenizer48shard4설정동일확인. REF-TRAIN actualbatch선정/peakLR실제검사연결구현·6CPU검사,실제성능GPU는pending. STATUS최상단참조.

2026-11-06T18:13:24.133369+00:00 REF-TRAIN → actual-validator선택geometry·C2accumulation에맞춘fault주입.3CPU검사+실제8GPU micro합성3case에서전rankopt이전중단검증. actual1B/3B최종입력은pending. STATUS참조.

2026-11-06T18:07:27.083363+00:00 REF-DATA/EVAL → prepare_holdout/run의검증된local reader·동일livecontext재사용,새2+기존6=8tests통과/작은실제raw→nativePPL동일·canonical출처보존·변조거부. 대규모실제PPL은checkpoint/최종tokenizer대기. STATUS·runs/f7_preflight_v1 참조.

2026-11-06T17:42:34.892356+00:00 REF-DATA/TOK/TRAIN/EVAL → 실제128M분포SHA검토·pilot_mix진단전용입력·validate_actual_training의실제1B/3BFSDP8준비. 새4CPU검사통과,실제GPU실행pending. NFS구간읽기/readonly전체prefetch실행으로원F4검증보존. 상세STATUS·runs/f7_preflight_v1 참조.

2026-11-06T17:23:00.931433+00:00 REF-DATA/TOK → run_final_tokens의F5실제증명·전체파일pilot/full재사용·storage/실제counts/mix검토연결,집중6tests통과.6worker Rayon2/8 동일512MiB출력실측36.8%시간단축후pilot8적용. gate대기실행기기동,실제F6토큰화는미시작. STATUS·runs/f6_tokens_v1 참조.

2026-11-06T17:06:47.498440+00:00 REF-DATA/TOK → parallel_tokenize의whole-file pilot/local reader/live context/full resume 및 파일별 성능기록,25tests/5.081초 통과. 실제6파일/3,571,498문서 계획은준비상태이며 final tokenizer/실제pilot/full token검증은pending. STATUS최상단 및 runs/f6_preflight_v1 참조.

2026-11-06 16:51 UTC REF-DATA/TOK → local_split_reader/verified_split_context/quality_inputs/run_final_tokenizer,집중10검사통과. 실제3M로컬reader/분포102.643초·논리동일확인,128M읽기전용분포선행준비기동. 실제3GB/BPE는F4최종완료·분포검토뒤진행,최종tokenizer합격미확인. 상세STATUS.

2026-11-06 15:41 UTC REF-DATA/TOK 최종화운영 → watch_split_finalization/test_split_finalization_observer,집중4tests통과·실제30초관측기기동. pinned publisher의실제검증/완료증명과보고서대조·예외활동기록·종료시STATUS기록. 실제F4완료미확인,범위/한계STATUS참조.

2026-11-06 12:36 UTC REF-DATA/TOK 준비·실측완료: verified_split_context/parallel_tokenize/report_split_distribution/benchmark_next_stage/benchmark_bpe_memory,집중6tests+실제512MiB13case토큰동일·BPE16/32/RAM의미동일확인. 24worker×2/batch1024/Arrow1/NFS직접과BPE16스트리밍후보,최종입력pilot채택은pending. STATUS최상단/두run의review참조.

2026-11-06 12:24 UTC REF-DATA/TOK → verified_split_context/parallel_tokenize/report_split_distribution/benchmark_next_stage,새집중6tests통과. 기존순차와v3 bytes동일·손상거부·process crash재개검증,512MiB실험성능campaign기동(미완료). 상세STATUS참조.

2026-11-06 11:49 UTC REF-TRAIN 병행확대 완료: validate_compile_long/parallel_job_support/test_compile_long_metrics, 실제3B 6case·20rank bounded수치기준통과와완료증명검증. 8GPU성능 -0.56%/+0.09%로생산채택없음. 실제token/수렴/재개는후속이며 review.json·STATUS에정량결과/반복오차/한계기록.

2026-11-06 11:27 UTC REF-TOK/REF-TRAIN 병행 확대: benchmark_tokenizer_resources + validate_compile_long 구현, 집중2tests 및 micro1/8GPU smoke 통과. 실제128MiB 실험표본 CPU sweep·3B512/4096 두seed·FSDP8 검증을 새 run에서 기동(결과미완료). 최종train-only tokenizer/생산compile 채택과 구분하며 STATUS 최상단에 적용파일·한계를 기록.

2026-11-06 REF-DATA/TOK 성능적용: 실제SHA32worker/동일실행검증hash재사용·NFC hotpath·발행중복scan제거·대형cache/RSS설정.115회귀+후속집중검사,300만전행논리동일·100만종단165.13초/기존대비2.09배. 실제full_v2(cache128GiB/RSS384GiB/hash32)실행중이며전체완료/전체속도개선보장은아님. STATUS최상단참조.

2026-11-06 REF-DATA/TOK/복구 운영: publish_local_split/run_local_split/test_local_split_publication,112회귀+최종집중4tests 및실제100만강제종료재개/local→NFS발행·전체논리동일검증통과. 실제128,856,258행split기동(완료아님). REF-TRAIN probe_compile_boundaries2GPU독립진단완료,attention경계후보발견/생산채택pending. STATUS최상단참조.

2026-11-06 REF-DATA/TOK 적용: run_f2_checks·sample_snapshot quota보완/test_sample_quota,full감사지문join·100만분할·cache논리동일비교·6000표본실행,전체105tests통과. 정책은범위명시한keep/group/byte결정이며전역near-free보증아님. F3실제원본내용재확인/ready binding은진행중,실제전체split/최종tokenizer는미완료.

2026-11-06 REF-DATA/TOK/TRAIN 후속운영적용: storage_budget·job_registry/register_pipeline_job·watch_data_pipeline/preflight,전체104tests통과. 실제receipt기반추정과관측예산/측정완료·검토상태를분리. final효율/대량DB실측·생산학습검증은pending. STATUS최상단참조.

2026-11-06 REF-TRAIN 정밀도분해: diagnose_compile_precision/probe_swiglu_rounding/probe_full_fp32_gate/probe_compile_casts/probe_eager_norm_casts,새tests4개통과. FP32/bf16·층별/optimizer실제진단과후보비교완료,단일GPUseq32한계·잔여수치검토·production채택pending은STATUS에기록.

2026-11-06 REF-TRAIN 추가 적용: validate_compile_pair·test_compile_pair_metrics2개 및 실제4case 측정. 3B8GPU 출력/gradient/update 차이가 사전기준을 넘어 needs_review, production compile 승인pending. REF-DATA/TOK 근거는 f2_policy_preparation_v1에서 현재candidate에 연결했으나5개정책조건pending. 적용/검증 한계는 STATUS 참조.

2026-11-06 REF-TRAIN 성능 적용: benchmark_gpu/benchmark_contract/check_gpu_numerics, tests/test_benchmark_contract. 실제H2008장 합성7case 및92tests 통과, compile/배치 최적화 후보 실측. 전체3B eager/compile 학습 loss 차이가 남아 동일 가중치의 logits/gradient/update 대조와 실제 입력 검증 pending. 성능 성공을 학습 수치/품질 완료로 간주하지 않는다. STATUS 최상단 참조.

2026-11-06 병행 적용 갱신: REF-DATA D03/그룹 분리·내용 식별을 probe_audit_overlap/probe_similarity_recall/measure_split/split_manifest와 새 테스트3개에 적용. 실제 pilot 지문 join(전체 coverage=false), 실제96+합성40의7,168쌍 대조, 합성1,000건 split 검증 및 CPU89개 통과. full726 감사/실제100만 측정/실제 품질 통과는 아직 pending. benchmark는 사용자 별도 작업으로 F2/F7에 시점을 알린다. 상세 실제 수치/남은 항목은 STATUS 최상단 참조.

2026-11-06 F0/F1 실제 적용 갱신: REF-DATA/TOK의 파일 내용 식별·원문 코드 보존·검증된 source 고정 조건을 source_audit/source_inventory/quality_report/finalizer에 연결했다. 후보→audited-ready는 tiny fixture로 검증, CPU81개 통과. 실제6파일 pilot은 통과했고726파일 full 감사는 진행 중이다. fixture/과거 acceptance를 실제 전체 감사 완료로 표시하지 않는다. 현재 model/train 가중치·학습 설정을 바꾼 작업이 아니다.



## 2026-11-06 중복 제거 완료 후 재대조 — 선행 LLM 반영 지점

최신 후속 계획은 WORKFLOW 최상단 「현재 다음 실행 계획 — 정확중복 제거 완료 이후」 F0~F10이다. 아래는 참고자료를 현 구현/실제 적용과 대조한 결과다. 신규 단계의 완료 보고가 아니다.

| REF / 과거 위험·활용점 | 현재 반영 증거 | 다음 반영/실제 확인 시점 |
|---|---|---|
| REF-DATA D01/D02: 단일 window/마지막 시작점 및 다음 token 관계 | train/data.py + tests/test_sampler.py가 acceptance 입력을 실제 함수에 연결. stateless batch/버전·재개 계약 유지 | F4 실제 그룹 교차0/row coverage, F6 실제 source token 수, F7 실제 sampler+checkpoint 경로 |
| REF-DATA D03: 웹 정규화를 코드에 적용해 의미가 바뀜 | code clean raw UTF8·indent/case, tests/test_code_clean.py. N5 raw tokenizer 결정·N10 train-only/eval-only 연결 | F1 code body SHA/provenance, F2 코드↔웹 범위·품질 결정. 코드 부분 일치로 웹 문서 전체 자동 삭제하지 않음 |
| REF-TOK T01/T02: uint16 overflow와 같은 vocab 캐시 재사용 | cast 전 범위/실제 EOS/내용 hash, tests/test_tokenization.py·test_tokenizer_check.py·test_split_inputs.py | F0 audit→ready→split 내용 ID 연결, F5 실제65536 raw 후보와 독립 검사, F6 실제 v3 tokens |
| REF-TRAIN: checkpoint/RNG/데이터 계약·전체 rank 오류 순서 | train/runtime.py/train.py, current micro 8GPU compile 6case summary. 현재 train/model/data/runtime/harness hash와 일치 | F7 실제 token 입력·1B/3B 크기·메모리/성능·재개 검증. 과거 DDP/FP8/NCCL·속도 그대로 승계하지 않음 |
| REF-EVAL E01/E02/E03: tail 중복/누락·평균의 평균·0-token 성공 | tests/test_evaluation.py가 사양을 직접 사용, native loader/ko-en JSON/고정 생성·N10 증명, local choice scorer | F2 공식 task 원본/revision/오염검사 범위 고정, F7 실제 ko/en holdout·native PPL/생성·공식 adapter/필수 task 상태 |
| REF-EXPORT X01: 과거 attention 누락 및 head/config 고정 | native checkpoint loader의 missing key/shape 거부는 tests/test_eval_runner.py에서 검증. HF exporter 자체는 미구현 | 필요 시 별도 구현. 현재3B는 untied이므로 out.weight를 보존, embedding을 head로 복사하지 않음. 필수 Q/K/V·eps/tie/RoPE/vocab 확인 및 tiny tied/untied 전후 logits 검사 |
| REF-SFT S01/S02/T02/E02: system 누락·mask -1·중복 shift·잘림/EOS·캐시 충돌 | 인계 사양은 있으나 현재 sft/와 새 구현 통과 증거 없음 | F10 assistant-only, 현재 CE의 -100, shift1회·EOS/end marker, system/다중 턴 유지, tokenizer/template/input 내용 캐시·valid-token 가중 집계. 단답/빈 응답/잘림/code/JSON 검증 |
| REF-PREF P01: winner 누락을 A로 추정 | 선호학습 미도입, 감사 도구는 새 구현 통과가 아님 | 별도 도입 결정 때 unknown/tie/missing 제외, prompt 그룹 분리와 라벨/출처 확인. Base/SFT 선행 조건 아님 |

원본 재확인:
- Git G: .cache/references/FRANKENSTALLM의 HEAD `50c6143629203815dfe7775e3b7c0dd0a3260324` 확인. convert_to_hf.py의 embedding→lm_head 복사/tie=true/norm eps 고정 부분 확인.
- HF H: revision `b9f7777f01479b18e5a3a55f9cad4582df122f7d`. data/dataset.py, data/sft_dataset.py, source/scripts/convert_to_hf.py의 로컬 SHA256이 고정 download manifest와 일치함을 확인했다. 새로 전체 모델 가중치를 받거나 품질 재평가한 것은 아니다.
- 현 구현 증거는 logs/tests_n9_n11_final_v3.log의 CPU74개와 runs/n2_compile_v1/summary.json의 current compile suite다. 이 turn에서 테스트를 재실행한 것은 아니며 해당 결과와 소스 hash를 대조했다. old evidence.json의 current 파일 hash는 이전 감사 시점 snapshot이다.

기존 모델/토크나이저 가중치를 새 모델에 가져오거나 이전 평가 점수를 새 합격 기준으로 쓰지 않는다. 현 모델은3.22B·vocab65536·raw byte-level BPE·untied3B·H200/FSDP2/bf16 설계를 유지한다. 과거 모델 복구는 비교 연구가 필요할 때 별도로 진행한다.


이 자료의 목적은 **현재 프로젝트에서 새 모델을 구현·학습할 때 검증된 동작을 활용하고, 과거 실패를 반복하지 않도록 하는 것**이다. 현재 작업자가 이 문서에서 해당 단계의 항목을 골라 코드와 테스트에 반영한다. 과거 모델 복구·재평가는 과거 성능과 직접 비교할 경우에만 별도로 진행하며 새 모델 개발의 선행 조건으로 두지 않는다.

## 작업자가 참고하는 순서

1. [STATUS](STATUS.md)에서 현재 단계와 진행 중인 작업을 확인하고 [PLAN](PLAN.md)·[DECISIONS](DECISIONS.md)의 확정 설계를 따른다.
2. 아래 표에서 이번 변경과 관련된 **REF 항목만** 선택한다. 해당 입력·출력 규칙과 완료 조건을 작업 계획에 적는다.
3. 구현을 확인할 필요가 있을 때 원본 파일을 읽는다. 상세 발견 사항은 [검토 보고서](FRANKENSTALLM_REVIEW.md), 사실 확인은 [증거 JSON](reviews/frankenstallm/evidence.json)을 조회한다.
4. 현재 코드에 맞춰 구현하고 [검증 입력](reviews/frankenstallm/acceptance_cases.json)을 실제 테스트에 연결한다. 이미 구현된 부분은 확인 후 재사용한다.
5. 작업 종료 시 STATUS에 `REF ID → 수정 파일 → 실행한 테스트·결과 → 남은 항목`을 남긴다. 적용하지 않는 항목은 이유를 적는다.

현재 3B Dense·순수 PyTorch·FSDP2·bf16·65536 byte-level BPE·3B untied embedding 설계를 기준으로 한다. 과거 DDP·MXFP8·SentencePiece 설정이나 가중치를 그대로 가져오지 않는다. 모델 설정을 바꿀 필요가 생기면 기존 결정 문서에 근거를 남긴다.

## 단계별 적용 항목

아래 파일 중 아직 없는 목적지는 구현 제안이다. **이 표와 JSON은 구현 요구사항이며, 새 모델 코드가 이미 통과했다는 기록이 아니다.**

| REF ID / 단계 | 새 모델에서 적용할 위치 | 참고해서 구현할 동작 | 검증 입력 또는 완료 조건 |
|---|---|---|---|
| **REF-DATA / 2·4** | `data/filters.py`, 중복제거 도구, `train/data.py` | 웹·코드의 정규화 분리. 코드는 대소문자·들여쓰기 보존. 유효 window 시작점은 `0..n-(seq_len+1)` 포함. 소스·샤드 간 중복제거 뒤 holdout 분리 | D01~D03. train/eval 문서 교집합 및 실제 소스별 토큰량 기록 |
| **REF-TOK / 3·4** | `tokenizer/`, `data/tokenize_shards.py`, 캐시·manifest | tokenizer 내용 해시·입력 버전·변환 설정으로 산출물 식별. uint16 저장 전 ID 범위 검사. 특수토큰 ID는 tokenizer에서 읽기 | T01~T02. 명시적 정규화 정책 roundtrip(2026-11-06 N5: 새 후보 raw, NFC는 비교), 한/영/코드/수학·개행·Unicode 사례 |
| **REF-TRAIN / 5·6** | `train/train.py`, checkpoint metadata | 완전한 checkpoint만 재개. 실제 batch·tokenizer/mix 해시·설정 기록. FSDP 전체 랭크의 평가·오류 종료 순서 일치 | 동일 seed 연속/재개 비교, 불균등 holdout, 비유한 loss/gradient의 optimizer 이전 종료. 기존 WORKFLOW 단계 5 검증 사용 |
| **REF-EVAL / 7 준비·10** | 새 `eval/`, 학습 중 holdout 평가 | NLL 합과 유효 표적 수로 집계. context overlap과 scoring mask 구분. 표적 중복/누락 방지. Base/Chat 프롬프트 구분. seed·길이·checkpoint·task 버전 기록 | E01~E03. 한/영 별도 결과, 필수 task 누락을 성공으로 처리하지 않기 |
| **REF-EXPORT / 7·10의 변환 시점** | HF 변환기, 모델 로더 | Q/K/V 등 필수 키·shape 검사. 현재 별도 LM head와 norm eps 보존. missing weights가 있으면 점수 산출 전에 중단 | X01. 작은 tied/untied 모델 변환 전후 logits 비교. CPU FP32 시작 기준 `atol=1e-5, rtol=1e-4`, bf16은 별도 기준 |
| **REF-SFT / 10** | 새 `sft/data.py`, 학습·추론 공통 대화 형식 | 응답만 loss 계산, shift는 한 번, EOS 학습. system·다중 턴 보존. tokenizer/템플릿 해시로 캐시 구분. 길이별 유효 토큰 가중 집계 | S01~S02, T02. 빈 응답·잘린 응답·정상 단답·코드·JSON·특수토큰 리터럴 추가 검사 |
| **REF-PREF / 필요성 판단 후** | 선호도 정규화·분할 도구 | 명시적 winner만 수용. 누락·동점·알 수 없는 값은 제외 사유 기록. source/row ID 유지, prompt 그룹 단위 분할 | P01. 선택한 라이브러리·버전의 빈 응답/NaN 검증. ORPO 도입 자체는 아직 결정하지 않음 |

적용 순서는 현재 WORKFLOW의 데이터 준비·검증 일정에 따른다. 예를 들어 정제 작업 중이면 REF-DATA부터 적용하고, 평가 코드를 준비하는 작업이면 REF-EVAL·REF-EXPORT를 적용한다. 이 자료의 표 순서가 기존 진행 중인 작업을 중단하거나 전체 일정을 바꾸는 지시가 되지는 않는다.

## 원본에서 볼 부분과 가져오지 않을 부분

원본은 [G 고정 커밋](https://github.com/pathcosmos/FRANKENSTALLM/tree/50c6143629203815dfe7775e3b7c0dd0a3260324)과 [H 고정 revision](https://huggingface.co/pathcosmos/frankenstallm/tree/b9f7777f01479b18e5a3a55f9cad4582df122f7d)이다. 코드가 필요하면 이 버전으로 확인한다.

| 작업 | 읽을 원본 | 활용할 부분 | 새 코드에서 바꿀 부분 |
|---|---|---|---|
| 데이터 로딩 | H `data/dataset.py` | uint16 memmap과 입력/다음 토큰 관계 | 현재 stateless sampler 유지, 경계·입력 검증 보완 |
| 토큰 파일 구성 | H `data/merge_bins.py`, G `scripts/prepare_3b_data.sh` | 토큰 수 집계·문서 EOS의 목적 | 현재 샤드 스트리밍·원자적 저장·버전 검증으로 구현 |
| 모델 연산 | G `model/attention.py`, `model/layers.py` | GQA/RoPE·SwiGLU 구조와 검증 사례 | 현재 모델의 API·초기화·tie·eps 유지. 단일 RoPE 함수 일치를 전체 모델 검증으로 확대하지 않기 |
| 평가 | G `eval/tasks/` | PPL·생성·벤치마크로 나누는 구성, 반복/EOS 지표 | 과거 PPL coverage 오류 수정, GPU/경로/토큰 ID 상수 제거 |
| SFT | H `data/sft_dataset.py`, `data/filter_sft_v2.py` | 응답 label 정렬·잘림 후 EOS·제거 사유 기록 | 과거 `-1` mask를 현재 계약에 맞춤, system 누락·캐시 충돌 방지, 50자 미만 일괄 제거 재검토 |
| 선호도 | H `data/prepare_preference_combined.py` | 소스별 포맷 정규화 | winner 누락 시 A 기본 선택 제거, 출처와 분할 근거 보존 |
| 변환 | G `scripts/convert_to_hf.py` | QKV 분리·가중치 이름 대응 구조 | embedding→head 복사 금지(현재 3B는 untied), eps/config 고정값 제거, 엄격한 키 검사 |

`.cache/references/`의 사본은 편의용이다. 캐시가 없어도 이 문서와 JSON의 입력·기대 결과로 새 구현을 진행할 수 있다. 원본을 다시 받을 때는 [증거 JSON](reviews/frankenstallm/evidence.json)의 고정 revision·파일 SHA256을 사용한다.

다음 항목은 이번 모델의 기본 선택으로 승계하지 않는다.

- 과거 checkpoint나 과거 성능 수치: 가중치 누락 경고·모델 크기 혼재가 있어 새 모델의 합격 기준이나 개선율 계산에 바로 쓰지 않는다.
- 과거 256-token byte-fallback 추가 패치: 현재 65536 BPE·uint16 계약과 다르다.
- B200 전용 FP8·DDP·NCCL 설정과 과거 속도: 현재 H200·FSDP2에서 별도 측정해야 한다.
- ORPO·Mamba·외부 알림·업로드: 현재 목표에 필요한 별도 요청·판단이 있을 때만 검토한다.

## 테스트로 연결하는 방법

[acceptance_cases.json](reviews/frankenstallm/acceptance_cases.json)은 **12개 구체 입력과 기대 결과**를 담는다. 구현자는 관련 사례를 실제 함수에 연결한 테스트로 작성한다. JSON의 숫자 token ID는 합성 테스트용이며 실제 tokenizer ID를 지정하지 않는다.

예: `train/data.py` 수정에서는 D01의 5개 토큰으로 길이 4인 입력/정답 한 쌍이 생성되는지 확인하고, D02의 가능한 마지막 시작점을 경계 검사로 확인한다. 무작위 추출 여러 번에서 두 시작점이 모두 나왔다는 것만으로 완료하지 않는다. sampler 규칙 변경 시 기존 checkpoint의 재개 호환성도 기록한다.

예: `eval/` 구현에서는 E01의 각 표적이 한 번씩만 채점되고 E02의 NLL 평균이 3.5인지 검사한다. tokenizer와 학습 가중치가 없어도 작은 합성 입력으로 먼저 검증할 수 있다.

`scripts/audit_frankenstallm.py`는 **과거 결함이 재현되는지 검사하는 감사 도구**다. 결함이 수정된 새 코드의 회귀 테스트로 그대로 사용하지 않는다. 감사 스냅샷은 보존하고, 새 구현의 성공 여부는 이 인계 자료의 기대 결과와 실제 테스트로 판단한다.

## 적용 기록과 완료 판단

작업 종료 시 STATUS에 다음 형태로 기록한다.

```text
참고 항목: REF-EVAL
적용 파일: eval/<실제 구현 파일>
검증: E01/E02/E03을 연결한 테스트 명령과 결과
상태: 구현 중 / 구현 완료·검증 대기 / 적용 검증 완료 / 적용하지 않음(이유)
남은 검증: 실제 holdout·8GPU 등 이번 검사에 포함되지 않은 항목
```

문서 전달 완료와 코드 적용 완료는 별개다. 이 자료를 읽었다는 이유만으로 해당 REF 항목을 완료 처리하지 않는다. 현재 코드에 이미 대응 기능이 있으면 새로 복사하지 않고 검증 근거를 남긴다.

## 새 작업 세션에 전달할 요청문

```text
현재 프로젝트의 CLAUDE.md, docs/STATUS.md, docs/WORKFLOW.md를 읽고 현재 작업을 이어가라.
docs/FRANKENSTALLM_HANDOFF.md에서 현재 단계에 해당하는 REF 항목을 적용하라.
목표는 새 3B Dense 모델 개발이며 과거 FRANKENSTALLM 복구가 선행 조건은 아니다.
원본 구현이 필요할 때만 고정 revision의 관련 파일과 검토 보고서를 확인하라.
acceptance_cases.json의 해당 입력을 실제 테스트에 연결하고 파일·결과·미완료 항목을 STATUS에 기록하라.
본학습 시작·기존 데이터 삭제/덮어쓰기·외부 업로드는 기존 확인 규칙을 따른다.
```

## 현재 코드 대조와 실제 적용 순서 (2026-11-05 15:53 UTC)

이 점검은 참고자료를 현재 구현과 대조한 결과다. 아래 미완료 항목을 완료로 해석하지 않는다.

| 시점 | REF / 관련 사례 | 현재 확인 | 다음 적용과 완료 증거 |
|---|---|---|---|
| 현재 정제·중복제거 | REF-DATA / D03 | `data/dedup.py`는 최종 텍스트 NFC SHA256으로 공백·대소문자 보존. D03의 두 코드 쌍을 실제 fingerprint에 넣어 `[false,false]` 확인. 기존 `data/filters.py exact_hash`의 웹 규칙은 그대로임 | dedup 전체 완료·소스별 제거량 검증. 코드 소스 추가 전 웹 정제의 norm_ws/exact_hash를 그대로 쓰지 않는 경로 준비. D03의 JSON을 지속 회귀 테스트에 직접 연결하는 작업은 별도 |
| 토큰화·실제 학습 데이터 로딩 전 | REF-DATA / D01,D02 | `train/data.py`의 `rng.integers(0,n-T)`가 아직 남아 단일 window 및 마지막 시작점 문제가 미수정 | 유효 window 수 `n-T+1`, 짧은/빈 샤드 검증, 제어된 RNG 경계 테스트. sampler 버전 기록·이전 checkpoint 재개 호환성 확인 |
| 최종 tokenizer·토큰화 전 | REF-TOK / T01,T02 | `tokenize_shards.py`는 ID 검사 없이 uint16 cast, .done은 token 수만 기록. tokenizer/train_tokenizer.py는 아직 원본 clean 경로 사용 | 검증된 dedup 경로 연결, tokenizer 내용 해시·입력 버전·EOS ID·설정 manifest, cast 이전 0..65535 검사, 캐시 충돌 테스트. 최종 holdout은 중복제거 후 문서 단위 분리 |
| 1B 리허설 전, 데이터 처리와 독립 준비 가능 | REF-EVAL / E01~E03, REF-TRAIN | `eval/` 미작성. 학습 코드의 재개·평가 동작은 기존 단계 5의 남은 검증과 연계 | 정확한 표적 coverage·tail·0-token 오류·토큰 가중 NLL, 한/영 분리 및 고정 생성 평가. 연속/재개 비교·비유한 gradient 사전 종료·불균등 랭크 평가 검증 |
| 평가에 HF 변환본을 처음 사용할 때 | REF-EXPORT / X01 | export 검증 경로 미작성. 현재 3B는 untied head | 필수 Q/K/V 키·shape 및 별도 head/norm eps 보존, 누락 시 평가 중단, tiny tied/untied 변환 전후 logits 비교. 추론 export 준비 때문에 현재 사전학습 코드 검증을 막지 않음 |
| Base 모델 평가 후 SFT 단계 | REF-SFT / S01,S02,T02 | `sft/` 미작성 | system/다중 턴, 응답만 학습·shift 1회·EOS·ignore_index=-100 계약, tokenizer/template 해시별 캐시, 빈 응답·단답·코드/JSON 테스트 |
| SFT 평가 후 필요성을 판단한 경우만 | REF-PREF / P01 | 미도입 | winner 없는 데이터 거부, prompt 그룹 분리·출처 보존, 빈 응답/NaN 검증. ORPO를 자동 도입하지 않음 |

활용 방법은 `관련 REF 선택 → 필요한 원본 함수/실패 사례만 확인 → 현재 API에 맞게 구현 → acceptance_cases.json을 실제 함수 테스트에 연결 → STATUS에 적용 파일·명령·결과 기록`이다. `scripts/audit_frankenstallm.py`는 과거 결함 감사용이며 새 코드의 합격 테스트를 대신하지 않는다.

즉시 우선순위는 **현재 정제/중복제거 유지 → sampler 경계 및 토큰화 계약 보완 → 최소 평가 준비 → 실제 데이터 8GPU 검증·리허설**이다. 데이터 변경은 기존 Sol / High 흐름, 복잡한 학습/평가 검증은 기존 Astra / High 권장 흐름으로 연결한다.

과거 점수·checkpoint는 새 모델의 목표치로 직접 사용하지 않는다. 과거 Base 변환본 일부의 필수 Q/K/V 누락이 확인됐으므로 과거 모델과 수치 비교를 원할 때만 원 checkpoint 확보·엄격한 변환 검증·같은 평가 조건 재측정을 별도 수행한다. 현재 새 모델의 데이터/학습 진행을 위한 선행 조건은 아니다.
