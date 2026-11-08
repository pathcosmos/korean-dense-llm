> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# FRANKENSTALLM 재사용 검토와 개선 작업

**새 모델 구현 작업자는 [인계 지침](FRANKENSTALLM_HANDOFF.md)을 먼저 읽는다.** 이 보고서는 상세 근거를 찾는 자료다. 아래 과거 checkpoint 복구·재평가 항목은 옛 성능과 직접 비교할 때 필요하며 새 모델 개발을 막는 선행 조건이 아니다.

현재 한국어 3B Dense 프로젝트에 가져올 코드, 수정할 동작, 추가 확보할 실행 자료를 정리한다. **평가의 신뢰성 확보를 먼저 하고, 데이터·SFT의 검증된 동작을 현재 인터페이스에 맞춰 구현한다.** 기존 순수 PyTorch·FSDP2·65536 BPE 설계는 유지한다.

GitHub에서 빠졌던 데이터 처리 소스는 Hugging Face에서 확보했다. 다만 보관된 Base 변환본 두 개에는 Q/K/V 가중치 84개가 없으며, 과거 평가 로그에도 해당 가중치의 누락·재초기화 경고가 있다. 따라서 과거 점수를 이번 모델의 기준점으로 사용하려면 평가별 체크포인트 식별과 재평가가 필요하다.

## 범위와 근거

- **G**: [GitHub 고정 커밋](https://github.com/pathcosmos/FRANKENSTALLM/tree/50c6143629203815dfe7775e3b7c0dd0a3260324), `50c6143629203815dfe7775e3b7c0dd0a3260324`.
- **H**: [Hugging Face 고정 revision](https://huggingface.co/pathcosmos/frankenstallm/tree/b9f7777f01479b18e5a3a55f9cad4582df122f7d), `b9f7777f01479b18e5a3a55f9cad4582df122f7d`.
- **C**: 현재 작업 트리. Git HEAD만으로 미커밋 변경을 식별할 수 없어 검토한 파일의 SHA256을 함께 기록했다.
- [기계 판독 증거](reviews/frankenstallm/evidence.json): 원격 목록, 확보 파일 해시, 가중치 헤더, 평가 경고·실행 설정, CPU 재현 결과. 실행 시각은 `created_utc`에 있다.
- [재현 스크립트](../scripts/audit_frankenstallm.py): 검토한 정의만 추출하여 CPU에서 확인한다. 기존 학습·배포·알림 진입점은 실행하지 않는다.

근거 수준은 **문서 기록 / 코드 확인 / 과거 실행 기록 / 이번 재현**으로 구분한다. 소스가 존재한다는 사실은 해당 버전으로 과거 학습이 실행됐다는 증거가 아니다. 전체 가중치·학습 데이터의 내용 검증과 모델 품질 재평가는 이번 범위에 포함하지 않았다.

## 확보한 자료와 남은 자료

GitHub의 현재 공개 브랜치 3개에서 도달 가능한 커밋 38개를 확인했다. 이 범위에서 `data/`와 `scripts/trl_patch.py`의 변경 이력은 없었다. 삭제된 브랜치·도달 불가능한 커밋까지 없다는 뜻은 아니다.

H의 파일 516개를 목록화하고 필요한 소스·설정·토크나이저·평가 기록 **201개, 13,567,977 bytes**를 별도 경로에 확보했다. 다운로드 파일에 SHA256을 기록하고 원격 크기와 대조했다. LFS SHA256이 제공되는 확보 파일은 그 값도 대조했다. 모델 가중치 네 파일은 HTTP Range로 헤더만 읽었다.

| 자산 | 확보 상태 | 활용과 남은 확인 |
|---|---|---|
| `H:data/dataset.py`, `sft_dataset.py`, `merge_bins.py` | 소스 확보 | GitHub 누락을 보완. 현재 API에 그대로 연결하지 않음 |
| `H:data/filter_sft_v2.py`, `prepare_sft_data.py`, `prepare_preference_combined.py` | 소스 확보 | 필터·정규화·응답 마스킹 검토 가능 |
| `H:source/` 설정·평가 로그·결과 JSON | 선택 확보 | 평가별 모델 크기와 누락 경고 식별 가능 |
| `H:model.safetensors`, `config.json`, `tokenizer.json` | 설정·토크나이저 및 가중치 헤더 확보 | 현재 배포본의 구조 확인. 가중치 본문은 미다운로드 |
| `H:source/eval/outputs/` Base·SFT HF 변환본 | 원격 존재 및 헤더 확보 | Base 두 사본은 필수 Q/K/V 누락. SFT 사본은 헤더 검사 통과 |
| `H:data/sft_combined/train_filtered.jsonl` | 원격 목록·크기·LFS 해시 확보 | 8,032,330,047 bytes. 본문 품질·중복·분할은 미검증 |
| `H:data/sft_combined/val_filtered.jsonl` | 동일 | 163,855,570 bytes. 학습과의 내용 교집합 확인 필요 |
| `H:data/preference/combined_preference.jsonl` | 동일 | 2,728,221,982 bytes. 선호 라벨·프롬프트 단위 분할 검증 필요 |
| 원래 Base/SFT/ORPO 학습 체크포인트 | 미확보 | HF 변환 이전 가중치와 optimizer·scheduler·RNG·실제 step 필요 |
| 과거 사전학습 `.bin`과 데이터 manifest | 미확보 | tokenizer 해시·소스 revision·고유 토큰량·중복제거 이력 필요 |
| 실제 TRL 패치와 고정 환경 | 미확보 | README의 `scripts/trl_patch.py`는 G/H 목록에 없음. 패치 diff·적용 버전 필요 |

참고 사본은 `.cache/references/FRANKENSTALLM/`, `.cache/references/frankenstallm_hf/<revision>/`에 있다. 현재 `data/raw`, `data/clean`, `data/tokens`와 분리되어 있다. 대용량 파일의 LFS 해시는 원격 선언값이며 본문을 내려받아 무결성을 검증했다는 뜻은 아니다.

## 모델과 실행 버전 구분

| 항목 | 관찰된 구성 | 판정 |
|---|---|---|
| `G:configs/3b_pretrain.yaml` | 32층, dim2560, vocab64000 | 별도 설계 후보. 최종 28층 모델과 구분 |
| `G:configs/korean_3b_fp8.yaml` | 28층, dim3072, vocab64256 | 현재 Git 설정. 과거 실행 당시 설정과 동일하다고 가정하지 않음 |
| `H:source/configs/korean_3b_fp8.yaml` | 28층, dim3072, vocab64000 | HF 소스 스냅샷. Git 설정과 차이 있음 |
| `H:model.safetensors` | 28층, dim3072, vocab64256, BF16, 저장 파라미터 3,016,141,824 | 필수 키·shape 검사 통과. tied embedding으로 head 별도 저장 없음 |
| `H:source/eval/outputs/hf_3b_base/model.safetensors` | config는 28층·dim3072·vocab64000, F32, 저장 원소 2,771,561,472 | Q/K/V 84개 누락. 정상 Base로 사용 보류 |
| `H:source/eval/outputs/3b_full_eval_20260305_0318/hf_3b_checkpoint-0057000/model.safetensors` | 위와 같은 누락 형태 | 이 사본도 사용 보류 |
| `H:source/eval/outputs/hf_3b_sft_best/model.safetensors` | 28층·dim3072·vocab64000, F32, 저장 원소 3,211,963,392 | 키·shape 통과. embedding/head 중복 저장, 실값 일치 미검증 |
| `H:source/eval/outputs/quick_sft_v2_best/.../results_2026-02-27*.json` | 평가 메타데이터의 파라미터 수 1,189,709,824 | 3B 평가와 별도 분류 |
| `H:source/eval/outputs/3b_base_quick/.../results_2026-03-05*.json` | 평가 메타데이터의 파라미터 수 3,015,355,392 | 실행 당시 로드된 모델 크기 기록. 온전한 가중치 로드를 증명하지 않음 |
| `C:model/model.py`, 3b preset | 28층, dim3072, vocab65536, untied, 3,221,400,576 parameters | meta device로 파라미터 수 확인 |

SFT 파일의 저장 원소 수는 별도 저장된 embedding/head를 포함한다. tied 모델의 고유 파라미터 수와 직접 비교하지 않는다. 현재 배포본의 헤더 검사는 구조 확인이며 전체 파일의 값·생성 품질 검증은 아니다.

다운로드한 `H:source/` 중 G에 대응 파일이 있는 68개를 바이트 비교했다. 62개 동일, 6개 상이: `configs/hybrid_3b.yaml`, `configs/korean_3b_fp8.yaml`, `configs/korean_3b_sft.yaml`, `configs/korean_3b_sft_v2.yaml`, `scripts/upload_to_huggingface.py`, `tokenizer/tokenizer_config.json`. 이 범위 밖 파일의 동일성은 판정하지 않았다.

현재 Git 사전학습 설정을 8GPU에서 그대로 적용한다고 계산하면 `57000 × 5 × 8 × 8 × 4096 = 74,711,040,000 tokens`이다. 과거 README의 약 38.5B와 일치하지 않는다. 실행 중 batch 변경·재개 여부를 포함한 원 로그 없이 실제 과거 학습량을 이 산식으로 확정하지 않는다.

## 우선 해결할 문제

### 과거 평가의 가중치 누락

Base 변환본 두 사본 모두 28개 층의 `q_proj`, `k_proj`, `v_proj`가 빠져 있다. 파일당 **84개 필수 텐서 누락**을 헤더에서 확인했다. 확보한 평가 로그 **46개 파일**에서도 attention 가중치 `MISSING` 경고를 확인했다. 이 개수는 독립 실험 46회의 뜻이 아니다.

일부 로그는 누락 가중치를 새로 초기화했다고 명시한다. 예를 들어 [Base 평가 로그](https://huggingface.co/pathcosmos/frankenstallm/blob/b9f7777f01479b18e5a3a55f9cad4582df122f7d/source/eval/outputs/3b_full_eval_20260305_0318/phase2_gpu3_0shot.log)와 [SFT 평가 로그](https://huggingface.co/pathcosmos/frankenstallm/blob/b9f7777f01479b18e5a3a55f9cad4582df122f7d/source/eval/outputs/3b_sft_eval_20260306_1536/phase2_gpu1_0shot.log)가 해당한다. **그 경고가 연결된 평가 결과는 정상 체크포인트의 성능 기준으로 사용하지 않는다.** 모든 보고서의 모든 수치가 잘못됐다고 일괄 판정하지는 않는다.

조치: 원래 학습 체크포인트 확보 → 필수 키·shape와 실제 parameter count 검사 → 변환 전후 logits 비교 → checkpoint 해시를 남겨 재평가한다. 현재 G의 변환 코드는 fused QKV 분기를 갖추고 있지만, 과거 누락이 어느 버전·실행에서 생겼는지는 별도 추적 대상이다.

### PPL 평가 범위의 중복과 누락

`G:eval/tasks/ppl_task.py`의 `SlidingWindowDataset`을 그대로 추출하여 13개 토큰, seq_len=8, stride=2로 실행했다. 다음 토큰 표적은 12개인데 **17회 평가, 고유 표적 11개, 중복 평가 6회, 마지막 표적 누락**이 발생했다. 길이가 seq_len과 같을 때는 window가 0개다. 해당 평가 함수는 분모가 0이면 NLL=0으로 두어 PPL=1을 만들 수 있다. [평가 코드](https://github.com/pathcosmos/FRANKENSTALLM/blob/50c6143629203815dfe7775e3b7c0dd0a3260324/eval/tasks/ppl_task.py#L39)

조치: 전체 목표 토큰 중 각 표적을 정확히 한 번 점수화하고, context overlap과 scoring mask를 별도로 정의한다. 유효 토큰 0개는 오류로 처리한다. 한/영 NLL 합계와 유효 토큰 수를 따로 기록한다. tokenizer가 다른 모델끼리 token PPL을 바로 비교하지 않고 공통 원문 기준 bits per byte 같은 보조 지표를 준비한다.

### SFT 캐시와 대화 형식

복원한 SFTDataset의 캐시 키는 파일 size/mtime, seq_len, vocab size만 사용한다. 같은 크기의 서로 다른 tokenizer에서 같은 문자열의 ID가 달라도 캐시 키가 같음을 재현했다. 또 conversation 변환에서 `system` 역할이 빠지는 사례를 재현했다. [복원한 SFT 코드](https://huggingface.co/pathcosmos/frankenstallm/blob/b9f7777f01479b18e5a3a55f9cad4582df122f7d/data/sft_dataset.py)

조치: tokenizer 내용 해시·정규화/템플릿 버전·입력 내용 버전을 캐시에 포함하고, system/user/assistant 역할 및 도중 종료·잘림 정책을 명시한다. `imap_unordered`로 합친 순서는 고정되지 않으므로 샘플 ID를 보존해 재현 가능한 순서를 만든다. 캐시는 임시 파일과 완료 마커로 확정한다.

정상 사례도 확인했다. 작은 순차 토큰화 입력에서 응답 첫 토큰은 prompt 마지막 위치의 label에 들어가며, 잘린 응답 끝 EOS도 label에 남는다. 이 정상 동작은 이식 시 유지한다. 과거의 ignore_index는 `-1`, 현재 모델 기본 CE는 `-100`이므로 그대로 연결하면 `Target -1 is out of bounds`가 발생한다. 현재 사전학습 경로의 오류가 아니라 SFT 이식 시 맞춰야 할 계약이다.

### 선호도 라벨 없는 데이터의 자동 선택

`H:data/prepare_preference_combined.py`에 `response_a`, `response_b`만 있고 winner가 없는 입력을 넣으면 A를 chosen으로 선택하고 품질 필터도 통과한다. [정규화 코드](https://huggingface.co/pathcosmos/frankenstallm/blob/b9f7777f01479b18e5a3a55f9cad4582df122f7d/data/prepare_preference_combined.py#L121)

조치: 소스별 라벨 스키마를 명시하고, 누락·동점·알 수 없는 라벨은 제외 사유와 함께 격리한다. source ID·원본 row ID를 출력에 보존하고, prompt 및 유사 문서 그룹 단위로 train/eval을 분리한다. 실제 공개 JSONL에 이런 입력이 얼마나 들어 있는지는 아직 측정하지 않았다.

### 변환 설정과 토크나이저의 차이

G의 모델은 embedding/head를 공유하며 변환기도 head를 embedding에서 복사한다. 현재 3B는 공유하지 않으므로 별도 head를 보존해야 한다. G의 일반 RMSNorm 기본 eps는 `1e-6`, 변환 config는 `1e-5`로 고정되어 있다. FP8 융합 경로의 실제 eps까지 확인한 것은 아니며, 변환 전에 경로별 값을 확인해야 한다. [RMSNorm](https://github.com/pathcosmos/FRANKENSTALLM/blob/50c6143629203815dfe7775e3b7c0dd0a3260324/model/layers.py#L34), [변환기](https://github.com/pathcosmos/FRANKENSTALLM/blob/50c6143629203815dfe7775e3b7c0dd0a3260324/scripts/convert_to_hf.py#L122)

G의 `tokenizer/korean_sp/tokenizer.json`은 Unigram 64000·byte_fallback=false, G의 `tokenizer/tokenizer.json`과 H 루트 tokenizer는 Unigram 64256·byte_fallback=true이다. 뒤의 두 JSON은 SHA256이 같다. 과거 평가 기본 경로가 전자를 가리키므로 파일명을 함께 기록해야 한다.

현재 byte-level BPE 65536에 과거의 256-token 추가 패치를 적용하면 65792가 되어 uint16 저장 범위를 넘는다. 패치를 가져오지 않고 현재 tokenizer의 Unicode·개행·특수토큰 roundtrip을 검증한다. CPU FP32 단일 입력에서는 과거/현재 RoPE 함수 출력 차이 0을 확인했으나, 전체 모델·FP8·변환 동등성으로 확대 해석하지 않는다.

### 현재 코드에서 먼저 보완할 경계 조건

- `C:train/data.py`는 `rng.integers(0, n - T)`를 사용한다. `n=T`인 유효한 단일 window에서 `high <= 0` 오류, `n=T+1`에서 마지막 시작점이 선택되지 않는 동작을 재현했다. 유효 window 수 `n-T+1`과 짧은 샤드 거부를 함께 수정해야 한다. 샘플링 규칙 변경은 학습 재개 호환성에 영향을 주므로 버전을 기록한다.
- `C:data/filters.py`의 웹용 `exact_hash`는 소문자화와 공백 축약을 한다. 의미가 다른 Python 들여쓰기·대소문자 입력도 같은 해시가 됨을 재현했다. 코드 소스에는 대소문자·들여쓰기를 보존하는 별도 정규화와 중복 기준이 필요하다.
- 현재 FSDP 평가의 랭크별 forward 횟수, 분산 재개 수치 비교, 비유한 gradient의 optimizer 이전 종료는 기존 단계 5의 미완료 검증으로 유지한다. 이번 CPU 검사는 이를 대신하지 않는다.

## 파일별 재사용 판단

P0는 성능 판단·본학습 전에 필요한 항목, P1은 현재 데이터 준비·평가에 필요한 항목, P2는 SFT/배포 단계 항목이다. 아래 C 목적지 중 아직 없는 파일은 구현 제안이다.

| 과거 위치 | 현재 대응 또는 목적지 | 판단 | 필요한 변경과 완료 조건 | 우선순위 |
|---|---|---|---|---|
| G `model/attention.py` | C `model/model.py` | 원리·검증 사례 참고 | GQA/RoPE causal 동등성. FP8/QKV 최적화는 별도 성능 비교 | P1 |
| G `model/layers.py`, `transformer.py` | C `model/model.py` | 선택 참고 | norm eps·tie·초기화·forward 반환 계약 유지 | P0 |
| G `train/pretrain.py`, `trainer.py` | C `train/train.py` | 구조 유지, 운영 사례 참고 | DDP 로직 복사 대신 FSDP2의 전체 랭크 종료·평가·재개 검증 | P0 |
| G `train/utils.py` | C DCP 저장·재개 | 사례 참고 | `.complete`·정확 step·optimizer 복원·설정 충돌 검증 | P0 |
| H `data/dataset.py` | C `train/data.py` | 메모리 매핑 참고 | 단일/짧은 샤드 경계, token ID·dtype·window 수 검증 | P1 |
| H `data/merge_bins.py`, G `scripts/prepare_3b_data.sh` | C `data/tokenize_shards.py`, `build_mix.py` | 개념 참고 | 샤드 스트리밍·입력/토크나이저 해시·holdout 원자적 생성 | P1 |
| G `eval/data_inventory/`, `eval/domain_survey/` | C 데이터 후보 목록 | 조사 출발점 | 접근·원 소스 조건·중복·실제 유효 토큰량 재확인 | P1 |
| H `data/filter_sft_v2.py` | C 향후 `sft/` 정제 | 수정 후 참고 | 50자 미만 일괄 제거를 과제별로 검토. 짧은 정답·코드·JSON 보존 | P2 |
| H `data/sft_dataset.py` | C 향후 `sft/data.py` | 재작성에 참고 | 캐시 해시·system 역할·순서·마스킹·EOS·빈 응답 검증 | P2 |
| G `train/sft.py` | C 향후 `sft/train.py` | 수정 후 참고 | FSDP2, 현재 tokenizer, ignore_index, 유효 응답 토큰 가중 loss | P2 |
| H `data/prepare_preference_combined.py` | C 향후 preference 전처리 | 수정 후 참고 | 불명 라벨 거부·출처 유지·prompt 그룹 분할 | P2 |
| G `train/orpo.py` | C 향후 선호학습 | 보류 | Base/SFT 정상 평가 후 필요성 판단. 실제 TRL 패치 확보·재현 | P2 |
| G `eval/tasks/ppl_task.py` | C 향후 `eval/perplexity.py` | 재작성 | 중복 없는 coverage·tail·0-token 오류·토큰 가중 집계 | P0 |
| G `eval/tasks/lm_eval_task.py` | C 향후 benchmark adapter | 수정 후 참고 | 모델 로드 실패 중단·누락 task 표시·버전/seed/checkpoint 해시 고정 | P0 |
| G `eval/tasks/generation_task.py` | C 향후 생성 평가 | 프롬프트/지표 참고 | Base/Chat 구분, 현재 EOS/template, seed·길이 고정, 표본 확대 | P1 |
| G `eval/tasks/token_nll_task.py`, `calibration_task.py` | C 보조 진단 | 평가 정상화 후 참고 | tokenizer 차이와 표본 선택을 기록 | P2 |
| G `scripts/convert_to_hf.py` | C 향후 export 도구 | 재작성 | DCP 복원·별도 head·eps·키·shape·logits 동등성 | P0 |
| G `scripts/fix_tokenizer_byte_fallback.py` | C tokenizer 검사 | 실패 사례만 활용 | 65536 BPE에 패치 적용 안 함. Unicode 회귀 입력으로 전환 | P1 |
| G `scripts/quality_gate.sh` | C 단계별 검증 | 기준 구성 참고 | 과거 임계값을 성능 근거로 승계하지 않음. 필수 평가 누락 시 실패 | P1 |
| G `scripts/monitor_3b.sh`, `training_watchdog.sh` | C 실행 관찰 | 필요한 관찰 항목만 참고 | 실제 PID·로그·OOM·디스크·완료 상태 추적 | P1 |
| G `scripts/telegram_notify.py`, `upload_to_huggingface.py` | 외부 연동 | 보류 | 현재 작업에 필요 없음. 실행 시 기존 사용자 승인 규칙 적용 | P2 |
| G `model/mamba_block.py`, hybrid configs | 현재 Dense 목표 | 보류 | 현재 확정 설계 범위 밖 | 해당 없음 |

## 추가로 확보하거나 측정할 것

1. **과거 정상 Base 가중치와 실행 연결 정보**: 변환 이전 checkpoint의 모델 config, tokenizer 해시, 실제 step, 실행 명령·수정 사항. optimizer·scheduler·RNG는 학습 재개 재현에 필요하다. 손상된 HF Base export를 원본 대체물로 사용하지 않는다.
2. **TRL 패치와 환경**: 패치 diff, 정확한 TRL/Transformers/torch/TransformerEngine 버전, 설치 또는 컨테이너 식별자. 이를 확보하기 전 과거 ORPO 그대로 재현 완료로 표시하지 않는다.
3. **현재 사전학습 데이터**: 코드 소스 선정·샤드 간/소스 간 중복제거·한국어/영어 holdout·고유 토큰량과 반복 노출량. 한국어 웹을 더 늘리기 전에 확보분의 겹침과 실제 품질을 측정한다.
4. **SFT·선호도 품질 근거**: 확보 가능한 JSONL에서 출처별 표본, 중복, 정상 단답·수학·코드·다중 턴·system 역할, 선호 라벨 신뢰도를 검사한다. 원본 데이터셋 이용 조건과 split/revision을 따로 기록한다. 공개 저장소라는 사실만으로 각 원 소스 조건을 대체하지 않는다.
5. **평가 표본과 실행 식별자**: KMMLU/KoBEST/HAE-RAE 및 고정 생성 표본의 정확한 revision·task·shot·seed·template·sample IDs. 학습 데이터와 평가 문서의 중복을 확인한다.

현재 다른 데이터 준비 작업에서 `check_math`와 URL overlap 표본 분석 코드가 추가된 사실을 확인했다. 따라서 이를 미작성으로 되돌리지 않는다. 전처리 실행 진척은 [STATUS](STATUS.md)의 해당 기록을 따르고, 이 검토의 다음 구현은 중복제거·holdout·코드 소스 및 평가 경계에 집중한다.

## 실행 순서와 완료 조건

| 순서 | 작업 | 완료 조건 | 권장 개발 모델 |
|---|---|---|---|
| 1 | 자산·코드·평가 분류 | 고정 revision, 파일 해시, 누락 목록, 모델 크기 구분 | Astra / High — 이번 완료 |
| 2 | 현재 sampler 경계와 최소 PPL 구현 | 단일 window·tail·표적 coverage·0-token 검사 통과 | Astra / High |
| 3 | export 및 평가 로드 검사 | tiny 모델 CPU에서 키·shape·tie·eps 검증, 동일 입력 logits 비교. 실제 DCP는 별도 확인 | Astra / High |
| 4 | 데이터 품질·holdout 확정 | 소스 간 중복제거, 코드용 정규화, tokenizer 해시, 소스별 고유 토큰·평가 분리 기록 | GPT-6.1 Sol / High |
| 5 | 기존 단계 5·6 검증 마무리 | 8GPU 불균등 평가·연속/재개·비유한 종료, 실제 mix I/O·compile 안정성 | Astra / High, 성능은 xhigh |
| 6 | 리허설·Base 평가 후 SFT | 위 검증과 데이터가 준비된 상태에서 기존 WORKFLOW 진행 | WORKFLOW의 단계 7 이후 권장값 |

이 표는 모델 전체 이식이나 본학습 착수의 완료 보고가 아니다. 이번에는 참고 자료 확보, 파일별 판단, 헤더 검사, 작은 CPU 재현까지 완료했다. 현재 학습/전처리 코드는 수정하지 않았다.

## 다음 구현에 사용할 검증 사례

| ID | 입력 또는 비교 | 기대 결과 |
|---|---|---|
| E01 | 평가 길이 0·1·seq_len·seq_len+1, 겹치는 window, 마지막 tail | 정책에 맞는 표적을 각각 1회 평가. 유효 표적 0이면 오류 |
| E02 | 유효 응답 길이가 다른 두 배치 | 전체 NLL 합 / 전체 유효 토큰 수와 일치 |
| E03 | 정상·Q/K/V 누락·shape 불일치 체크포인트 | 필수 키/shape 실패 시 점수 산출 전에 중단 |
| E04 | tied/untied 작은 모델 export | head 관계 보존, 동일 float32 입력에서 logits `atol=1e-5, rtol=1e-4` 시작 기준 검증 |
| E05 | 동일 vocab 크기, 다른 tokenizer 매핑 | 캐시 재사용 거부 |
| E06 | system/다중 턴/빈 응답/긴 응답/특수토큰 리터럴 | 템플릿 정책·응답 mask·EOS·정상 거부 사유 확인 |
| E07 | preference winner 누락·동점·알 수 없는 값 | 임의 A/B 선택 없이 분리·제외 |
| E08 | 코드 들여쓰기·대소문자만 다른 문서 | 의미를 바꿀 수 있는 입력을 정확중복으로 합치지 않음 |
| E09 | 8GPU에 나누어떨어지지 않는 holdout window 수 | 모든 랭크 통신 순서 일치, hang 없이 정확 집계 |
| E10 | 동일 seed 연속 실행과 저장·재개, 비유한 gradient 주입 | 배치 동일성·가중치 오차 기록, optimizer 갱신 전 전체 랭크 종료 |

E04 허용 오차는 작은 CPU FP32 검증의 시작 기준이다. bf16·compile·8GPU에는 별도의 사전 허용 오차와 근거를 기록한다. E01~E10 전체가 구현·통과한 것은 아니다. 이번 실행 결과는 아래와 같다.

| 이번 CPU 검사 | 결과 |
|---|---|
| 과거 PPL coverage | 중복 6회·마지막 표적 누락 재현 |
| 과거 SFT 캐시 | 서로 다른 tokenizer의 캐시 키 충돌 재현 |
| 과거 SFT system 역할 | 누락 재현 |
| 과거 SFT shift/EOS | 합성 입력의 정상 정렬·잘림 후 EOS 유지 확인 |
| 과거 preference winner 누락 | A 자동 선택 및 필터 통과 재현 |
| RoPE FP32 함수 비교 | 단일 입력 최대 절대차 0 |
| 현재 모델의 과거 ignore_index 입력 | -1 label 거부 확인 |
| 현재 3B 파라미터 수 | meta device에서 3,221,400,576 확인 |
| 현재 sampler 경계 | 단일 window 오류·마지막 시작점 배제 재현 |
| 현재 웹 해시를 코드에 적용 | 들여쓰기·대소문자 차이를 같은 값으로 처리함을 확인 |

## 재현과 기록 위치

프로젝트 루트에서 실행한다.

```bash
python scripts/audit_frankenstallm.py
```

기본 출력은 `docs/reviews/frankenstallm/evidence.json`과 `.done`이다. 같은 입력과 결과 해시가 확인되면 완료 결과를 재사용한다. 입력 파일이 바뀌면 이전 결과를 덮어쓰지 않고 새 `--out-dir`를 요구한다. 새로 실행하려면 예를 들어 다음처럼 별도 경로를 사용한다.

```bash
python scripts/audit_frankenstallm.py --out-dir runs/frankenstallm_audit_repeat
```

새 환경에서 CPU 검사를 재현할 때 필요한 입력은 다음과 같다.

1. G를 `.cache/references/FRANKENSTALLM`에 clone하고 위 커밋으로 detached checkout한다.
2. 증거 JSON의 `huggingface.download_manifest.files` 경로들을 H의 고정 revision에서 받아 `.cache/references/frankenstallm_hf/<revision>/`에 둔다. 기록된 SHA256과 대조한다. 가중치 본문·학습 JSONL은 이 목록에 없으므로 받지 않는다.
3. 상위 참고 디렉터리에 `download_manifest.json`, HF pinned revision의 `model_info_full.json`, 네 가중치의 `all_safetensors_headers.json`을 둔다. 헤더는 Range 응답 206과 Content-Range를 확인한 뒤 첫 8 bytes의 길이 및 그 길이만큼의 JSON을 읽는다. 원격 독립 재검증 없이 CPU 검사만 반복할 경우 증거 JSON의 `weight_profiles[].header`를 입력 스냅샷으로 사용할 수 있다.
4. 새 출력 경로로 스크립트를 실행한다. 원격 다운로드 카운터 같은 메타데이터는 달라질 수 있으므로 이전 `.done`과 같은 fingerprint를 기대하지 않는다.

`evidence.json`의 `evaluation_missing_weight_warnings`는 경고가 확인된 파일과 행 번호를, `evaluation_provenance`는 읽은 결과 JSON의 모델·seed·task 정보를 담는다. `source_comparison`과 `current.working_file_sha256`으로 실제 비교 범위를 확인할 수 있다.
