# 실행 가이드 / Execution guide

## 실행 경로 구분

1. `examples/quickstart_cpu.py`: 네트워크·GPU 없이 전체 흐름을 익히는 합성 예제.
2. `data/`, `tokenizer/`, `model/`, `train/`, `eval/`: 실제 사용한 구현. 입력 manifest·완료 증명·내용 해시를 요구한다.
3. `scripts/benchmark_*`, `probe_*`, `validate_*`: 실험·검증 도구. 인자와 계약, 입력 규모를 먼저 확인한다.
4. supervisor/retention/upload/finalizer: 특정 실행 기록을 연결하는 운영 도구. 원래 `runs/`가 없으면 준비 단계가 필요하다. 기본 quickstart에서 실행하지 않는다.
5. `experiments/code/`: 바이트 그대로 보존한 과거 코드. 파일명은 SHA256이며 실행 당시 버전 연결용이다. 현재 import 경로에 넣지 않는다.

## 데이터 준비 순서

| 단계 | 주요 진입점 | 입력/산출물 |
|---|---|---|
| 후보 확인·다운로드 | data/download.py | source 이름, --list-only, --revision; download_manifest |
| 정제 | data/clean.py, data/clean_code.py | 원본 Parquet → 별도 clean + 사유별 통계 |
| 정확중복 | data/dedup.py | 순서가 고정된 clean → 별도 output/SQLite/완료 증명 |
| 표본 조사·전수 감사 | data/probe_*.py, data/source_audit.py | 범위가 명시된 정책 판단과 내용 감사 |
| 최종 확정 | scripts/finalize_source_manifest.py | candidate + 감사 + 정책 → audited ready |
| 그룹 분할 | data/split_manifest.py | raw/NFC/repo 연결 그룹을 train/eval로 분리 |
| 토크나이저 | data/split_inputs.py, tokenizer/train_tokenizer.py | train-only 표본·준비 manifest → tokenizer |
| 토큰화·mix | data/parallel_tokenize.py, data/tokenize_shards.py, data/build_mix.py | 내용 계약과 uint16/EOS/split 증명 |

초기 정제의 소문자/공백 hash와 최종 웹 NFC SHA256은 기준이 다르다. 코드 정제와 tokenizer는 원문 보존 정책을 사용한다. 전역 유사중복 제거·완전한 개인정보 제거·공식 벤치마크 무오염을 보장하지 않는다. 정책 선언은 실제 근거와 검토를 요구하며 quickstart의 합성 정책을 실제 데이터 승인에 복사하지 않는다.

## 실제 데이터 배포본과 tokenizer

기존 학습 입력은 `pathcosmos/korean-pretrain-training-data`의 **v1.0.1**에 고정되어 있다. 데이터는 별도 배포이며 약242GB의 bin을 사용하므로 quickstart와 자원 요구가 다르다. tokenizer만 받을 때는 dataset 엔드포인트를 명시한다.

```python
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
snapshot = snapshot_download('pathcosmos/korean-pretrain-training-data',
    repo_type='dataset', revision='v1.0.1', allow_patterns=['tokenizer/*'])
# snapshot의 tokenizer 디렉터리에서 tokenizer.json 경로를 확인한 후 로드한다.
```

전체 입력은 배포본 카드의 snapshot/체크섬 절차로 확보하며 이 예제에서 자동 다운로드하지 않는다. 원래 source를 재구성하려면 DATA_RETENTION_AND_RECOVERY의 고정 revision·선택 파일·정제 순서를 사용한다.

## GPU 학습 recipe

`examples/3b.observed.json`은 snapshot 시점의 실제 설정이며 자동 실행용 기본값이 아니다. `examples/3b.fresh.json`은 모델·batch·schedule을 유지하고 출력 경로와 native 보존 개수만 새 실행용으로 조정했다. 원래 supervisor의 NFS/로컬 보존·인증 기능은 fresh native 실행에 자동 포함되지 않는다.

```bash
python scripts/hardware_profile.py --output runs/hardware.json
# 실제 입력/토크나이저/평가 파일 확보 및 내용 검증, 해당 환경의 GPU 검증 이후:
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

3B: vocab65536, context4096, micro4, 8 ranks, global2,097,152 tokens, eager, peak LR3e-4, 57,000 steps. 위 경로의 정확한 CLI는 `--help`로 확인한다. 기존 train/configs/3b.json의 compile-on/micro8은 이전 실험 초안이므로 관측된 설정과 혼동하지 않는다. 메모리와 전체 입력을 가진 GPU 환경이 필요하며 CPU quickstart 통과는 8GPU 재현 통과가 아니다.

## 최적화 탐색 방법

대표 입력에서 후보를 좁히고 같은 검증 조건으로 총시간·처리량·peak 메모리를 측정한다. cache 한도와 실제 RSS를 구분하고 local/NFS, worker/thread/batch를 함께 기록한다. 느리거나 수치 정확성을 통과하지 못한 후보는 채택하지 않는다. 기존 결과는 experiments의 recipe/contract 및 코드 SHA 연결을 참고한다.

## 문서 읽는 순서

README → 이 가이드 → PUBLIC_VALIDATION → experiments/README → DATA_CONTRACT/HARDWARE_RESOURCES.
기존 RUNBOOK·LESSONS·FRANKENSTALLM_HANDOFF 등은 역사적 기술 기록이다. 오래된 경로·완료 상태를 새 공개본의 현재 상태로 해석하지 않는다. REF-DATA/TOK/TRAIN/EVAL의 요구사항은 테스트와 연결하며 acceptance JSON 자체를 검증 성공으로 간주하지 않는다.
