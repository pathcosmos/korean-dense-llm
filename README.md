# Korean Dense LLM

한국어 중심 3B Dense LLM을 직접 구현하며 사용한 **데이터 준비, 모델·학습·평가, 최적화 탐색, 검증 코드**입니다. 순수 PyTorch 기반의 학습 경로와 실제 실험 기록을 함께 제공합니다.

[English](README.en.md) · [실행 가이드](docs/PUBLIC_GUIDE.md) · [벤치마크](experiments/README.md) · [검증과 한계](docs/PUBLIC_VALIDATION.md) · [코드 목록](release/source_inventory.json)

## 범위와 현재 상태

- 데이터 탐색/선정, 고정 revision 다운로드, 언어·코드 정제, 정확중복 제거, 유사중복 표본 조사, 내용 감사, 그룹별 train/eval 분리.
- train-only byte-level BPE, 내용 계약에 연결된 토큰화와 mix, stateless sampler.
- RMSNorm/RoPE/GQA/SwiGLU Dense 모델, FSDP2 학습, native 평가, 체크포인트·RNG 재개와 보존.
- CPU·메모리·저장소·GPU 설정 탐색, 성능·수치 비교, 채택하지 않은 후보의 근거.

**v0.1.0은 소스와 실험 기록 공개판입니다.** 원문 데이터·토큰 바이너리·DB·모델 가중치는 포함하지 않습니다. 학습은 공개 snapshot 시점에 진행 중이며 최종 모델 품질·공식 언어능력 벤치마크 통과를 주장하지 않습니다. SFT와 HF 모델 export는 이 버전의 완성 기능이 아닙니다.

## 확인된 검증

독립 로컬 검사284개 통과/9개 건너뜀, 새 GitHub CPU 환경258개 통과/35개 건너뜀입니다. 건너뜀은 원래 운영 기록 또는 별도 파일시스템이 필요한 검사입니다. 인증 없는 public clone의 내용 해시와 quickstart도 확인했습니다. [검증 근거](docs/PUBLIC_VALIDATION.md)

## 작은 CPU 실행

Python 3.11/3.12 환경에서 저장소 루트에서 실행합니다. 아래 설치는 CPU 예제용이며 GPU 학습 환경과 구분합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```

quickstart는 합성 문서를 만들고 정제→중복 제거→감사/확정→그룹 분할→작은 토크나이저→토큰화→micro 모델 학습/평가→저장/정확 재개를 실행합니다. 외부 데이터나 GPU를 사용하지 않습니다. 기본 임시 출력은 종료 시 정리되며, 보존하려면 새 경로 `--output-dir runs/my_demo`를 지정합니다.

실제 검증 환경은 [tested_environment.json](release/tested_environment.json)에 기록합니다. 위 표준 wheel로 새 환경을 설치한 검증과 기존 NVIDIA PyTorch 환경에서의 검증은 동일하지 않습니다. 실제 수행 범위는 [검증 보고](docs/PUBLIC_VALIDATION.md)를 확인하세요.

## 디렉터리

| 경로 | 내용 |
|---|---|
| common/, data/ | 경로·내용 계약, 다운로드·정제·중복·분할·토큰화 |
| tokenizer/ | train-only 학습과 원문/특수토큰 검사 |
| model/, train/, eval/ | 모델·학습·평가·재개 |
| scripts/ | 운영·보존·복구·자원·최적화·데이터 배포 도구 |
| tests/, examples/ | 회귀 검사, CPU 시작 예제, 실제 설정과 fresh-run 설정 |
| experiments/ | 실험 결과·코드 snapshot·성능표·측정 조건 |
| docs/, release/ | 기술 문서·한계·출처·검증·파일 내용 증명 |

## 데이터와 실제 학습

기존 [전체 corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus) 및 [실제 학습 입력 배포본](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data)을 별도로 사용할 수 있습니다. 필요한 고정 revision·라이선스·크기·SHA를 먼저 확인합니다. 본 저장소 설치나 테스트가 전체 데이터를 자동 다운로드하지 않습니다.

실제 8GPU 학습 명령, 자원 요구사항, 각 단계의 입력은 [실행 가이드](docs/PUBLIC_GUIDE.md)에 있습니다. 일부 운영·발행 도구는 원래 실행 manifest를 요구하는 연구용 도구이며 바로 실행하는 범용 서비스가 아닙니다. 역사 문서는 당시 경로·설정·미완료 기록을 보존합니다.

## 라이선스

프로젝트 원본 코드 MIT, 원본 문서 CC BY 4.0. 제3자 코드·원문 데이터의 조건은 별도입니다. [NOTICE](NOTICE.md)를 확인하세요.
