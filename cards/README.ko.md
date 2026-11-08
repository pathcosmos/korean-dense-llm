# Korean Dense LLM — 한국어

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## 개요

한국어 중심 Dense LLM을 직접 구현하며 사용한 데이터 준비·모델·학습·평가·복구·최적화 탐색 코드를 공개합니다. 소스와 실험 기록 배포이며 데이터 본문·토큰 바이너리·DB·모델 가중치는 별도로 제공합니다.

문서 번역 언어는 학습 데이터에 그 언어가 포함된다는 뜻이 아닙니다. 주 자연언어는 한국어·영어이며 웹·수학·코드에 다른 언어가 섞일 수 있습니다.

## 구성과 규모

| | |
|---|---:|
| common/, data/ | 처리와 분리 |
| tokenizer/ | 처리와 분리 |
| model/, train/, eval/ | 모델과 학습 |
| scripts/ | 최적화·벤치마크 |
| tests/, examples/ | 검증 |
| experiments/ | 47 / 99 |


## 처리와 분리

웹은 공백 정리, 길이·반복·언어 휴리스틱, 일부 개인정보 패턴 마스킹을 거칩니다. 초기 샤드 내부 hash와 최종 웹 NFC SHA256의 기준은 다릅니다. 코드 정제는 대소문자·들여쓰기·원문 Unicode를 보존하고 경로·라이선스 표기·생성 코드·일부 비밀정보 패턴을 검사합니다. 동일 본문·웹 NFC 동등 본문·코드 저장소의 연결 그룹을 train/eval로 분리합니다.

## 사용법

아래 명령은 저장소 루트에서 실행합니다. 새 출력/다운로드 디렉터리를 사용하고 다운로드 전에 범위·용량·고정 revision을 확인하세요. 합성 CPU 예제는 외부 데이터를 받지 않습니다. 실제 데이터는 별도 배포에서 필요한 파일만 선택하고 CHECKSUMS.json 또는 소스별 manifest의 크기·SHA256을 검증하세요.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## 모델과 학습

모델은 RMSNorm·RoPE·GQA·SwiGLU를 사용하는 순수 PyTorch Dense decoder입니다. 실제 3B 경로는 FSDP2·8GPU·eager이며 문맥4096, micro batch4, global batch2,097,152 tokens, peak LR3e-4, 목표57,000step입니다. 예제의 CPU micro 모델과 실제 분산 학습 경로는 다릅니다. 체크포인트 재개에는 모델·optimizer·RNG·입력/설정 계약 검증이 필요합니다.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## 최적화·벤치마크

worker·thread·batch·SQLite 캐시·RAM·로컬/NFS와 eager/compile·정밀도·GPU 메모리를 비교했습니다. 총시간·처리량·peak 메모리·내용/수치 동등성을 함께 확인하고 채택·기각 이유를 기록합니다. 47개 캠페인·99개 결과 JSON을 제공하며 빠르더라도 수치 조건에 실패한 후보를 권장하지 않습니다. 짧은 표본의 최적값을 전체 최적값으로 주장하지 않습니다.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## 검증

GH 소스 공개판의 독립 CPU 검사는284통과/9건너뜀, 새 GitHub CPU 환경은258통과/35건너뜀입니다. 건너뜀은 원래 운영 기록 또는 별도 파일시스템이 필요한 검사입니다. 합성 quickstart는120문서·train100/eval20·vocab272·108,864parameter 모델로 저장/재개 후 model과optimizer 일치를 확인합니다. 데이터 배포의 내용 해시·분할/reader 검사와 전체3B 독립 재현은 구분합니다.

## 출처와 라이선스

프로젝트 원본 코드는 MIT, 원본 설명문은 CC BY4.0입니다. 이는 원문 데이터의 권리를 대신하지 않습니다. 소스별 라이선스·귀속·고지·알려진 미확정 항목은 source.json, LICENSE-SCOPE.md와 출처 문서를 확인하세요. 모든 원문에 대한 무조건적 상업 재배포 허가를 주장하지 않습니다.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## 한계

전역 유사중복 제거·개인정보/secret 완전 제거·공식 평가 오염 없음·최종 모델 품질을 보장하지 않습니다. 공개 시점의 본학습은 진행 중이며 SFT·모델 export·공식 언어능력 benchmark 완료를 뜻하지 않습니다. 과거 실험11개 코드 참조/고유9hash는 원본 snapshot을 회수하지 못했습니다. 일부 운영 도구는 별도 원래 실행 manifest가 필요합니다.

## 관련 저장소

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / 사용법](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / 최적화·벤치마크](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / 검증](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
