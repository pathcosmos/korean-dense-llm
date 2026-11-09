# 토큰화·학습 입력 계층 평가 (읽기 전용)

대상 저장소: /home/user/korean-dense-llm. 저장소 파일은 수정하지 않았고, 실험은 scratchpad/syn 에서만 수행했다. PYTHONDONTWRITEBYTECODE=1 로 실행했으며 전체 테스트 스위트는 실행하지 않았다.

## 점수표
| 축 | 점수 | 한 줄 근거 |
|---|---|---|
| A 기술품질 | 4 | content-hash 계약, 크래시 재개, 다층 게이트와 실패경로 테스트가 있다. 감점: literal special token 주입 미차단, 한국어 품질 지표가 얇음, 실측 결과 미동봉 |
| B 포트폴리오 | 4 | 3GB 학습 → 독립 eval 검사 → 1224억 토큰 전수 처리 → 가중치 리뷰 게이트까지 증거 체인이 있다. 감점: 외부 토크나이저 대비 한국어 지표가 없음 |
| C 재사용성 | 2 | 핵심 로직은 일반적이지만 repo 전용 계약(split/F4/hardware_profile)에 강하게 결합돼 있고, 한 줄 압축 스타일에다 하드코딩(96 CPU, 512GiB, 16/24 worker)이 있다. 별도 호출 가능한 부분(make_tokenizer, uint16_ids, MixSampler)만 재사용 가능 |

## 질문별 답
### 1. BPE 학습 입력과 byte mix 정책
- 학습기는 byte-level BPE다. `Digits(individual_digits=True)` 뒤에 `ByteLevel(use_regex=True)`를 쓰고, `initial_alphabet=ByteLevel.alphabet()`, `min_frequency=2`, 특수토큰 16개(5개 명명 + reserved 11)를 둔다 (tokenizer/train_tokenizer.py:16-29, 72-73).
- 정규화는 `raw`/`NFC` 선택형이다. 최종 선택은 raw이고, 근거는 LESSONS 10번(NFC가 코드 literal의 바이트를 바꿈)이다 (scripts/run_final_tokenizer.py:99, 129).
- mix 가중치는 토큰이 아니라 UTF-8 바이트 기준이다 (train_tokenizer.py:4). 실제 F5 예산은 `byte_budgets` 가 합계 3,000,000,000을 정확히 나누고 최대잔여법으로 반올림한다 (run_final_tokenizer.py:50-54). 실측 예산은 fineweb_edu 1.05GB(35%), fineweb2_ko 0.91GB, hplt_ko 0.58GB, wiki_ko 5MB, github_code 125MB(약 4.2%), owm 325MB(약 10.8%)이고, 한국어 합계는 50%다 (experiments/results/f5_tokenizer_v1/contract.json, report.json의 train_sample_stats).
- 학습 입력은 `data.split_inputs.prepare(mode='train')` 가 만든 train-only 표본이다. eval 그룹이 제외되고 검증 가능한 매니페스트와 결합된다 (train_tokenizer.py:54-70, 102-107). 문서는 20000자로 절단된다 (:106).
- 토큰 mix(학습 시 노출)는 바이트 mix와 의도적으로 분리됐다. 카테고리 목표는 ko/en/code/math = 50/30/10/10이다 (scripts/run_final_tokens.py:102-106, 136). 실제 토큰 수를 본 뒤 wiki를 0.5%로 상한했고 그 근거는 mix_review.json 에 있다.
- 불일치 한 가지: 토크나이저는 code 4.2%·math 10.8% 바이트로 학습됐는데 학습 토큰 mix는 code 10%·math 10%다. 코드 어휘가 상대적으로 덜 학습됐을 수 있다. 이 영향은 효율 실측값이 없어 측정 불가다.
- 레거시 `--experimental-unsplit` 경로에는 소스별 예산 미달을 감지하는 검사가 없다. `sample_texts` 의 `per_file` 은 파일 수로 균등 분배되며 부족분을 재배분하지 않는다 (train_tokenizer.py:32-34, 125-127). 반면 `train_prepared` 경로는 어휘 도달을 검사한다 (:74).

### 2. 한국어 품질 측정이 실제로 있는가 (있는 것만)
있는 것:
- 카테고리(ko/en/code/math)별 `chars_per_token`, `bytes_per_token` 을 계산한다 (tokenizer/check_tokenizer.py:225-227). fertility 의 역수 성격이다. 단어 단위 fertility 는 없다.
- 효율 게이트: baseline(128MiB 실험용 64k 토크나이저) 대비 카테고리별 토큰 비율 ≤1.10, byte 가중 토큰/바이트 비율 ≤1.02 (run_final_tokenizer.py:57-66, 143-144). 둘 다 train 에서 제외된 독립 eval 문서(소스당 1024개)로 측정한다 (tokenizer/quality_inputs.py:301-328).
- 바이트 폴백: ByteLevel 이라 UNK 가 없다. 합성 실험에서 미등록 한글 '뷁쀍 꿻'(10바이트)이 10토큰으로 바이트 단위 분해되고 왕복이 정확했다.
- NFC/자모: 정책 불일치(NFC 선언인데 raw 토크나이저 등) 감지, NFC 왕복 검사, 코드 raw 변화 카운트로 needs_review(exit 2)를 낸다 (check_tokenizer.py:183-184, 204-217, 229). 합성 실험에서 NFD 한글 '한'(3자모)은 raw 에서 12바이트 12토큰, NFC 정책에서 5토큰이었다. NFC 정책은 NFD 입력 복원에 실패(False)해 정책 검사에서 잡힌다.
없는 것(코드에서 확인 불가):
- 한국어 전용 지표(음절당·어절당 토큰, 형태소 경계 정렬)와 외부 토크나이저(Llama/Qwen 등) 대비 비교.
- 분해형 자모(NFD) probe. PROBES 의 ko 항목은 합성형 한글뿐이다 (check_tokenizer.py:158-164).
- 효율의 절대 임계값. 게이트는 baseline 대비 상대비뿐이다. baseline 도 같은 계열 byte-BPE 이므로 상대 우위만 판단 가능하다.
- 실측 수치: f5_tokenizer_v1 에는 contract.json 과 report.json 만 있고 tokenizer_check.json, efficiency.json 은 sha 참조만 되어 있다. 따라서 실제 한국어 chars/token 값은 측정 불가이며 numbers 를 인용하지 않는다. 합성 코퍼스 수치(ko 3.93 chars/token)는 600 어휘 장난감 값이라 의미가 없다.

### 3. 토큰 shard 포맷
- 포맷은 `uint16-eos-v2`/`uint16-eos-split-v3` 이고 문서 뒤에 EOS 를 붙이며 `<u2` 로 직렬화한다 (data/tokenize_shards.py:19-20, 126, data/parallel_tokenize.py:99).
- 어휘 한계: 어휘 65536 이면 ID 0..65535 라 uint16 에 정확히 들어맞는다. 여유 ID 는 0이다. 학습기·CLI 가 272..65536 으로 제한하고 (train_tokenizer.py:57, 93), 체커가 ID 범위·연속성을 검사한다 (check_tokenizer.py:186-189). 캐스트 전에 `uint16_ids` 가 범위·정수형을 검사한다 (tokenize_shards.py:23-27). 실험에서 70000/-1/1.5 가 모두 거부됐다.
- 한계: 이 포맷은 vocab>65536 으로 확장할 수 없다 (포맷 변경 필요).
- 쓰기 프로토콜: `.tmp` → sha256 → `.pending.json` → rename → `.done`. 크래시 후 재개와 손상 탐지가 있다 (tokenize_shards.py:34-44, 138-143). 합성 재실행에서 REUSED 가 확인됐고, 계약이 바뀌면 거부됐다.
- 실제 결과: 약 1224억 토큰(122,402,393,813), 726 shards, 출력 약 245GB (experiments/results/f6_tokens_v1/report.json).

### 4. mix 가중치 샘플링의 결정성
- `MixSampler.batch(step,rank,micro,bsz)` 는 `np.random.default_rng([seed,step,rank,micro])` 로 상태 없이 동작한다 (train/data.py:342-357). 합성 실험에서 같은 인자는 동일 배치, step/rank 가 다르면 다른 배치였다. bsz 를 바꿔도 첫 샘플은 안정적이었다. 소스 선택 비율은 0.75/0.25 대비 실측 0.737/0.263 이었다(1600 draws).
- 샘플러 v2 는 윈도우 수 가중치로 모든 유효 오프셋(마지막 포함)을 포함한다. v1 은 호환용이며, 재개 시 버전 불일치를 거부한다 (train/data.py:278-283, 305-306).
- 주의: ① 복원추출이라 epoch 보장이 없다. mix_review.json 도 "expectations, not strict epochs" 라고 명시한다. wiki 는 대상 토큰 119.5B 에서 기대 노출 2.04회, github_code 2.98회다. ② 결정성이 rank 에 묶여 world size 가 바뀌면 데이터 순서가 달라진다. ③ 윈도우가 문서 경계(EOS)를 넘고 마스킹은 없다. 이는 흔한 방식이다. ④ build_mix 는 `expected_exposures` 를 계산해 매니페스트에 기록한다 (data/build_mix.py:230-233).

### 5. 실패경로 게이트
- build_mix: 불완전 소스, tokenizer/EOS/dtype 불일치, 짧은 shard, 비정상 weight 를 거부한다 (build_mix.py:185-224). 합성 실험에서 'Token shard is short' 가 확인됐다.
- final tokenizer: 효율 비율 초과 시 needs_review 가 되고 verify_tokenizer_run 이 거부한다 (run_final_tokenizer.py:64, 109; run_final_tokens.py:37, 44). 분포 리뷰·실행 리뷰·mix 리뷰는 sha 결합 증거 파일이 없으면 진행하지 않는다 (run_final_tokenizer.py:42-47; run_final_tokens.py:86-99, 128-140). 이 모두를 `while not exists: sleep(30)` 으로 대기한다.
- 자원 감시: RSS/디스크/시간 예산을 초과하면 abort 한다 (run_final_tokens.py:259-261).
- 병렬 토큰화: 입력 변경, 원문 해시 변조, 분할 행 커버리지 불일치를 거부한다 (parallel_tokenize.py:64, 74-77). `os._exit` 크래시 복구 테스트가 있다 (test_parallel_tokenize.py:86). serial/병렬 바이트 동일성도 테스트한다 (:45-52).
- 테스트: 해당 파일들에 31개 테스트(def test 개수 합)가 있다. 이번에 실행하지는 않았다.
- 한계 1(확인된 결함): corpus 문서에 literal 문자열 `<|eos|>` 등이 있으면 실제 special ID 로 토큰화된다. 실험에서 'a <|eos|> b' → ids [80,236,**1**,283] 였다(1=<|eos|>). tokenize_shards.py:116 과 parallel_tokenize.py:68 은 `encode_batch(texts, add_special_tokens=False)` 만 쓰고 sanitize/차단이 없다 (`add_special_tokens` 는 post-processor 플래그라 본문의 added-token 매칭을 막지 않는다). 사후 탐지는 data/release_offsets.py:31 의 EOS 개수 대조('literal special tokens require replay')뿐이다. 업스트림 정제에서 이를 제거하는지는 불확실이다(범위 밖 코드 미확인).
- 한계 2: 체크 probe 는 special token 의 atomic 여부만 본다. 코퍼스 내 문자열 주입 여부는 보지 않는다.
- 한계 3: 대규모 게이트는 hardware_profile(512GiB headroom, 96 CPU 가정)에 하드코딩돼 있어 작은 머신에서는 시작 자체가 거부된다 (run_final_tokenizer.py:124, run_final_tokens.py:170, 91 의 worker 16/24 고정).

## 근거 기반 상세 점수
### A 기술품질: 4
- 정확성: 합성 코퍼스에서 raw 왕복 passed, NFC 정책은 needs_review(exit 2)였고, uint16 가드, 재개, 계약 변경 거부, 샘플러 결정성이 모두 의도대로 동작했다.
- 견고성: atomic 쓰기, sha 계약, 코드 해시 고정, 실행 중 변경 감지 (parallel_tokenize.py:143-146) 가 있다. BPE 스레드 수 간 semantic identity 동일성도 검사한다 (benchmark_tokenizer_resources.py:162, 169).
- 감점 사유: literal special 주입, 한국어 지표가 얇음, 효율 임계값이 상대값뿐, 실측 결과 파일(체크/효율) 미동봉, 레거시 경로 예산 미달 무검사.

### B 포트폴리오: 4
- 실제 규모 증거(1224억 토큰, 726 shards, 전체 인코딩 약 5,416초)와 "리뷰를 증거 sha 에 묶는" 설계가 차별적이다. 가정과 범위를 명시하는 scope 문구 (예: efficiency 의 "not corpus-wide tokenizer quality") 도 정직하다.
- 5점이 아닌 이유: 한국어 토크나이저의 핵심 수치(fertility, 외부 비교)가 공개 산출물에 없고, 코드가 한 줄 압축 스타일이라 가독성이 떨어진다.

### C 재사용성: 2
- 재사용 가능한 부분: `make_tokenizer`, `uint16_ids`, `MixSampler`, `build_mix`, `check`(JSONL {category,text} 입력)는 독립 호출이 가능하다. 합성 실험이 그대로 동작했다.
- 방해 요인: 매니페스트/split/F4/로컬 sqlite 계약 의존, 하드웨어·경로 하드코딩, 압축 스타일, 한국어 전용 설정이 없음, 문서화된 간단 진입점이 없음.

## 핵심 발견 5개
1. byte mix 는 UTF-8 바이트 기준 3GB 정확분할(ko 50%, en 35%, code 약 4.2%, math 약 10.8%)이고 토큰 mix 는 별도의 50/30/10/10 이다. 이 분리는 의도적이지만, code 가 학습 바이트 4.2% 대 토큰 목표 10% 로 어긋난다.
2. 한국어 품질 측정은 chars/bytes per token 과 baseline 대비 상대 효율 게이트뿐이다. fertility(어절당), NFD 자모 probe, 외부 토크나이저 비교는 없고, 실측 수치 파일이 동봉되지 않아 값은 측정 불가다.
3. vocab 65536 은 uint16 의 상한에 정확히 맞고 여유 ID 가 없다. 범위·연속성·캐스트 전 검사가 3중으로 있어 안전하다.
4. 샘플러는 (seed,step,rank,micro) 기반 stateless 라 결정적이지만, 복원추출이고 rank 에 종속되며 wiki/code 노출이 2~3회라 epoch 보장은 없다.
5. 실패경로 게이트는 강력하다(sha 결합 리뷰, 크래시 재개, 변조 거부). 그러나 corpus 의 literal `<|eos|>` 가 실제 EOS 로 토큰화되는 결함은 인코딩 시점에 막히지 않는다(실험 확인). 사후 EOS 개수 대조에만 의존한다.
