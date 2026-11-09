# 데이터 처리 피드백 루프 설계안 (측정 → 반영 → 수정)

상태: **T0 하네스는 `dataqa/`로 구현됨** (신규 모듈, 기존 파일 무수정, 사용법과 한계는 [dataqa/README.md](../../dataqa/README.md)). T1 일부와 T2는 설계 상태이고, 설정 분리·`clean.py` 서명 등 기존 코드 변경은 별도 승인이 필요하다. 일정은 제시하지 않는다.
관련: [README.md](README.md)의 격차 목록, 기존 실험 계약 체계(`experiments/*/contract.json`, `report.json`).

## 1. 목적과 원칙

- 목적: 필터·중복 제거·PII 규칙을 바꿀 때 "무엇이 좋아졌는지"를 측정으로 결정하고, 채택·기각을 기록한다.
- 기존 자산을 확장한다. 새 병렬 체계를 만들지 않는다: `probe_filters.py`(거절 사유별 비율), `probe_similarity_recall.py`(합성/실제 분모 분리), `eval/`(perplexity, multiple_choice, holdout), `pilot_mix.py`, 실험 contract/report 형식.
- 기존 원칙을 유지한다: 기각된 후보도 보존, 짧은 표본의 최적값을 전체 최적값으로 주장하지 않기, 측정 완료와 채택을 분리하기.

## 2. 선결 제약 (설계를 좌우하는 사실)

1. 임계값이 `data/filters.py`와 `data/clean*.py`에 하드코딩되어 있다. 변형을 실험하려면 설정으로 분리하고 그 설정을 계약에 포함해야 한다.
2. `clean.py`는 입력/정책 서명이 없다. 설정을 바꿔도 `.done`이 있으면 이전 결과를 재사용한다. 루프 도입 전에 반드시 고쳐야 한다.
3. `pilot_mix.py` 등은 producer 코드의 sha256을 검증한다. 필터를 바꾸면 하위 계약이 모두 무효가 된다. 따라서 **변형마다 새 산출물 트리**를 만들고 제자리 수정하지 않는다(저장 공간과 계산 비용이 든다).
4. 저장소에 원본 데이터가 없다. 실제 코퍼스 단계는 raw 재다운로드가 필요하다.

## 3. 변형(Variant) 정의

```yaml
# variants/<id>.yaml  (예시 스키마, 값은 설명용)
variant_id: ko_filter_v2_a
parent: ko_filter_v1
filters:
  check_ko:
    min_chars: 200
    min_hangul: 50
    min_hangul_ratio: 0.30
    max_dup_line_frac: 0.30
    max_ngram_rep: 0.30
    min_alnum_ratio: 0.50
  normalization: nfc          # 명시적 정책 (NFD 입력 처리)
pii:
  rules: [email, kr_phone, kr_rrn, kr_alien_reg]
  validators: {kr_rrn: checksum_and_date}
dedup:
  exact: nfc-sha256-v1
  near: null                  # 추후 minhash 후보를 여기에 추가
```

- 정책 해시 = 설정 + 필터/마스킹 코드 SHA. `clean.py`의 `.done` 판정에 사용한다.
- 변형은 항상 `parent`와 함께 기록해 비교 가능성을 보존한다.

## 4. 3단계 측정

### T0 — CPU, gold set 기반 (이 세션에서 실행 가능)

- **Gold set 포맷** (JSONL): `{id, text, source_kind, label: keep|reject, reject_reason?, pii_spans:[{start,end,type}], notes}`. `source_kind`는 `synthetic` 또는 `real_reviewed`로 분리한다.
- **구성:** (a) 합성 seed — 규칙별 경계 사례(한글 비율 경계, 길이 경계, 코드 섞인 글, 존댓말 짧은 글, 반복, NFD, 날짜·일련번호 형태의 오탐 유발 문자열, 전화·주민번호·외국인번호 변형). (b) 소량 `real_reviewed` — 실제 공개 한국어 코퍼스 표본을 JUN이 검수(JUN 합의: 합성 seed + 소량 검수).
- **지표:** 규칙별 precision/recall/F1, PII span 단위 precision/recall, 거절 사유 혼동행렬. **합성 분모와 실제 분모를 분리 보고**하고 양성 0이면 null로 보고한다(`probe_similarity_recall` 방식). 합성 수치를 실제 recall로 주장하지 않는다.
- **sweep:** 임계값 격자에서 precision-recall 곡선과 변경에 민감한 구간을 출력한다.
- **회귀 게이트:** 채택된 변형의 gold set 결과를 고정하고 이후 변형이 악화하면 실패시킨다.

### T1 — 학습 없음, 분포·proxy 통계 (일부 가능)

- 소스별 reject 비율과 사유 분포, 문서 길이·한글 비율 분포, 유지된 문자 비율(`probe_filters.py` 확장).
- 중복: exact 제거율, 근사 중복 후보 수, 클러스터 크기 분포, split 거대 컴포넌트 크기와 eval 비율 편차.
- 토큰: 카테고리별 bytes/chars per token, 어절당 토큰 수(fertility), EOS 개수 대조.
- 한국어 품질 분류기나 KenLM 같은 외부 proxy가 존재하는지는 **불확실**. 먼저 조사하고 있으면 별도 항목으로 추가한다. 없으면 T1은 위 통계로 제한한다.

### T2 — GPU, 소규모 proxy 모델 ablation (설계만, 실행 불가)

- 같은 모델·시드·토큰 예산에서 데이터 변형만 바꿔 학습하고 `eval/perplexity.py`와 `eval/multiple_choice.py`로 평가한다.
- FineWeb 계열 ablation은 약 1.7B 파라미터에 약 28B 토큰 규모였다고 보고되었다(검색 결과 기준). 그보다 훨씬 작은 모델에서는 벤치마크가 무작위 수준에 가까워 신호가 안 날 수 있다. 사용 가능한 한국어 벤치마크가 해당 규모에서 무작위 대비 유의하게 높은지 먼저 확인해야 하며, 이는 **불확실**이다.
- FineWeb 팀은 직관적 품질이 downstream 성능과 항상 일치하지 않는다고 보고했다. T0 개선이 T2 개선을 보장하지 않는다.
- 시드 반복으로 잡음 바닥(noise floor)을 먼저 측정하고, 채택 기준을 학습 전에 정해 둔다(사후 기준 변경 금지).

## 5. 반영·수정 절차

1. 변형 YAML 작성 → 정책 해시 생성 → **새 출력 트리** 생성 (`runs/…/variants/<id>/`).
2. T0 자동 보고서, T1 통계 → `experiments/results/<variant>/{contract.json,report.json}`에 기록.
3. 채택 기준 평가 (사전 등록된 기준):
   - T0: 목표 규칙의 recall 개선이 다른 규칙의 precision을 사전 허용치 이상 훼손하지 않을 것.
   - T1: reject 비율·유지 문자 비율의 변화가 설명 가능한 범위일 것.
   - T2(가능한 경우): 잡음 바닥을 넘는 개선.
4. 결과 기록: 채택·기각·보류와 사유, 측정 불가 항목을 같은 report에 남긴다. 기각된 변형도 삭제하지 않는다.
5. 채택된 변형만 다운스트림(dedup → split → tokenize → pilot_mix) 계약 재생성 대상으로 승격한다. 승격은 별도 승인 단계다.

## 6. 필요한 코드 변경 (모두 승인 후)

| 변경 | 범위 | 영향 |
|---|---|---|
| 임계값을 설정으로 분리 | `data/filters.py`, `data/clean.py` | 기존 테스트, 코드 SHA 계약 |
| `clean.py` 입력·정책 서명 | `data/clean.py` | `.done` 판정 변경, 기존 산출물 재검증 정책 필요 |
| T0 하네스 신규 모듈 | 신규 파일 | 기존 파일 비수정으로 시작 가능 |
| `probe_filters.py` 확장 | `data/probe_filters.py` | 소규모 |
| 특수 토큰 문자열 차단 | `data/tokenize_shards.py`, `data/parallel_tokenize.py` | 토큰 계약 재생성 필요 |
| 코드 SHA 재산출 여부 확인 | `scripts/audit_retention_recovery.py`, `scripts/prepare_public_recipe.py` | 영향 범위 **불확실**, 먼저 조사 |

권장 시작점: 기존 파일을 건드리지 않는 T0 하네스(신규 모듈)부터 시작해 규칙별 정확도를 실제로 측정하고, 그 결과로 설정 분리의 우선순위를 정한다.

## 7. 구현 현황과 이 세션에서 확인된 것 (T0)

- 구현: gold 스키마, `check_ko` 파라미터화 복제본(패리티 게이트: 거절 사유 문자열 일치, 퍼징 포함), `mask_pii` 재현과 후보 PII 규칙(마스킹 결과 바이트 일치 게이트), 정책 flip, 임계값 sweep, 실제 샘플 추출·sha256 검증 워크플로, 테스트 35개.
- 설계 단계에서 수정한 점: 합성 gold의 임계값 sweep은 최적값 근거가 아니라고 리포트가 스스로 출력한다. 규칙으로 표현할 수 없는 거절(`semantic_low_quality`)은 precision/recall 분모에서 분리한다.
- 실제 파일럿(n=20)의 가설 "이메일 마스킹은 새로 지우는 개인정보가 거의 없다"는 T1(16만 문서)에서 **소스별로 갈렸다**: FineWeb-2는 이미 익명화(실제 이메일 0.08%), HPLT는 실제 이메일이 문서의 11.5%. 소표본 파일럿의 일반화가 틀릴 수 있다는 예다.
- 이후 구현: 정책 객체(`KoPolicy`/`PiiPolicy`)로 하네스가 규칙 복제본 없이 프로덕션을 호출, 골든 오라클, T1 군집 통계, 3소스 층화 100건 블라인드 라벨, 검수 시트. 결과와 한계는 [IMPROVEMENTS.md](IMPROVEMENTS.md).

## 8. 이 세션에서 할 수 없는 것

- 실제 코퍼스 규모 측정, T2 학습, 모델 성능 기반 결론.
- 외부 도구(datatrove, NeMo Curator)와의 동일 입력 성능 비교(설치·데이터 필요).
