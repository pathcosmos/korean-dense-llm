# dataqa — T0 데이터 처리 QA 하네스

정제 필터와 PII 규칙을 gold set으로 측정하는 CPU 전용 도구다. 설계 배경은 [docs/evaluation/FEEDBACK_LOOP_DESIGN.md](../docs/evaluation/FEEDBACK_LOOP_DESIGN.md)의 T0 단계다.
**신규 모듈이며 `data/`, `common/` 등 기존 파일은 수정하지 않는다.** 산출물을 승격하거나 데이터를 바꾸지 않는다.

## 구성

| 파일 | 역할 |
|---|---|
| `gold.py` | gold 레코드 스키마·검증, 정책 override (`apply_policy`) |
| `ko_rules.py` | `data.filters.KoPolicy`/`check_ko`를 호출하는 얇은 래퍼 (복제본 없음) |
| `pii_rules.py` | 원문 좌표 매핑 + 프로덕션 `find_pii`/`PiiPolicy` 호출 + 검증 불가 사실에 의존하는 하네스 전용 후보 + 패리티 게이트 |
| `oracle.py` | `data/filters.py` 기본 동작을 동결하는 골든 오라클 생성 (`tests/golden/filters_oracle_v1.json`) |
| `metrics.py` | 필터·PII 지표. 분모 0이면 `None`, 합성/실제는 합치지 않음 |
| `seed_gold.py` | 결정적 합성 seed (`gold/seed_v1.jsonl`, 94건) |
| `run.py` | `report.json` + `report.md` 생성, 임계값 sweep |
| `real_sample.py` | 고정 revision에서 무작위 실제 문서 추출·라벨 결합 (원문은 커밋하지 않음) |
| `real/` | 실제 샘플 manifest와 라벨 (텍스트 없음) |

## 사용법

저장소 루트에서, 의존성은 `requirements-cpu.txt` + CPU torch (이 하네스 자체는 torch를 쓰지 않음).

```bash
python -m dataqa.seed_gold --out dataqa/gold/seed_v1.jsonl
python -m dataqa.run --gold dataqa/gold/seed_v1.jsonl --out runs/dataqa/seed_v1
python -m dataqa.run --gold dataqa/gold/seed_v1.jsonl --out runs/dataqa/seed_v1_nfd_reject --policy nfd_text_is_valid_korean=reject
python -m pytest -q tests/test_dataqa.py
```

출력 디렉터리가 비어 있지 않으면 거부한다. 리포트에는 gold sha256, 코드 identity, 파라미터, 정책 override가 기록된다.

## 신뢰를 지키는 장치

1. **복제본 제거 + 골든 오라클.** `data/filters.py`가 `KoPolicy`/`PiiPolicy`를 노출하므로 하네스는 규칙 로직을 복제하지 않는다. 기본값이 리팩터링 전과 **바이트 단위로 같음**은 `tests/golden/filters_oracle_v1.json`(수정 전 코드에서 생성, 합성 seed·시드 고정 퍼징·로컬 파일럿의 거절 사유·해시·마스킹 sha256)이 고정한다. 하네스 전용 PII 후보(접두 허용 목록, 7번째 자리 5-8)만 하네스에 남아 있고, 이 후보들의 마스킹 결과는 `apply`가 `mask_pii`와 같은 순서로 재현하는지 `parity_mismatches`로 검사한다. 하나라도 어긋나면 `run`은 리포트를 쓰지 않고 종료한다.
2. **파이프라인 순서.** `data/clean.py`의 문서별 순서(`norm_ws → check → exact_hash → mask_pii`)를 따른다. 문서 간 `exact_dup`은 측정하지 않는다.
3. **분모 규율.** 모든 비율에 분모를 함께 적고, 분모 0은 `None`이다. `ambiguous` 레코드와 span은 precision/recall에서 제외하고 개수만 센다.
4. **정책 선택 분리.** 정책에 따라 달라지는 라벨(NFD 문서, 코드가 섞인 튜토리얼, 후렴 있는 시)은 `policy_assumption`으로 표시하고 `--policy 이름=keep|reject`로 뒤집는다. 다른 라벨은 건드리지 않는다.
5. **의미적 거절 분리.** 임계값 규칙이 표현할 수 없는 거절(광고성 상품 목록, 기계번역 스팸 등)은 `semantic_low_quality`로 라벨하고 규칙 필터의 precision/recall 분모에서 제외한다. 규칙이 몇 건을 우연히 잡았는지만 따로 센다.

## 읽는 법과 한계

- **합성 seed 결과는 하네스와 규칙이 설계대로 반응한다는 확인이다.** 작성자가 케이스와 라벨을 모두 정했고, PII 후보 규칙도 같은 작성자가 만들었다. 임계값 근처에 일부러 놓은 `boundary_by_construction` 케이스가 있어, sweep에서 값을 옮기면 판정이 뒤집히는 것은 최적값의 근거가 아니다. 리포트가 이 문구를 스스로 출력한다.
- 후보 PII 규칙의 사실 주장(전화번호 접두 허용 목록, 주민등록번호 7번째 자리 5-8, 생년월일 범위)은 **검증하지 않았다 (불확실)**. 주민등록번호 검증자리는 2020년 번호 체계 변경 이후 발급분에 적용되는지 불확실해 일부러 구현하지 않았다. 도입 전에 공식 자료로 확인해야 한다.
- 대상은 `check_ko`와 `mask_pii`뿐이다. `check_en`, `check_math`, 코드 정제, 중복 제거, split은 다루지 않는다.
- 합성 텍스트는 템플릿으로 만든 문장이라 실제 웹 텍스트의 분포를 대표하지 않는다.

## 실제 샘플 워크플로 (T0의 실제 분모)

```bash
echo 'dataqa/review/' >> .git/info/exclude          # 원문 시트는 git에서 제외 (tracked 파일을 건드리지 않음)
python -m dataqa.real_sample sample --n 20 --seed 1  # manifest(커밋 가능) + 로컬 시트
# dataqa/review/review.md 를 보며 dataqa/real/pilot_labels_v1.jsonl 작성 → 검수
python -m dataqa.real_sample materialize              # 텍스트를 다시 받아 sha256 검증 후 로컬 gold 생성
python -m dataqa.run --gold dataqa/review/real_gold.jsonl --out runs/dataqa/real_pilot
```

- 추출 대상: `docs/data_provenance_v1/source_registry.json`에 고정된 repository와 revision. 샤드·row group·행을 시드로 무작위 추출한다(0번 샤드·0번 row group 편향 방지).
- **원문은 커밋하지 않는다.** 실제 웹 텍스트에는 실제 개인정보가 있을 수 있다. 커밋 대상은 manifest(repo, revision, file, row_group, row, sha256)와 라벨(오프셋·유형·라벨)뿐이고, 로컬 시트는 Artifact 등 외부에 게시하지 않는다.
- 지원 소스는 `fineweb2_ko`뿐이다. HPLT와 wiki는 컬럼 구성이 달라 아직 지원하지 않는다(미검증). 소스별 층화 추출이 필요하면 리더 확장이 선행 작업이다.
- 원 파이프라인이 실제로 받은 25개 샤드가 정확히 어느 것인지는 registry에 개수만 있다. `data/download.py`가 정렬된 목록의 앞 N개를 취하므로 같은 규칙으로 추출하지만 **이 동일성은 가정이다 (불확실)**.

## 파일럿 결과 (n=20, 에이전트 사전 라벨, 사람 검수 전)

`dataqa/real/pilot_manifest_v1.jsonl` (seed 1, FineWeb-2 `kor_Hang`, revision `af9c1333…`). n=20이므로 비율은 구간으로만 읽는다 (Wilson 95%).

| 관찰 | 건수 | 95% 구간 | 해석 |
|---|---|---|---|
| 현행 필터가 거절 (`short`) | 2/20 | 3–30% | 두 건 모두 사전 라벨과 일치 |
| 사전 라벨이 거절하려는데 규칙이 못 잡음 (`semantic_low_quality`) | 2/20 | 3–30% | 상품 목록·푸터 위주 페이지와 기계번역 시장보고서 스팸. 규칙 필터의 설계 범위 밖 |
| 입력에 이미 이메일 플레이스홀더가 있음 | 4/20 | 8–42% | 입력 데이터가 이미 이메일을 치환한 상태. 이 저장소의 이메일 마스킹이 실제로 새로 지우는 개인정보가 거의 없을 가능성. 단 소표본 |
| 개인 휴대전화번호 (진짜 PII) | 1/20 | 1–24% | 기본 전화 정규식이 탐지 |
| 기관 대표번호 | 1/20 | 1–24% | 개인정보인지 정책 판단 필요 → `ambiguous` |
| 사람 이름·주소·사업자번호가 본문에 있고 어떤 규칙도 다루지 않음 | 5/20 | 11–47% | 라벨은 태그(`has_person_name` 등)로만 표시했고 span 라벨링은 하지 않았다. 이름을 어디까지 개인정보로 볼지가 정책 결정이기 때문 |
| 판단 보류 | 1/20 | — | 활동 목록과 로그인 안내가 섞인 페이지 |

비정상 가정이 많은 소표본이다. 위 표로 필터의 정밀도·재현율이나 PII 발생률을 주장할 수 없다. 이 파일럿은 워크플로가 동작함과 JUN이 어디를 봐야 하는지를 보여 주는 용도다.

## 사람 검수에서 볼 것 (파일럿 기준)

로컬 시트 `dataqa/review/review.md`에서 아래 문서만 보면 된다. 규칙과 사전 라벨이 일치한 나머지는 무작위 대조군으로 몇 건만 훑으면 충분하다.

| 문서 id | 확인할 질문 |
|---|---|
| `real-b31675404383` | 상품 목록·푸터 위주 페이지를 거절(`semantic_low_quality`)하는 게 맞는가 |
| `real-d95e8aaee59d` | 기계번역 시장보고서 스팸 거절이 맞는가 |
| `real-1563566411b2` | 짧은 상품 설명(301자)을 유지해도 되는가 (경계 사례) |
| `real-da62a9ae8013` | 판단 보류(`ambiguous`)를 keep/reject 중 어느 쪽으로 볼 것인가 |
| `real-ad77a00ce258` | 개인 휴대전화번호 span이 맞는가 |
| `real-17f93f792ab2` | 기관 대표번호를 개인정보로 볼 것인가 (현재 `ambiguous`) |
| 이름이 있는 5건 (`has_person_name` 태그) | 기사·공지에 실린 실명을 마스킹 대상으로 볼 것인가 (정책 결정) |

검수가 끝나면 `label_source`를 `human_review`로 바꾸고 에이전트 라벨과의 일치율을 보고하는 것을 권한다. 일치율이 낮으면 사전 라벨을 믿고 규모를 키울 수 없다.
