# 데이터 처리 소스 가치 평가 — 결과 보고 브랜치

이 브랜치는 `pathcosmos/korean-dense-llm`의 **데이터 처리 소스 가치 평가**의 결과와 방법, 향후 계획만 담은 **문서 전용 브랜치**다.

- **PR 대상이 아니다.** 소스 이력과 이어지지 않는 독립 브랜치(orphan)이며 코드는 없다.
- 평가 기준일: 2026-11-09.
- 평가 중 수정한 소스와 측정 도구, 근거 보고서, 실험 스크립트는 작업 브랜치에 있다.

| 항목 | 위치 |
|---|---|
| 작업 브랜치 | `claude/gracious-wozniak-lrksyz` |
| 이 문서가 기준으로 삼은 커밋 | `a89c0a5` (작업 브랜치의 최종 요약 커밋) |
| 평가 원본 문서 | 작업 브랜치의 `docs/evaluation/` (README, IMPROVEMENTS, FINAL_SUMMARY, FEEDBACK_LOOP_DESIGN, `evidence/`, `t1/`) |
| 측정 하네스와 라벨 | 작업 브랜치의 `dataqa/` |
| 수정한 소스 | 작업 브랜치의 `data/filters.py`, `data/clean.py`, `data/quality_fw2.py`, `common/data_contracts.py` |

## 이 브랜치의 문서

| 파일 | 내용 |
|---|---|
| [ASSESSMENT.md](ASSESSMENT.md) | 평가 결과와 방법. 결론, 점수, 발견, 수정한 소스, 증거의 질, 평가할 수 없는 것, 재현 방법 |
| [TODO_AND_IMPROVEMENTS.md](TODO_AND_IMPROVEMENTS.md) | 향후 TO-DO와 개선점. 결정 필요 항목, 우선순위, 선행 조건, 완료 기준, 하지 말 것, 재개 방법 |

## 한 줄 결론

오픈소스·포트폴리오 자산으로서의 가치는 **중간 이상**이고, 그 가치는 정제 알고리즘이 아니라 재현성·계약·누수 방지 분할·출처 추적에 있다. 이 데이터 처리가 모델 품질을 높이는지는 **평가할 수 없다**(학습 ablation 필요).

## 읽는 순서

1. 이 문서와 [ASSESSMENT.md](ASSESSMENT.md)의 1–2절(결론과 평가 기준)
2. 결과가 궁금하면 ASSESSMENT.md 4–6절, 방법이 궁금하면 3절
3. 다음에 무엇을 할지는 [TODO_AND_IMPROVEMENTS.md](TODO_AND_IMPROVEMENTS.md)

## 주의

- 점수는 평가자 판단이고 계층 간 교정되지 않았다. 숫자의 소수점 차이에 의미를 두지 말 것.
- 라벨의 절반(50건)은 사람(JUN)이 구두로 확인했고, 나머지 50건은 에이전트 라벨과 독립 검수자 2명의 일치에 근거한다. 규칙 필터의 정확도 수치를 인용할 때는 이 구분을 함께 쓸 것.
- 실제 웹 텍스트와 개인정보는 어느 브랜치에도 커밋하지 않았다. 라벨은 오프셋과 유형만 담는다.
