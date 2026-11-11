# Pretraining progress: measured probe, not final evaluation

## 한국어 요약

2026년 11월 11일 기준 3B 모델의 본학습 목표는 50,000스텝이며 최종 평가 전입니다.
아래는 동일한 6개 소스·약 157만 토큰의 고정 평가 손실로, 낮을수록 좋습니다.
이 수치는 전체 원문 PPL이나 공식 벤치마크가 아닙니다. 고정 생성 사례에서는
산술 오답과 부정확한 설명이 여전히 관찰됐으며, 손실 감소가 실용 품질을
보장하지는 않습니다. 최종 저장·복원·평가 및 추론 패키징은 미완료입니다.
`release/source_inventory.json`과 과거 검증 보고서는 v0.1.1 공개 당시 기록이며,
현재 파일 목록은 `release/files.json`을 참고하세요.

## English

As of 11 November 2026, the 3B Dense training run is in progress with a target
of 50,000 steps. The original 57,000-step configuration mentioned elsewhere
in the v0.1.1 documentation describes the earlier release snapshot.

The table reports next-token negative log-likelihood on the same fixed probe:
six sources, 384 windows and 1,572,864 token targets. Lower is better. This is
not a full-corpus perplexity measurement or an official capability benchmark.

| Step | Overall | Korean | English | Math | Code |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 5.2304 | 5.5783 | 5.3361 | 4.9598 | 4.3517 |
| 5,000 | 2.2504 | 2.5056 | 2.7024 | 2.1638 | 1.1190 |
| 10,000 | 2.1002 | 2.3325 | 2.5578 | 2.0313 | 1.0145 |
| 15,000 | 2.0352 | 2.2605 | 2.4900 | 1.9701 | 0.9696 |
| 20,500 | 1.9912 | 2.2058 | 2.4519 | 1.9356 | 0.9424 |
| 21,000 | 1.9893 | — | — | — | — |

The six source probes contribute equal numbers of tokens; Korean combines
three sources. At 20,000 steps, fixed example generations still contained
arithmetic errors and an inaccurate explanation. These few examples are not a
representative benchmark. Falling loss is evidence of improved prediction on
this probe, not proof of instruction-following quality or factual reliability.
Final native evaluation, restoration and inference packaging remain pending.
