# 증거 실험 스크립트

`evidence/*.md` 보고서가 인용한 합성 입력 실험이다. 저장소 루트에서 실행한다(경로는 `sys.path.insert(0,'.')`).

```bash
python -I docs/evaluation/evidence/scripts/exp1.py          # 정제 계층 (exp1~exp8)
python -I docs/evaluation/evidence/scripts/exp_dedup.py /tmp/out_dedup
python -I docs/evaluation/evidence/scripts/exp_split.py /tmp/out_split
S=/tmp python -I docs/evaluation/evidence/scripts/probe.py  # 수집·배포 계층 (probe, probe2)
```

- 의존성: `requirements-cpu.txt`와 CPU torch. 외부 데이터·네트워크는 필요 없다.
- 출력은 표준 출력과 `tempfile` 또는 인자로 준 디렉터리에만 쓴다.
- 결과는 "그런 입력이 있을 때 이렇게 동작한다"는 존재 증명이며 발생 빈도를 말하지 않는다.
- 토큰화 보고서(`03_tokenization.md`)의 실험은 인라인 실행이라 스크립트가 없다.
