# FineWeb-2 필터 비교 스크립트 (as run)

`evidence/06_datatrove_comparison.md`의 수치를 만든 스크립트다. 세션에서 실행한 그대로이며 환경 경로가 일부 남아 있을 수 있다.

- 업스트림 쪽은 별도 venv(datatrove 0.10.1, kiwipiepy 0.21.0)에서, 이 저장소 쪽(`repo_ko.py`)은 저장소 venv에서 실행한다.
- 입력 텍스트(`sample_b.jsonl`, `gold100_text.jsonl` 등)는 `sample_b.py`와 로컬 gold에서 만들어지는 **실제 웹 텍스트라 커밋하지 않았다.** 스크립트와 `analysis_out.md`(집계 수치)만 있다.
- `fw2_kor_Hang.yml`은 FineWeb-2 저장소의 공개 설정 사본이다.
- 재구성의 정당성: `upstream_fw2.py`의 필터 인자는 `fineweb-2-pipeline.py`(커밋 d0defb24) 원본과 대조해 일치함을 확인했다. 이 환경에서 FineWeb-2가 실제로 쓴 datatrove 버전은 **불확실**하다.
