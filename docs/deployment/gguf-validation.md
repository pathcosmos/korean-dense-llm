# GGUF validation: experimental preparation

## 한국어 요약

이 검사는 실험 단계입니다. 제공받은 GGUF와 HF 설정을 사용해 텐서 구조와
F32 가중치 값을 비교하며, 입력 인증과 자원 관리는 호출자의 책임입니다.
작은 임의 모델로만 시험했으며 최종 3B 모델, 토크나이저 동등성, logits 및
Ollama 로딩·생성의 성공을 의미하지 않습니다.

## English

The GGUF comparison helpers in `scripts/ollama_gguf_structure.py` and
`scripts/ollama_gguf_tensor_values.py` are research tools. They accept a caller-provided
GGUF reader and Hugging Face configuration and check the project's untied Llama
tensor layout and exact F32 tensor values. The Q and K comparison accounts for
the RoPE row permutation used by the pinned converter examined in this project.

The caller must authenticate the input files and implementation before and
after running the checks, and must ensure sufficient CPU, memory and storage.
These helpers do not load a model in Ollama, prove tokenizer equivalence, compare
logits or publish a model. A successful structural or value comparison does not
mean that a 3B model has passed final validation.

The checks have been exercised with a small random model, not the final trained
model. See [the training progress note](../results/pretraining-progress.md) for
the scope of current results. The operational conversion and publication path
is still being developed.
