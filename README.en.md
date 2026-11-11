# Korean Dense LLM — English

> **Training update (2026-11-11):** The current target is 50,000 steps; training and final evaluation are still in progress. The 57,000-step configuration below describes the v0.1.1 release snapshot. [Progress and limitations](docs/results/pretraining-progress.md) · [GGUF validation preparation](docs/deployment/gguf-validation.md)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## Overview

This release publishes the data preparation, model, training, evaluation, recovery and optimization code used to build a Korean-centered dense LLM. It contains source and experiment records, excluding corpus text, token binaries, databases and model weights. Separate HF data releases are linked below.

Translation languages do not imply training-data coverage in those languages. The principal natural languages are Korean and English; web, math and code may contain other languages.

## Contents and scale

| | |
|---|---:|
| common/, data/ | Processing and splits |
| tokenizer/ | Processing and splits |
| model/, train/, eval/ | Model and training |
| scripts/ | Optimization and benchmarks |
| tests/, examples/ | Validation |
| experiments/ | 47 / 99 |


## Processing and splits

Web processing applies whitespace cleanup, length/repetition/language heuristics and selected PII-pattern masking. Initial within-shard hashes differ from final web NFC SHA256. Code filtering preserves case, indentation and original Unicode while checking paths, license labels, generated code and selected secret patterns. Connected groups of identical bodies, NFC-equivalent web bodies and code repositories are assigned to train/eval.

## Usage

Run commands below from the repository root. Use new output/download directories and check scope, size and pinned revision first. The synthetic CPU example downloads no external data. Select only the needed files from the separate data releases and verify sizes/SHA256 against CHECKSUMS.json or the source manifests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## Model and training

The model is a plain PyTorch dense decoder with RMSNorm, RoPE, GQA and SwiGLU. The actual3B path uses FSDP2,8GPUs and eager execution: context4096, micro batch4, global batch2,097,152tokens, peak LR3e-4 and target57,000steps. The CPU micro demonstration differs from distributed training. Exact resume requires model, optimizer, RNG and input/configuration contracts.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## Optimization and benchmarks

Experiments compare workers, threads, batches, SQLite caches, RAM, local/NFS storage, eager/compile execution, precision and GPU memory. Total time, throughput, peak memory and content/numerical equivalence accompany adoption or rejection decisions.47 campaigns and99 result JSON files are included. Speed alone does not justify a numerically failing candidate, and a small sample does not establish a full-scale optimum.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## Validation

The GH source release passed284 independent CPU tests with9 skips; fresh GitHub CPU CI passed258 with35 skips. Skips require original-run records or a second filesystem. The synthetic quickstart uses120documents, train100/eval20, vocab272 and a108,864parameter model, verifying exact model/optimizer resume. Dataset content-hash, split and reader checks are separate from full independent3B reproduction.

## Sources and licensing

Original project code is MIT and original prose is CC BY4.0. These grants do not replace original-content rights. Consult source.json, LICENSE-SCOPE.md and provenance documents for source-specific conditions, attribution, notices and known unresolved items. Unrestricted commercial redistribution of every original text is not asserted.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## Limitations

Global near-duplicate removal, complete PII/secret removal, contamination-free official evaluation and final model quality are not guaranteed. Main training was ongoing at the publication snapshot; SFT, model export and official language benchmarks are not completed features.11 historical code references/9unique hashes lack a recovered snapshot. Some operational tools require original-run manifests.

## Related repositories

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / Usage](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / Optimization and benchmarks](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / Validation](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/cards/README.en.md)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data/blob/main/cards/README.en.md)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia/blob/main/cards/README.en.md)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2/blob/main/cards/README.en.md)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2/blob/main/cards/README.en.md)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu/blob/main/cards/README.en.md)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math/blob/main/cards/README.en.md)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code/blob/main/cards/README.en.md)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
