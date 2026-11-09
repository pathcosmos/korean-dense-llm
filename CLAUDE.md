# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **source-only** release of a Korean-centric dense LLM pipeline (data prep → tokenizer → pure-PyTorch decoder → distributed training → eval → checkpoint retention/recovery → optimization experiments). It ships **no data bodies, token binaries, DBs or weights**; data lives in separate Hugging Face dataset repos (`pathcosmos/korean-pretrain-*`, training input pinned at `pathcosmos/korean-pretrain-training-data` tag `v1.0.1`, ~242GB). Docs are bilingual KO/EN (README.md / README.en.md) with 13-language cards in `cards/`.

The real 3B run (FSDP2, 8 GPU, eager, ctx 4096, global batch 2,097,152 tokens, peak LR 3e-4, 57,000 steps) cannot be reproduced on a dev/CPU box. CPU quickstart passing ≠ 8-GPU reproduction.

## Commands

Run from the repo root. CI (`.github/workflows/cpu.yml`) uses Python 3.12 and CPU torch 2.7.0:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python examples/quickstart_cpu.py          # synthetic 120 docs, tiny model, save/resume check
python -m pytest -q tests                  # full suite
python -m pytest -q tests/test_dedup.py::test_name   # single test
```

- `requirements.txt` is the full GPU/data-pipeline stack (transformers, datasets, fasttext, wandb…); `requirements-cpu.txt` is the minimal set CI uses. `requirements-bench.txt` adds matplotlib for experiment summaries.
- Expected CPU result: roughly 258 passed / 35 skipped on a fresh CPU env (284/9 on the original box). Skips are intentional — see `tests/conftest.py`: original-run integration tests need `--run-original-integrations` plus restored production manifests, and the tiered-retention tests need `TMPDIR` on a different filesystem than the repo.
- `tests/conftest.py` refuses to run if `runs/` is a symlink; `runs/*` is gitignored. Do not point it at production output.
- GPU training entry: `torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json` (preceded by `python scripts/hardware_profile.py --output runs/hardware.json`). `examples/3b.observed.json` is the as-run snapshot, not an auto-run default; `train/configs/3b.json` (compile-on, micro 8) is a stale draft — don't confuse them.
- Only the tokenizer can be fetched cheaply: `snapshot_download('pathcosmos/korean-pretrain-training-data', repo_type='dataset', revision='v1.0.1', allow_patterns=['tokenizer/*'])`. Note `AutoTokenizer.from_pretrained(<dataset_id>)` hits the model endpoint and 401s; download the dataset snapshot first, then load locally (`docs/LESSONS.md`).
- Cloud-sandbox note: HF access needs `huggingface.co` and `us.aws.cdn.hf.co` (Xet CDN for large files) on the network allowlist.

## Architecture (big picture)

Pipeline stages, each a directory; scripts in `scripts/` orchestrate/verify them:

1. **`data/`** – download (`download.py`, pinned HF revisions, selective files) → `clean.py` / `clean_code.py` (web vs. code policies differ: web is whitespace/length/repeat/language heuristics + partial PII masking and NFC; code preserves case/indent/raw Unicode and checks paths/licenses/generated/secret patterns) → `dedup.py` → group-aware **train/eval split** (`split_manifest.py`, `release_split.py`: identical text, web NFC-equal text and code-repo connected groups go to the same side) → `tokenize_shards.py`/`parallel_tokenize.py` → `build_mix.py` (weighted mix from *completed* token sources, never stale globs). `probe_*` files are leakage/overlap/recall audits.
2. **`tokenizer/`** – BPE trained only on train-side samples (vocab 65,536).
3. **`model/model.py`** – pure-PyTorch Dense decoder: RMSNorm, RoPE, GQA, SwiGLU. `get_config()` selects sizes.
4. **`train/`** – `train.py` is the trainer; `deterministic_train.py` is the **public entry point** that monkey-patches `train.training_contract` to add determinism settings (cuBLAS workspace, deterministic algos) and its own code hash to the contract id. `data.py` holds the sampler (`SAMPLER_VERSION` is part of the contract). `monitored.py`, `rehearsal.py`, `final_resume_probe.py`, `parallel_contract.py` support supervised runs and resume validation. Configs: `train/configs/` and `examples/`.
5. **`eval/`** – `run.py` computes checkpoint perplexity and fixed-prompt generation (DCP checkpoints) from `{category: ko|en, text}` JSONL; prompts need unique ids. Inputs must come from the chosen holdout snapshot.
6. **`common/`** – `data_contracts.py` is the core: manifest schema v1, `identity()` content hashing, `sha256_file`, `atomic_json`, `resolve`. `paths.py` defines `PROJECT_ROOT` (override with env var `PROJECT_ROOT`). `source_inventory.py` backs `release/source_inventory.json`.
7. **`experiments/`** + `scripts/benchmark_*`, `validate_*`, `probe_*` – 47 campaigns / 99 result JSONs (worker/thread/batch/SQLite cache/NFS, eager vs compile, precision) with `code_bindings.json` tying results to code SHAs. Candidates that fail numeric-equivalence are recorded as rejected even if faster.
8. **Checkpoint retention/recovery** – `scripts/tiered_checkpoint_retention.py`, `audit_checkpoint_retention.py`, `execute_reviewed_eviction.py`, `docs/CHECKPOINT_RETENTION.md`, `docs/DATA_RETENTION_AND_RECOVERY.md`.

### Cross-cutting conventions

- **Contract/manifest-driven**: stages exchange JSON manifests with content hashes and completion markers (`.done`); consumers verify them (resume requires model + optimizer + RNG + input/config contract match). When changing a stage's output format, update the contract/identity and version constants, not just the producer.
- Code style is dense and compact (semicolon-joined statements, minimal whitespace); match the surrounding style when editing.
- `release/*.json` (files, validation, public_changes, tested_environment) and `docs/multilingual/*` are generated/verified artifacts; `scripts/check_multilingual_readmes.py` and `scripts/update_public_corpus_card_links.py` maintain the 13-language cards. Editing README/cards by hand can desync these checks.
- Many `scripts/` (supervise_*, observe_*, finalize_*, upload_*, run_final_*) are operational tools for the original run and need original manifests/paths that aren't in this checkout. `docs/RUNBOOK.md`, `LESSONS.md`, `FRANKENSTALLM_*` are historical records; read `docs/PUBLIC_GUIDE.md` → `PUBLIC_VALIDATION.md` → `experiments/README.md` first for current status.
- Provenance/licensing: `docs/DATA_SOURCES_AND_LICENSES.md`, `docs/data_provenance_v1/source_registry.json` (pinned upstream revisions: Wikipedia ko, FineWeb2 kor_Hang, HPLT2.0 cleaned, FineWeb-Edu 100BT first 56 shards, OpenWebMath, CodeParrot github-code 64 shards). Code is MIT, original prose CC BY 4.0; this does not relicense upstream data. Don't claim unconditional commercial redistribution rights, global near-dup removal, full PII removal, or benchmark-contamination-free status.
- Known limits stated by the project: main pretraining was still in progress at release; no SFT, model export, or official benchmark results.
