# Korean Dense LLM

A source-only research release of a Korean-centered dense language-model project: data acquisition and cleaning, deduplication, audited splits, train-only tokenization, a PyTorch model/FSDP2 trainer, evaluation, recovery, and measured optimization experiments.

[한국어](README.md) · [Guide](docs/PUBLIC_GUIDE.md) · [Experiments](experiments/README.md) · [Validation](docs/PUBLIC_VALIDATION.md)

Version 0.1.0 publishes code and the observations available at the snapshot date. It does not bundle corpus text, token binaries, SQLite databases, checkpoints, or tokenizer vocabulary. Main training was still running at the snapshot; this is not a claim of final model quality, completed official benchmarks, SFT, or a supported Hugging Face model export.

## Verified checks

Independent local suite:284 passed/9 skipped. Fresh GitHub CPU environment:258 passed/35 skipped. Skips require original-run manifests or a second filesystem. An unauthenticated clone passed manifest verification and the end-to-end quickstart. See the [validation evidence](docs/PUBLIC_VALIDATION.md).

## CPU quickstart

From the repository root, with Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```

The offline example generates synthetic input and runs cleaning, exact deduplication, audit/finalization, grouped splitting, a small train-only BPE, tokenization, micro-model training/evaluation and exact model/optimizer resume. Use a new `--output-dir runs/demo` to retain outputs; otherwise temporary outputs are cleaned up.

Measured package versions and the distinction between an isolated source checkout and a fresh package installation are in [release/tested_environment.json](release/tested_environment.json). Original-run integrations are explicitly skipped without the excluded manifests. Retention tests require separate filesystems; missing prerequisites are reported as skips, not passes.

The original trainer and real measured 3B settings are included. Read the guide before a GPU run: the operational wrappers are bound to explicit manifests/contracts and some are historical, run-specific research utilities. The quickstart's simple CPU checkpoint is distinct from the FSDP distributed checkpoint path, which has separate CPU DCP tests and historical GPU evidence.

Original code: MIT. Original prose: CC BY 4.0. Third-party materials and data retain their own terms; see [NOTICE](NOTICE.md).
