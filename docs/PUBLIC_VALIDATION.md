# Public release validation / 공개본 검증

Snapshot: 2026-11-08. Source-only v0.1.0. Machine-readable evidence: [validation.json](../release/validation.json).

## Actually executed

- Independent staging checkout: **284 passed, 9 explicitly skipped, 0 failed**; 106.85s elapsed. Original source directory is not on PYTHONPATH; credential variables and GPU visibility were removed for the subprocesses.
- Offline CPU quickstart: **passed**, 4.03s, sampled group RSS peak 0.690GiB. Synthetic120 retained documents, train100/eval20, train65,211/eval13,061 tokens, vocabulary272, micro model108,864 parameters. The next update's loss, model and optimizer state exactly match after save/resume.
- Full CPU suite sampled process-group RSS peak 1.697GiB. 8-core affinity, two numerical-library threads, 6GiB group RSS guard and at least8GiB cgroup free space; no GPU workload. RSS is sampled, not an exact continuous maximum.
- Retention fixtures exercised real CPU distributed-checkpoint contents across the checkout's NFS and local TMPDIR. On a single filesystem these tests explicitly skip; set TMPDIR to a separate device for that coverage.

## Explicit limits

Nine integration tests require original training/validation checkpoint metadata or dataset-publication manifests excluded from this source-only repository. They remain in the source and are skipped by default with individual reasons. `--run-original-integrations` is only for an independently restored integration environment. Synthetic optimizer-map rejection and supervisor native-pruning rejection remain enabled; they were separated from the original-run checks.

The tested installed NVIDIA PyTorch build is recorded in [tested_environment.json](../release/tested_environment.json); this local result alone does **not** certify a fresh standard-wheel installation. The separate public CI result below verifies that installation. The quickstart uses a simple CPU checkpoint; distributed checkpoint/RNG behavior has additional CPU tests. New 8GPU/full3B training was not run for this release.

Initial packaging runs revealed missing local run-directory prerequisites and original-run dependencies; these were addressed in the public-only test setup. The initial quickstart requested more BPE merges than its small synthetic vocabulary supported and used an output path outside the portable-mix root; the final example uses vocab272 and checkout-owned outputs. Initial failures are not counted as passes.

## Historical experiments

47 campaigns and99 result JSON files are included. Numbers and scope come from original reports, not newly measured public-package performance. Of explicit code-hash references, 285 match the current source and 34 match archived code bytes. 11 references (9 unique hashes) lack a recovered code snapshot. Those cases are historical measurements with partial reproducibility, not fully reproducible claims. See [code_bindings.json](../experiments/code_bindings.json).

Original and public result hashes differ when paths/content fields are sanitized; both are in [index.json](../experiments/index.json). Original execution IDs and referenced excluded artifacts describe provenance and do not imply those inputs are bundled.

## Publication checks

The first published code commit `3ea7171e4dfc7adb6582a9fb7484c30283ea28c9` passed [fresh GitHub CPU CI](https://github.com/pathcosmos/korean-dense-llm/actions/runs/42451765321): **258 passed, 35 skipped, 0 failed**, plus the end-to-end quickstart. This used a new Python3.12 environment and standard `torch2.7.0+cpu`. The additional26 skips are the explicitly declared two-filesystem retention tests on the single-filesystem runner; the local NFS/local validation above exercised those fixtures.

An unauthenticated clone with system/global Git configuration and credential helpers disabled verified all531 manifest payload files and passed the CPU quickstart in4.03s. These checks apply to the published implementation; subsequent release-finalization changes update documentation and checksums only. Exact final tag/commit and the final workflow are in the [v0.1.0 release verification attachment](https://github.com/pathcosmos/korean-dense-llm/releases/tag/v0.1.0).
