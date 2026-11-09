Stage-1 near-dup measurement scripts (docs/DATA_VALUE_ASSESSMENT.md sec.15, docs/NEAR_DUP_DESIGN.md sec.6).
Recorded as run in a sandbox session: paths are hard-coded to /tmp/claude-0/scratch, they read the public HF dataset
`pathcosmos/korean-pretrain-corpus@v1.1.1`, and they are not wired into tests or release manifests.
nd_lib: MinHash signature (NFKC lowercase, word 5-gram, 128 perms, max 5000 words; hashes utf-8 bytes with xxhash>=3.x).
nd_run/nd_strat: index builders; s1_measure: stage-1 analyses; s1_wiki: wiki vs web.
