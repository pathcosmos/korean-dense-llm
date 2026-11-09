"""Re-read the same fineweb2_ko row groups (plan seed 77) for language_score only; write one score per doc in sample_b order."""
import json, sys
sys.path.insert(0, '/home/user/korean-dense-llm')
from dataqa import real_sample as R
repo, rev = R.pinned('fineweb2_ko'); reader = R.Reader(repo, rev)
shards = R.list_shards(repo, rev, R.SUBSETS['fineweb2_ko'], R.DEFAULT_SHARDS['fineweb2_ko'])
with open(sys.argv[1], 'w') as f:
    for item in R.plan(shards, R.LazyCounts(reader), 20, seed=77):
        t = reader.group(item['file'], item['row_group'], columns=('language_score', 'language')).column('language_score')
        for v in t.to_pylist(): f.write(json.dumps(v) + '\n')
