"""Sample (b): 20 random row groups per source -> local jsonl (do not commit). Run from repo root with /tmp/claude-0/venv/bin/python."""
import json, sys
sys.path.insert(0, '/home/user/korean-dense-llm')
from dataqa import real_sample as R
OUT = sys.argv[1]
with open(OUT, 'w', encoding='utf-8') as f:
    for source in ['fineweb2_ko', 'hplt_ko', 'wiki_ko']:
        repo, rev = R.pinned(source)
        reader = R.Reader(repo, rev)
        shards = R.list_shards(repo, rev, R.SUBSETS[source], R.DEFAULT_SHARDS[source])
        plan = R.plan(shards, R.LazyCounts(reader), 20, seed=77)
        for gi, item in enumerate(plan):
            col = reader.group(item['file'], item['row_group']).column('text')
            for ri in range(len(col)):
                f.write(json.dumps(dict(source=source, group=f"{source}:{gi}", file=item['file'], rg=item['row_group'], row=ri, text=col[ri].as_py()), ensure_ascii=False) + '\n')
        print(source, repo, rev, len(shards), flush=True)
