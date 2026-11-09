"""Run from repo root with /tmp/claude-0/venv/bin/python: repo_ko.py IN.jsonl OUT.jsonl. Output has no text."""
import sys, json, re
sys.path.insert(0, '/home/user/korean-dense-llm')
from data import filters as F
H = re.compile('[가-힣]')
with open(sys.argv[2], 'w') as fo:
    for i, l in enumerate(open(sys.argv[1], encoding='utf-8')):
        t = F.norm_ws(json.loads(l)['text'])
        ns = sum(1 for c in t if not c.isspace())
        fo.write(json.dumps(dict(i=i, ko=F.check_ko(t), n=len(t), hr=(len(H.findall(t)) / ns if ns else 0))) + '\n')
