"""Check complete bilingual roots and13language README navigation. No network writes."""
from pathlib import Path
import json,re
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'docs/multilingual/manifest.json').read_text())
for code in m['languages']:
    p=root/'cards'/('README.'+code+'.md');s=p.read_text()
    assert len(s)>2500,p
    assert s.count('## ')>=10,p
    for target in m['languages']:assert 'cards/README.'+target+'.md' in s,(p,target)
    for repo in m['hf_repositories']:assert repo in s,(p,repo)
    for link in re.findall(r'\]\(([^)]+)\)',s):
        prefix=m['github']+'/blob/main/'
        if link.startswith(prefix):assert (root/link[len(prefix):].split('#')[0]).exists(),link
s=(root/'README.md').read_text()
assert s.index('## 한국어')<s.index('## English')
assert (root/'README.en.md').read_text()==(root/'cards/README.en.md').read_text()
print(json.dumps({'status':'passed','languages':len(m['languages']),'readmes':15,'scope':'Local language structure, same-language navigation and GH targets'}))
