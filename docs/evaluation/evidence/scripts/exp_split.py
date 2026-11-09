import sys, json, shutil
from pathlib import Path
sys.path.insert(0,'.')
import pyarrow as pa, pyarrow.parquet as pq
from data.split_manifest import build, SplitSelection
root=Path(sys.argv[1]); shutil.rmtree(root,ignore_errors=True); root.mkdir(parents=True)
def mkfile(name,table):
    f=root/name; pq.write_table(table,f); Path(str(f)+'.done').touch(); return f
N=200
# code: N repos, 3 unique files each; ONE boilerplate file ('' / LICENSE) in 30% of repos
texts=[];repos=[]
for r in range(N):
    for k in range(3): texts.append(f'def f{r}_{k}(): pass'); repos.append(f'org/repo{r}')
    if r%10<3: texts.append('MIT License\n'); repos.append(f'org/repo{r}')
code=mkfile('code.parquet',pa.table({'text':texts,'repo_name':repos}))
web_t=['Hello World','hello world','Hello  World','café one','café one']+[f'w{i}' for i in range(300)]
web=mkfile('web.parquet',pa.table({'text':web_t}))
marker=root/'s.done'; marker.touch()
srcs={'web':dict(files=[str(web)],text_policy='nfc-web-exact-v1',completion_marker=str(marker)),
      'code':dict(files=[str(code)],text_policy='raw-utf8-code-v1',completion_marker=str(marker))}
man=root/'sources.json'; man.write_text(json.dumps(dict(schema_version=1,status='ready',sources=srcs)))
for seed in (1,2,3,4):
    rep=build(man,root/f'split{seed}',.1,seed)
    s=SplitSelection(root/f'split{seed}/manifest.json')
    top=s.db.execute('SELECT group_id,COUNT(*),COUNT(DISTINCT repo),split FROM docs GROUP BY group_id ORDER BY 2 DESC LIMIT 2').fetchall()
    web_rows=s.db.execute("SELECT row,group_id,split FROM docs WHERE source='web' AND row<5 ORDER BY row").fetchall()
    print('seed',seed,rep['counts'],'largest comps',[(g[:14],n,r,sp) for g,n,r,sp in top])
    print('  web rows0-4 (Hello World/hello world/Hello  World/café NFC/café NFD) groups distinct:',len({g for _,g,_ in web_rows}),[sp for *_,sp in web_rows])
    s.close()
