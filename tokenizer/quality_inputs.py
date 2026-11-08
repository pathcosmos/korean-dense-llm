"""Independent, complete eval documents for final tokenizer quality/efficiency checks."""
import json
from pathlib import Path
from common.data_contracts import resolve,atomic_json,sha256_file,identity,artifact_path
from data.split_manifest import text_hash

REQUIRED={'ko','en','code','math'}


def baseline_ids(frame_path,expected_sha,output):
    """Persist original row identities so comparison independence survives scratch loss."""
    path=resolve(frame_path);out=resolve(output)
    if sha256_file(path)!=expected_sha:raise ValueError('Baseline frame changed')
    frame=json.loads(path.read_text());count=0
    if out.exists():raise ValueError('Preserve existing baseline identities')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(out)+'.tmp')
    if tmp.exists():raise ValueError('Preserve partial baseline identities')
    with tmp.open('w') as stream:
        for f in frame['shards']:
            sample=resolve(f['path'])
            if sha256_file(sample)!=f['sha256']:raise ValueError('Baseline sample changed')
            with sample.open() as inp:
                for line in inp:
                    r=json.loads(line);item={k:r[k] for k in ['source','file','row','body_sha256']}
                    stream.write(json.dumps(item,ensure_ascii=False)+'\n');count+=1
    tmp.rename(out)
    receipt=dict(frame_sha256=expected_sha,source_manifest_sha256=frame['source_sha256'],identities_sha256=sha256_file(out),documents=count)
    atomic_json(str(out)+'.done',receipt);return receipt


def exclusion_groups(inputs,identities):
    path=resolve(identities);receipt=json.loads(Path(str(path)+'.done').read_text())
    if sha256_file(path)!=receipt['identities_sha256'] or receipt['source_manifest_sha256']!=inputs.context.source_sha:
        raise ValueError('Baseline identities/input changed')
    groups=set();count=0
    with path.open() as stream:
        for line in stream:
            r=json.loads(line);stored=inputs.selection.db.execute('SELECT content_sha256,group_id FROM docs WHERE source=? AND file=? AND row=?',
                                                                  (r['source'],r['file'],r['row'])).fetchone()
            if stored is None or stored[0]!=r['body_sha256']:raise ValueError('Baseline row binding differs')
            groups.add(stored[1]);count+=1
    if count!=receipt['documents']:raise ValueError('Baseline identity count differs')
    return groups


def prepare_checks(inputs,output,categories,*,per_source=1024,seed=9327,excluded_groups=None,max_utf8_bytes=128*1024**2):
    if set(categories)!=set(inputs.files) or set(categories.values())!=REQUIRED:raise ValueError('All sources and four categories required')
    if type(per_source) is not int or per_source<1:raise ValueError('Positive per-source count required')
    inputs.assert_bound(inputs.manifest,inputs.split_manifest);out=resolve(output)
    if out.exists():raise ValueError('Preserve existing quality inputs')
    out.mkdir(parents=True);target=out/'samples.jsonl';tmp=out/'samples.jsonl.tmp';excluded=set(excluded_groups or [])
    stats={};used=0;seen=set()
    with tmp.open('w') as stream:
        for source,category in categories.items():
            docs=bytes_=0
            for r in inputs.rows(source,'eval',seed):
                key=(r['file'],r['row'])
                if r['group_id'] in excluded or key in seen:continue
                n=len(r['text'].encode())
                if used+n>max_utf8_bytes:raise ValueError('Quality UTF8 budget reached; partial preserved')
                r.update(category=category,split='eval',truncated=False,text_sha256=text_hash(r['text']))
                stream.write(json.dumps(r,ensure_ascii=False)+'\n');seen.add(key);docs+=1;used+=n;bytes_+=n
                if docs==per_source:break
            if docs!=per_source:raise ValueError('Insufficient independent eval documents for '+source)
            stats[source]=dict(documents=docs,utf8_bytes=bytes_)
    inputs.assert_bound(inputs.manifest,inputs.split_manifest);tmp.rename(target)
    contract=dict(source_sha256=inputs.context.source_sha,split_manifest_sha256=inputs.context.split_sha,split_contract_id=inputs.selection.report['contract_id'],
                  categories=categories,per_source=per_source,seed=seed,max_utf8_bytes=max_utf8_bytes,
                  excluded_groups_id=identity(sorted(excluded)),code_sha256=sha256_file(Path(__file__)),scope='Whole eval documents; final-tokenizer training excluded by group; baseline comparison training groups excluded explicitly.')
    report=dict(status='prepared_independent_checks',contract=contract,contract_id=identity(contract),stats=stats,
                samples=dict(path=artifact_path(target),sha256=sha256_file(target)),excluded_groups=len(excluded))
    atomic_json(out/'manifest.json',report);atomic_json(out/'.done',dict(contract_id=report['contract_id'],manifest_sha256=sha256_file(out/'manifest.json')))
    return verify_checks(inputs,out/'manifest.json',excluded_groups=excluded)


def verify_checks(inputs,manifest,*,excluded_groups=None):
    path=resolve(manifest);r=json.loads(path.read_text());c=r['contract'];excluded=set(excluded_groups or [])
    if set(c['categories'])!=set(inputs.files) or set(c['categories'].values())!=REQUIRED:raise ValueError('Quality categories/source coverage differs')
    if (identity(c)!=r['contract_id'] or json.loads((path.parent/'.done').read_text())!=dict(contract_id=r['contract_id'],manifest_sha256=sha256_file(path)) or
            sha256_file(resolve(r['samples']['path']))!=r['samples']['sha256']):raise ValueError('Quality certificate changed')
    inputs.assert_bound(inputs.manifest,inputs.split_manifest)
    if (c['source_sha256']!=inputs.context.source_sha or c['split_manifest_sha256']!=inputs.context.split_sha or
            c['split_contract_id']!=inputs.selection.report['contract_id'] or c['excluded_groups_id']!=identity(sorted(excluded))):raise ValueError('Quality input binding differs')
    stats={source:dict(documents=0,utf8_bytes=0) for source in c['categories']};seen=set()
    with resolve(r['samples']['path']).open() as stream:
        for line in stream:
            item=json.loads(line);source=item['source'];key=(item['file'],item['row'])
            if source not in stats or key in seen or item['group_id'] in excluded:raise ValueError('Duplicate/excluded quality row')
            actual=inputs.selection.db.execute('SELECT split,content_sha256,group_id,repo FROM docs WHERE source=? AND file=? AND row=?',(source,*key)).fetchone()
            if (actual!=('eval',item['content_sha256'],item['group_id'],item['repo']) or item['split']!='eval' or item['truncated'] or
                    item['category']!=c['categories'][source] or text_hash(item['text'])!=item['content_sha256'] or item['text_sha256']!=item['content_sha256']):
                raise ValueError('Quality document is not complete independent eval text')
            seen.add(key);stats[source]['documents']+=1;stats[source]['utf8_bytes']+=len(item['text'].encode())
    if stats!=r['stats'] or any(v['documents']!=c['per_source'] for v in stats.values()):raise ValueError('Quality coverage differs')
    return r
