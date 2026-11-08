"""Fresh parallel input hashing with byte-identical native training identities.

No persistent validation cache. One invocation hashes every immutable input once;
the returned hashes populate the native schema directly and include stat guards.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
from common.data_contracts import resolve,sha256_file,identity
from train.runtime import RESUME_KEYS


def native_expected_inputs(token_root,mix,hold_files,*,require_production=True):
    root=resolve(token_root);report=json.loads((root/'report.json').read_text());contract=json.loads((root/'contract.json').read_text())
    if json.loads((root/'.done').read_text())!=dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')):
        raise ValueError('Native completion proof differs')
    if report['shards']!=report['planned_shards'] or report['status'] not in ('complete','experimental_complete'):
        raise ValueError('Expected whole native token output')
    if require_production and (contract['experimental'] or report['status']!='complete'):raise ValueError('Expected production native token output')
    logical=[];outputs={}
    for profile in report['files']:
        source=profile['source'];file=profile['path'];directory=root/source
        record=json.loads((directory/(Path(file).stem+'.bin.done')).read_text());logical.append(dict(source=source,file=file,record=record))
        for entry in record['outputs']:
            path=str((directory/entry['name']).resolve())
            if path in outputs:raise ValueError('Repeated native output path')
            outputs[path]=dict(bytes=entry['bytes'],sha256=entry['sha256'])
    if identity(logical)!=report['logical_id']:raise ValueError('Native output digest records changed')
    wanted={str(resolve(p).resolve()) for source in mix['sources'].values() for p in source['files']}|{str(resolve(p).resolve()) for p in hold_files}
    if wanted-set(outputs):raise ValueError('Training input is outside certified native outputs')
    return {p:outputs[p] for p in sorted(wanted)}


def training_contract(config,world,hold_files,*,workers=32,observe=None,expected_files=None):
    if type(workers) is not int or not 1<=workers<=32:raise ValueError('Invalid input hash concurrency')
    start=time.monotonic();mix=json.loads(resolve(config['mix']).read_text())
    paths=list(dict.fromkeys(str(resolve(p).resolve()) for s in mix['sources'].values() for p in s['files']))
    paths=list(dict.fromkeys(paths+[str(resolve(p).resolve()) for p in hold_files]))
    count=min(workers,len(os.sched_getaffinity(0)),max(1,len(paths)))
    def record(name):
        path=Path(name);before=path.stat();digest=sha256_file(path);after=path.stat()
        if any(getattr(before,k)!=getattr(after,k) for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']):
            raise ValueError('Token input changed during fresh hashing')
        return name,dict(bytes=before.st_size,sha256=digest)
    with ThreadPoolExecutor(max_workers=count) as pool:records=dict(pool.map(record,paths))
    if expected_files is not None and records!=expected_files:raise ValueError('Fresh training input differs from certified native output digest')
    value=dict(version=1,world_size=world,config={k:config[k] for k in RESUME_KEYS},mix=mix,source_order=list(mix['sources']),
               holdout=list(hold_files),files=records,code={p:sha256_file(resolve(p)) for p in
               ['model/model.py','train/__init__.py','train/train.py','train/data.py','train/runtime.py']})
    value['id']=identity(value)
    if observe is not None:observe(dict(workers=count,files=len(records),bytes=sum(r['bytes'] for r in records.values()),elapsed_seconds=time.monotonic()-start,
                                      scope='Fresh parallel SHA for this live contract invocation; identical native schema, no persistent skip-validation cache.'))
    return value
