"""One invocation's verified, closed split and source inventory; no persistent trust cache."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common.data_contracts import resolve, artifact_path, source_files, sha256_file
from data.split_manifest import SplitSelection


class VerifiedSplitContext:
    def __init__(self, sources, split, *, hash_workers=16, cache_mib=256):
        self.source_path=resolve(sources);self.split_path=resolve(split)
        self.source_sha=sha256_file(self.source_path);self.split_sha=sha256_file(self.split_path)
        self.inventory=json.loads(self.source_path.read_text())
        self.selection=SplitSelection(self.split_path,sqlite_cache_mib=cache_mib)
        self.records={};self.snapshot=None
        try:
            for source in self.inventory['sources']:
                paths,snapshot,bound=source_files(source,self.source_path,return_verified_records=True,verification_workers=hash_workers)
                verified={r['path']:r for r in bound} if bound else {}
                def record(path):
                    key=artifact_path(path);prior=verified.get(key);size=path.stat().st_size
                    if prior and prior['bytes']!=size:raise ValueError('Verified source size changed')
                    return dict(path=key,bytes=size,sha256=prior['sha256'] if prior else sha256_file(path))
                with ThreadPoolExecutor(max_workers=hash_workers) as pool:records=list(pool.map(record,paths))
                self.selection.verify_source(source,records,snapshot)
                self.records[source]=records;self.snapshot=snapshot
            self.assert_control_files()
        except BaseException:
            self.close();raise

    def assert_control_files(self):
        if sha256_file(self.source_path)!=self.source_sha or sha256_file(self.split_path)!=self.split_sha:
            raise ValueError('Source/split manifest changed within invocation')

    def close(self):self.selection.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def distribution(context):
    """Additional distribution only; completed SplitSelection already checked leakage/coverage."""
    s=context.selection;db=s.db;sources={}
    for name in context.records:
        rows=db.execute('SELECT split,COUNT(*),COUNT(DISTINCT group_id) FROM docs WHERE source=? GROUP BY split',(name,)).fetchall()
        largest=db.execute('SELECT group_id,COUNT(*) AS n FROM docs WHERE source=? GROUP BY group_id ORDER BY n DESC,group_id LIMIT 5',(name,)).fetchall()
        sources[name]=dict(splits={label:dict(documents=n,groups=g) for label,n,g in rows},largest_components=[dict(group_id=g,documents=n) for g,n in largest])
    largest=db.execute('SELECT group_id,COUNT(*) AS n,COUNT(DISTINCT source) FROM docs GROUP BY group_id ORDER BY n DESC,group_id LIMIT 10').fetchall()
    return dict(split_contract_id=s.report['contract_id'],source_sha256=context.source_sha,split_manifest_sha256=context.split_sha,
                database=s.report['database'],counts=s.report['counts'],sources=sources,
                largest_components=[dict(group_id=g,documents=n,sources=c) for g,n,c in largest],
                validation='Fresh SplitSelection: DB SHA/integrity/raw-group-repo cross-split zero/complete row coverage; same open context reused.',
                limitation='Document/group distribution only. Raw UTF8 byte and token distribution require later source/token pilot.')
