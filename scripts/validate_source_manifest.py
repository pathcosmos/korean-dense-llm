"""Structural ready-manifest audit. Does not certify quality or change draft/ready."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import source_files,resolve,artifact_path,sha256_file,atomic_json


def validate(manifest, candidate=False):
    manifest=resolve(manifest);inventory=json.loads(manifest.read_text());seen=set();sources={}
    for name in inventory['sources']:
        if candidate:
            if inventory.get('schema_version')!=1 or inventory.get('status') not in ('draft','ready'):
                raise ValueError('Unsupported candidate inventory')
            entry=inventory['sources'][name]
            if entry.get('text_policy') not in ('nfc-web-exact-v1','raw-utf8-code-v1') or not resolve(entry['completion_marker']).is_file():
                raise ValueError('Candidate source is incomplete')
            paths=[resolve(p) for p in entry['files']]
            if not paths or any(not p.is_file() or not Path(str(p)+'.done').is_file() for p in paths):
                raise ValueError('Candidate file is incomplete')
        else:
            paths,snapshot=source_files(name,manifest)
        rows=0;files=[]
        for p in paths:
            resolved=str(p.resolve())
            if resolved in seen:raise ValueError('Repeated file across sources')
            seen.add(resolved);pf=pq.ParquetFile(p)
            if 'text' not in pf.schema_arrow.names:raise ValueError('Missing text column')
            if not (pa.types.is_string(pf.schema_arrow.field('text').type) or pa.types.is_large_string(pf.schema_arrow.field('text').type)):
                raise ValueError('Text column must contain strings')
            count=0
            for batch in pf.iter_batches(batch_size=1024,columns=['text']):
                if batch.column(0).null_count:raise ValueError('Null text')
                count+=batch.num_rows
            if count!=pf.metadata.num_rows:raise ValueError('Parquet count mismatch')
            rows+=count;files.append(dict(path=artifact_path(p),rows=count,sha256=sha256_file(p)))
        sources[name]=dict(rows=rows,files=files)
    if not sources:raise ValueError('Empty source inventory')
    return dict(structural_status='passed',manifest_sha256=sha256_file(manifest),sources=sources,
                quality_status='not_certified_by_this_tool',candidate_audit=bool(candidate))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--candidate',action='store_true',help='Audit a completed draft without promoting it')
    a=ap.parse_args();r=validate(a.manifest,a.candidate);out=resolve(a.output)
    if out.exists() and json.loads(out.read_text())!=r:raise ValueError('Existing report differs; choose new output')
    if not out.exists():atomic_json(out,r)
    print(json.dumps(dict(structural_status=r['structural_status'],quality_status=r['quality_status'])))


if __name__=='__main__':main()
