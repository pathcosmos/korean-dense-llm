"""Prepare a DRAFT six-source input inventory. Never marks data ready or starts training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA
from common.data_contracts import atomic_json, artifact_path, resolve

SOURCES = ['wiki_ko', 'fineweb2_ko', 'hplt_ko', 'fineweb_edu', 'owm', 'github_code_v1']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='data/datasets/final_v1.draft.json')
    a=ap.parse_args();sources={}
    for source in SOURCES:
        raw_manifest=Path(DATA)/'raw'/source/'download_manifest.json'
        raw=json.loads(raw_manifest.read_text())
        code=source=='github_code_v1'
        directory=Path(DATA)/'clean'/source if code else Path(DATA)/'dedup/exact_v1/clean'/source
        files=[directory/Path(f).name for f in raw['files']]
        completion=directory/'.done' if code else Path(DATA)/'dedup/exact_v1/.done'
        sources[source]=dict(text_policy='raw-utf8-code-v1' if code else 'nfc-web-exact-v1',
            completion_marker=artifact_path(completion),files=[artifact_path(f) for f in files],
            origin=dict(repo=raw['repo'],revision=raw['revision'],raw_manifest=artifact_path(raw_manifest)),
            inventory=dict(expected_shards=len(files),completed_shards=sum(Path(str(f)+'.done').exists() for f in files),
                           complete=completion.exists()))
    report=dict(schema_version=1,status='draft',created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
                unresolved_checks=['final exact_v1 output verification','near-duplicate policy',
                    'code cross-check scope with Korean/English','holdout grouping policy','tokenizer byte mixture'],
                note='Inventory only. Completed counts are not full content verification. Do not change ready without resolving quality checks.')
    output=resolve(a.output)
    if output.exists() and json.loads(output.read_text()).get('status')!='draft':
        raise ValueError('Will not overwrite a reviewed manifest')
    atomic_json(output,report)
    print('DRAFT ONLY',output)
    print(json.dumps({s:e['inventory'] for s,e in sources.items()},indent=2))


if __name__=='__main__':
    main()
