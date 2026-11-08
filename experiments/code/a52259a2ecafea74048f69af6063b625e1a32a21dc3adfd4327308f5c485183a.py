"""Download only the two missing pinned wiki shards into a new scratch window."""
import hashlib,json,os,sys,time
from pathlib import Path
import requests
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from data.release_provenance import restore,sha
os.sched_setaffinity(0,[92,93,94,95])
records=json.loads((root/'docs/data_provenance_v1/file_inventory.json').read_text())['files']
registry=json.loads((root/'docs/data_provenance_v1/source_registry.json').read_text())
hardware=json.loads((root/'runs/dataset_public_release_v1/hardware_start.json').read_text())
scratch=Path(hardware['storage']['scratch_dir'])/'dataset_public_release_v1/raw/wiki_ko'
scratch.mkdir(parents=True,exist_ok=True)
for rec in records:
    if rec['source']!='wiki_ko' or Path(rec['final']['path']).name=='train-00002-of-00003.parquet':continue
    upstream=rec['upstream_shard'];target=scratch/Path(upstream['remote_path']).name
    if not target.exists():
        partial=target.with_suffix('.parquet.part')
        offset=partial.stat().st_size if partial.exists() else 0
        headers={'Range':f'bytes={offset}-'} if offset else {}
        response=requests.get(upstream['url'],headers=headers,stream=True,timeout=(15,60));response.raise_for_status()
        append=offset>0 and response.status_code==206
        if append:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
        with partial.open('ab' if append else 'wb') as f:
            for block in response.iter_content(8*1024**2):
                f.write(block)
                assert f.tell()<=upstream['bytes']
            f.flush();os.fsync(f.fileno())
        assert partial.stat().st_size==upstream['bytes'] and sha(partial)==upstream['sha256']
        partial.rename(target)
    out=root/'runs/dataset_public_release_v1/provenance/wiki_ko'/target.stem
    if (out/'report.json').exists():
        print(json.dumps({'existing_report_preserved':str(out.relative_to(root))}),flush=True);continue
    result=restore(rec,target,root/'data/datasets/final_v2.split/manifest.json',registry['content_inventory_id'],out)
    print(json.dumps(result),flush=True)
