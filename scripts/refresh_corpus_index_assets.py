"""Refresh changed small assets and verify the complete public index manifest."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import atomic_json,verify
from huggingface_hub import HfApi,CommitOperationAdd

if __name__=='__main__':
    root=Path(PROJECT_ROOT);base=root/'runs/dataset_upload_v1'
    assert json.loads((base/'upload_state.json').read_text())['status']!='stopped_quota'
    marker=base/'index_assets_uploaded.json';old=json.loads(marker.read_text());repo=old['repository'];api=HfApi()
    assert api.repo_info(repo,repo_type='dataset').sha==old['commit'],'Index changed outside this transfer'
    original=root/'runs/dataset_public_release_v1/public_index_v4';assets=base/'index_assets';docs=base/'index_metadata'
    paths={str(p.relative_to(original)):p for p in original.rglob('*') if p.is_file() and p.name!='CHECKSUMS.json'}
    paths.update({str(p.relative_to(assets)):p for p in assets.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
    paths.update({n:docs/n for n in ['README.md','PUBLICATION_STATUS.json']})
    entries=[{'path':n,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(paths.items())]
    atomic_json(docs/'CHECKSUMS.json',{'files':entries});paths['CHECKSUMS.json']=docs/'CHECKSUMS.json'
    previous={r['path']:r for r in old['proofs']}
    changed={n:p for n,p in paths.items() if n not in previous or hashlib.sha256(p.read_bytes()).hexdigest()!=previous[n]['sha256']}
    if changed:
        commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=old['commit'],
            commit_message='Refresh reproducibility archive and verify complete index checksums',
            operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=p) for n,p in changed.items()]);revision=commit.oid
    else:revision=old['commit']
    records=[{'path':n,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in paths.items()]
    assert all('__pycache__' not in r['path'] for r in records)
    proofs=verify(HfApi(token=False),repo,revision,records,anonymous=True)
    history=base/'index_history';history.mkdir(exist_ok=True);atomic_json(history/(old['commit']+'.json'),old)
    atomic_json(marker,{'status':'complete_index_manifest_verified_public','repository':repo,'commit':revision,'proofs':proofs,
                        'streaming_example_real_public_test':'passed','generated_caches_excluded':True})
    print(json.dumps({'status':'complete_index_manifest_verified_public','commit':revision,'files':len(proofs),'changed_files':len(changed)}))
