"""Publish the authorized small code/source preview; never corpus payload."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests
from huggingface_hub import HfApi, CommitOperationAdd


def main():
    root=Path(__file__).resolve().parents[2]
    folder=root/'runs/dataset_public_release_v1/public_index_v1'
    report_file=folder.parent/'public_index_upload.json'
    if report_file.exists():raise FileExistsError('Preserve the existing publication report')
    allowed={'README.md','SOURCES.md','sources.json','CONTACT.md','LICENSE-SCOPE.md',
             'LICENSE-CODE','code-preview.tar.gz','CHECKSUMS.json'}
    assert {p.name for p in folder.iterdir()}==allowed
    for r in json.loads((folder/'CHECKSUMS.json').read_text())['files']:
        p=folder/r['path'];assert p.name in allowed
        assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    api=HfApi(); who=api.whoami(); assert who['name']=='pathcosmos'
    repo='pathcosmos/ko-en-code-math-corpus'
    info=api.repo_info(repo,repo_type='dataset');assert info.private is False
    assert info.sha=='dfc37737ee60d12fdf3e026aa2c72513795ad809', 'Remote changed; review first'
    commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=info.sha,
        commit_message='Publish code preview, pinned source inventory and contact policy',
        operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=p) for p in sorted(folder.iterdir())])
    verified=[]
    session=requests.Session();session.trust_env=False
    for p in sorted(folder.iterdir()):
        url=f'https://huggingface.co/datasets/{repo}/resolve/{commit.oid}/{p.name}'
        response=session.get(url,timeout=45);response.raise_for_status()
        assert response.content==p.read_bytes(), p.name
        verified.append({'path':p.name,'bytes':len(response.content),'sha256':hashlib.sha256(response.content).hexdigest()})
    report={'utc':datetime.now(timezone.utc).isoformat(),'status':'verified_public_code_metadata_preview',
        'repository':repo,'commit':commit.oid,'private':False,'anonymous_downloads_verified':verified,
        'new_corpus_text_uploaded':False,'token_bins_uploaded':False,'paid_services_used':False}
    report_file.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        print(json.dumps({'error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None),
            'stack':[(Path(f.filename).name,f.lineno,f.name) for f in traceback.extract_tb(e.__traceback__)]}))
        sys.exit(1)
