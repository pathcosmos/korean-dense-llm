"""Publish tokenizer and reproducibility assets while source transfer proceeds."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import verify,atomic_json
from huggingface_hub import HfApi,CommitOperationAdd


if __name__=='__main__':
    root=Path(PROJECT_ROOT);base=root/'runs/dataset_upload_v1'
    state=json.loads((base/'upload_state.json').read_text())
    if state['status']=='stopped_quota':raise RuntimeError('Quota-stop latched')
    marker=base/'index_assets_uploaded.json'
    if marker.exists():raise FileExistsError('Already published')
    repo='pathcosmos/korean-pretrain-corpus';api=HfApi();info=api.repo_info(repo,repo_type='dataset')
    assert info.sha=='22737538aedc37a4cfd74aed41dc46af53d22da6' and not info.private
    source=root/'runs/dataset_public_release_v1/public_index_v4'
    docs=base/'index_metadata';docs.mkdir(exist_ok=True)
    text=(source/'README.md').read_text()
    text=text.replace('| Tokenizer distribution | No | Local raw BPE verified; public packaging/loader compatibility pending |',
        '| Tokenizer distribution | Yes | Exact raw JSON and AutoTokenizer configuration; all 65,536 vocabulary IDs and 16 special IDs preserved |')
    text=text.replace('"PUBLICATION_STATUS.json", "code-preview.tar.gz",','"PUBLICATION_STATUS.json", "code-preview.tar.gz",\n        "tokenizer/*", "recipe/*", "examples/*", "REPRODUCE.md", "reproduction-code.tar.gz",')
    text+='\nTokenizer: [loading guide](tokenizer/README.md). Reuse: [REPRODUCE.md](REPRODUCE.md). Source payloads are being transferred in verified batches; a final pinned release index will follow.\n'
    (docs/'README.md').write_text(text)
    status=json.loads((source/'PUBLICATION_STATUS.json').read_text());status['source_repositories_created']=True
    status['tokenizer_public']=True;status['payload_transfer_status']='running';status['text_representation']='original_shards_with_canonical_split_sidecars'
    (docs/'PUBLICATION_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    allfiles={str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file() and p.name!='CHECKSUMS.json'}
    assets=base/'index_assets'
    allfiles.update({str(p.relative_to(assets)):p for p in assets.rglob('*') if p.is_file()})
    allfiles.update({'README.md':docs/'README.md','PUBLICATION_STATUS.json':docs/'PUBLICATION_STATUS.json'})
    checks=[{'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in sorted(allfiles.items())]
    (docs/'CHECKSUMS.json').write_text(json.dumps({'files':checks},indent=2)+'\n')
    changes={str(p.relative_to(assets)):p for p in assets.rglob('*') if p.is_file()}
    changes.update({n:docs/n for n in ['README.md','PUBLICATION_STATUS.json','CHECKSUMS.json']})
    operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=p) for n,p in sorted(changes.items())]
    commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=info.sha,
        commit_message='Publish exact tokenizer, portable recipe and canonical streaming reader',operations=operations)
    records=[{'path':n,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(changes.items())]
    proofs=verify(HfApi(token=False),repo,commit.oid,records,anonymous=True)
    atomic_json(marker,{'status':'index_assets_verified_public','repository':repo,'commit':commit.oid,'proofs':proofs})
    print(json.dumps({'status':'index_assets_verified_public','commit':commit.oid,'files':len(proofs)}))
