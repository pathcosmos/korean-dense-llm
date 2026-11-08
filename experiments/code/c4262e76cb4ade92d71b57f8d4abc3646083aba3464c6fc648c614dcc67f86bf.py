"""Publish the requested Korean-first card and linked language documents."""
import hashlib,json,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
import requests
from huggingface_hub import HfApi,CommitOperationAdd


def main():
    base=Path(__file__).resolve().parent;folder=base/'public_index_v4';repo='pathcosmos/korean-pretrain-corpus'
    report_file=base/'multilingual_cards_upload.json'
    if report_file.exists():raise FileExistsError('Publication already recorded')
    api=HfApi();assert api.whoami()['name']=='pathcosmos'
    info=api.repo_info(repo,repo_type='dataset');assert info.sha=='fee16329049218b64e8c6b69d626ec94c2c4f32e' and not info.private
    names=['README.md','PUBLICATION_STATUS.json','CHECKSUMS.json']+[str(p.relative_to(folder)) for p in sorted((folder/'cards').glob('*.md'))]
    assert len(names)==16
    for entry in json.loads((folder/'CHECKSUMS.json').read_text())['files']:
        p=folder/entry['path'];assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
    commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=info.sha,
        commit_message='Add Korean-first and English dataset card with 11 linked translations',
        operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=folder/n) for n in names])
    session=requests.Session();session.trust_env=False;verified=[]
    for n in names:
        r=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{commit.oid}/{n}',timeout=45);r.raise_for_status()
        assert r.content==(folder/n).read_bytes(),n
        verified.append({'path':n,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content)})
    public=HfApi(token=False).repo_info(repo,repo_type='dataset',revision=commit.oid)
    assert not public.private
    report={'utc':datetime.now(timezone.utc).isoformat(),'status':'multilingual_cards_published_and_anonymously_verified',
        'repository':repo,'commit':commit.oid,'main_card_order':['ko','en'],'linked_other_language_count':11,
        'language_documents':13,'verified_files':verified,'dataset_language_metadata':['ko','en'],
        'documentation_translation_is_not_data_language_coverage':True,'new_corpus_text_uploaded':False}
    report_file.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='verified_files'}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}));sys.exit(1)
