"""Apply the user's selected name and verify preservation/public redirects."""
import hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import requests
from huggingface_hub import HfApi,CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError


def main():
    base=Path(__file__).resolve().parent;old='pathcosmos/ko-en-code-math-corpus';new='pathcosmos/korean-pretrain-corpus'
    report_path=base/'repository_name_applied.json'
    if report_path.exists():raise FileExistsError('Already applied; inspect saved report')
    api=HfApi();assert api.whoami()['name']=='pathcosmos'
    before=api.repo_info(old,repo_type='dataset');assert before.sha=='9b8ddf6f225c79ad42c98eae44aeea7b270913bf'
    old_files=api.list_repo_files(old,repo_type='dataset',revision=before.sha)
    try:
        api.repo_info(new,repo_type='dataset')
    except RepositoryNotFoundError:pass
    else:raise RuntimeError('Destination exists; inspect before changing')
    api.move_repo(from_id=old,to_id=new,repo_type='dataset')
    after=api.repo_info(new,repo_type='dataset');assert after.sha==before.sha and after.private is False
    assert api.list_repo_files(new,repo_type='dataset',revision=before.sha)==old_files
    folder=base/'public_index_v2';shutil.copytree(base/'public_index_v1',folder)
    p=folder/'README.md';text=p.read_text().replace(old,new).replace('Korean-English Code and Math Corpus','Korean Pretraining Corpus')
    text+='\n## Preparation progress\n\nAll 128,856,258 canonical document assignments and offsets for all 122,402,393,813 tokens have been prepared and verified locally. Wikipedia provenance was recovered for 455,397 documents. These prepared payloads have not yet been uploaded. Remaining source attribution, text export and full release validation are ongoing.\n'
    p.write_text(text)
    names={'wiki_ko':'korean-pretrain-wikipedia','fineweb2_ko':'korean-pretrain-fineweb2',
        'hplt_ko':'korean-pretrain-hplt2','fineweb_edu':'korean-pretrain-fineweb-edu',
        'owm':'korean-pretrain-math','github_code_v1':'korean-pretrain-code'}
    state={'public_index':new,'planned_source_repositories':{k:'pathcosmos/'+v for k,v in names.items()},
        'status':'code_source_metadata_preview_public_full_corpus_pending','all_repositories_public':True,
        'source_repositories_created':False,'free_capacity_strategy':'attempt_public_upload_stop_on_quota_error',
        'local_verified_documents':128856258,'local_verified_tokens':122402393813,'local_wiki_provenance_documents':455397,
        'corpus_text_public':False,'token_bins_public':False,'code_license':'MIT','prose_license':'CC-BY-4.0',
        'contact':'miracle.yang@pathcosmos.com'}
    (folder/'PUBLICATION_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    checks=[{'path':q.name,'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in sorted(folder.iterdir()) if q.name!='CHECKSUMS.json']
    (folder/'CHECKSUMS.json').write_text(json.dumps({'files':checks},indent=2)+'\n')
    commit=api.create_commit(repo_id=new,repo_type='dataset',parent_commit=after.sha,
        commit_message='Apply Korean Pretraining Corpus name and publication status',operations=[
            CommitOperationAdd(path_in_repo=n,path_or_fileobj=folder/n) for n in ['README.md','PUBLICATION_STATUS.json','CHECKSUMS.json']])
    session=requests.Session();session.trust_env=False
    for n in ['README.md','PUBLICATION_STATUS.json','CHECKSUMS.json']:
        r=session.get(f'https://huggingface.co/datasets/{new}/resolve/{commit.oid}/{n}',timeout=45);r.raise_for_status()
        assert r.content==(folder/n).read_bytes()
    redirect=session.get(f'https://huggingface.co/datasets/{old}/resolve/{before.sha}/README.md',timeout=45);redirect.raise_for_status()
    assert redirect.content==(base/'public_index_v1/README.md').read_bytes()
    report={'utc':datetime.now(timezone.utc).isoformat(),'status':'rename_and_public_access_verified','old_repository':old,
        'new_repository':new,'preserved_commit':before.sha,'new_commit':commit.oid,'old_files_preserved':True,
        'old_url_anonymous_download_verified':True,'new_files_anonymous_download_verified':True,'private':False,
        'source_repository_names':state['planned_source_repositories']}
    report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(json.dumps({'error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}));sys.exit(1)
