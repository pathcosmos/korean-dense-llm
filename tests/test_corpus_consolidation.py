import copy
from scripts.consolidate_public_corpus import rebase_manifest,REPO


def test_rebase_preserves_canonical_identity_hashes_and_upstream():
    original={'repository':'legacy','content_inventory_id':'inventory','source':'wiki_ko',
        'source_documents':[{'original_final_file':'original/shard.parquet','rows':7,
            'text_path':'text/a.parquet','assignment_path':'splits/a.parquet','offset_path':'offsets/a.parquet',
            'text_sha256':'body-sha','assignment_sha256':'split-sha','offset_sha256':'offset-sha',
            'upstream_shard':{'path':'upstream/a.parquet','revision':'original'}}],
        'payload_files':[{'path':'text/a.parquet','bytes':100,'sha256':'body-sha'}]}
    before=copy.deepcopy(original);result=rebase_manifest(original,'wiki_ko')
    assert original==before and result['repository']==REPO
    item=result['source_documents'][0];old=original['source_documents'][0]
    for key in ['text_path','assignment_path','offset_path']:assert item[key]=='wiki_ko/'+old[key]
    for key in ['original_final_file','rows','text_sha256','assignment_sha256','offset_sha256','upstream_shard']:
        assert item[key]==old[key]
    assert result['content_inventory_id']==original['content_inventory_id']
    assert result['payload_files']==[{'path':'wiki_ko/text/a.parquet','bytes':100,'sha256':'body-sha'}]


def test_single_repository_release_download_paths_and_retry(tmp_path,monkeypatch):
    import json,shutil,yaml
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.consolidate_public_corpus as release
    real=Path(release.PROJECT_ROOT);base=tmp_path/'runs/dataset_unified_v1';base.mkdir(parents=True)
    for name in ['root_template','metadata']:
        shutil.copytree(real/'runs/dataset_unified_v1'/name,base/name)
    shutil.copyfile(real/'runs/dataset_unified_v1/preparation.json',base/'preparation.json')
    old=tmp_path/'runs/dataset_upload_v1';old.mkdir()
    shutil.copyfile(real/'runs/dataset_upload_v1/upload_spec.json',old/'upload_spec.json')
    (base/'consumer_validation.json').write_text(json.dumps({'status':'passed','fixture_only':True}))
    monkeypatch.setattr(release,'PROJECT_ROOT',tmp_path);monkeypatch.setattr(release,'BASE',base)
    commits=[];tags={}
    class API:
        def repo_info(self,*a,revision=None,files_metadata=False,**kw):
            entries=json.loads((base/'root/CHECKSUMS.json').read_text())['files'] if files_metadata else []
            return SimpleNamespace(sha=tags.get(revision,revision) if revision else ('final' if commits else 'data'),
                                   siblings=[SimpleNamespace(size=e['bytes']) for e in entries])
        def create_commit(self,*a,**kw):commits.append(kw);return SimpleNamespace(oid='final')
        def create_tag(self,repo,tag,revision,**kw):
            assert tags.get(tag,revision)==revision;tags[tag]=revision
        def list_repo_files(self,*a,**kw):
            entries=json.loads((base/'root/CHECKSUMS.json').read_text())['files']
            return ['.gitattributes','CHECKSUMS.json']+[e['path'] for e in entries]
    monkeypatch.setattr(release,'HfApi',lambda **kw:API())
    monkeypatch.setattr(release,'verify',lambda *a,**kw:None)
    records=[{'path':'wiki_ko/text/a.parquet','bytes':100,'sha256':'0'*64},
             {'path':'README.md','bytes':1,'sha256':'1'*64}]
    spec={'files':2,'bytes':101,'repositories':[{'files':records}]}
    state={'repositories':{REPO:{'head':'data'}}}
    first=release.finalize(spec,state);second=release.finalize(spec,state)
    assert first['commit']==second['commit']=='final' and len(commits)==1
    text=(base/'root/README.md').read_text();metadata=yaml.safe_load(text.split('---',2)[1])
    assert len(metadata['configs'])==6 and all(c['data_files'][0]['path']==c['config_name']+'/text/*.parquet' for c in metadata['configs'])
    assert 'allow_patterns=[' not in text and 'local_dir="korean-pretrain-corpus"' in text
    assert text.index('## 한국어')<text.index('## English')
    assert len(list((base/'root/cards').glob('*.md')))==13
    assert 'six separate datasets' not in text
    for card in [base/'root/README.md',*list((base/'root/cards').glob('*.md'))]:
        rendered=card.read_text()
        assert 'https://hf.co/datasets/pathcosmos/korean-pretrain-wikipedia' in rendered
        assert 'https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko' in rendered
        assert rendered.count('<!-- corpus-repository-links:start -->')==1
    checks=json.loads((base/'root/CHECKSUMS.json').read_text())['files']
    assert any(r['path']=='wiki_ko/text/a.parquet' for r in checks)
    assert not any('__pycache__' in r['path'] for r in checks)
    assert set(tags)=={'v1.1.0'}
