"""Offline release-assembly test. No remote APIs or corpus payloads are used."""
import hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import scripts.finalize_public_corpus_v2 as finalizer


def test_release_assembly_and_retry_keep_13_languages_and_pins(tmp_path,monkeypatch):
    real=Path(__file__).resolve().parents[1];realbase=real/'runs/dataset_upload_v1';base=tmp_path/'runs/dataset_upload_v1';base.mkdir(parents=True)
    spec=json.loads((realbase/'upload_spec.json').read_text())
    (base/'upload_spec.json').write_text(json.dumps(spec));(base/'index_assets_uploaded.json').write_text(json.dumps({'commit':'index-parent'}))
    heads={'pathcosmos/korean-pretrain-corpus':'index-parent'};repos={}
    for source in spec['repositories']:
        target=tmp_path/source['metadata_folder'];shutil.copytree(real/source['metadata_folder'],target)
        heads[source['repository']]='data-'+source['source'];repos[source['repository']]={'head':heads[source['repository']]}
    upload={'status':'all_source_payloads_verified','verified_files':spec['files'],'verified_bytes':spec['bytes'],'repositories':repos}
    report=base/'supervisor_report.json';report.write_text(json.dumps({'status':'passed','upload_state':upload}))
    (base/'supervisor.done').write_text(json.dumps({'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}))
    original=tmp_path/'runs/dataset_public_release_v1';original.mkdir(parents=True)
    shutil.copytree(real/'runs/dataset_public_release_v1/public_index_v4',original/'public_index_v4')
    shutil.copyfile(real/'runs/dataset_public_release_v1/build_multilingual_cards.py',original/'build_multilingual_cards.py')
    shutil.copytree(realbase/'index_metadata',base/'index_metadata');(base/'index_assets').mkdir()
    cache=base/'index_assets/examples/__pycache__';cache.mkdir(parents=True);(cache/'local.pyc').write_bytes(b'private-path-cache')
    work=base/'final_metadata';work.mkdir();(work/'consumer_validation.json').write_text(json.dumps({'status':'passed','test_fixture_only':True}))
    tags={};published=[]
    class API:
        def whoami(self):return {'name':'pathcosmos'}
        def repo_info(self,repo,repo_type=None,revision=None):
            return SimpleNamespace(sha=tags[(repo,revision)] if revision else heads[repo],private=False)
        def create_tag(self,repo,tag,revision,**kw):
            assert tags.get((repo,tag),revision)==revision;tags[(repo,tag)]=revision
    api=API();monkeypatch.setattr(finalizer,'HfApi',lambda **kw:api)
    def publish(api,repo,head,changes):
        assert head==heads[repo]
        for n,p in changes.items():
            if n.endswith('.json'):json.loads(p.read_text())
        revision='commit-'+str(len(published));heads[repo]=revision;published.append((repo,revision));return revision
    monkeypatch.setattr(finalizer,'publish',publish)
    first=finalizer.finalize(tmp_path)
    assert first['status']=='passed' and len(first['sources'])==6 and len(published)==7
    index=work/'index';text=(index/'README.md').read_text()
    assert text.index('## 한국어')<text.index('## English')
    import yaml
    metadata=yaml.safe_load(text.split('---',2)[1])
    assert {'text','llm','pretraining','korean-pretrain-corpus','code','mathematics'}<=set(metadata['tags'])
    assert metadata['language']==['ko','en'] and metadata['license']=='other'
    for source in spec['repositories']:
        card=(work/source['source']/'README.md').read_text()
        header=yaml.safe_load(card.split('---',2)[1])
        assert 'korean-pretrain-corpus' in header['tags'] and header['license']=='other'
        assert header['configs'][0]['data_files'][0]['split']=='all'
    assert len(list((index/'cards').glob('*.md')))==13
    assert '__pycache__' not in (index/'CHECKSUMS.json').read_text()
    for code in finalizer.FINAL_STATUS:
        card=(index/'cards'/f'README.{code}.md').read_text()
        assert finalizer.FINAL_STATUS[code] in card and finalizer.COMPAT[code] in card and finalizer.USAGE[code] in card
    second=finalizer.finalize(tmp_path)
    assert second['commit']==first['commit'] and len(published)==7
    assert (index/'README.md').read_text().count('## Pinned public release / 고정 공개 버전')==1
