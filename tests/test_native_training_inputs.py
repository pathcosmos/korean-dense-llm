from scripts.publish_native_training_inputs import effective_input_paths


def test_explicit_evaluation_subset_preserves_native_source_and_file_order():
    mix={'sources':{'second':{'files':['b.bin','a.bin']},'first':{'files':['c.bin']}},
         'holdout':['unused.hold.bin','selected.hold.bin']}
    config={'holdout':['selected.hold.bin']}
    assert effective_input_paths(config,mix)==['b.bin','a.bin','c.bin','selected.hold.bin']
    assert effective_input_paths({'holdout':[]},mix)==['b.bin','a.bin','c.bin','unused.hold.bin','selected.hold.bin']


def test_final_native_package_retains_exact_732_inputs_and_reuses_release_on_retry(tmp_path,monkeypatch):
    import json,shutil
    from pathlib import Path
    from types import SimpleNamespace
    import scripts.publish_native_training_inputs as pub
    real=Path(pub.PROJECT_ROOT);base=tmp_path/'runs/training_inputs_release_v1';base.mkdir(parents=True)
    shutil.copytree(real/'runs/training_inputs_release_v1/metadata',base/'metadata')
    spec=json.loads((real/'runs/training_inputs_release_v1/upload_spec.json').read_text())
    (base/'consumer_validation.json').write_text(json.dumps({'status':'passed','fixture_only':True}))
    monkeypatch.setattr(pub,'PROJECT_ROOT',tmp_path);monkeypatch.setattr(pub,'BASE',base)
    commits=[];tags={}
    class API:
        def repo_info(self,*args,revision=None,**kwargs):
            return SimpleNamespace(sha=tags.get(revision,revision) if revision else ('final' if commits else 'data'))
        def create_commit(self,*args,**kwargs):commits.append(kwargs);return SimpleNamespace(oid='final')
        def create_tag(self,repo,tag,revision,**kwargs):
            assert tags.get(tag,revision)==revision;tags[tag]=revision
        def list_repo_files(self,*args,**kwargs):
            return ['.gitattributes','CHECKSUMS.json']+[r['path'] for r in json.loads((base/'final/CHECKSUMS.json').read_text())['files']]
    monkeypatch.setattr(pub,'HfApi',lambda **kwargs:API());monkeypatch.setattr(pub,'verify',lambda *args,**kwargs:None)
    state={'repositories':{pub.REPO:{'head':'data'}}}
    first=pub.finish(spec,state);second=pub.finish(spec,state)
    assert first['commit']==second['commit']=='final' and len(commits)==1
    entries=json.loads((base/'final/CHECKSUMS.json').read_text())['files']
    binaries=[r for r in entries if r['path'].endswith('.bin')]
    assert len(binaries)==732 and sum(r['bytes'] for r in binaries)==242377468750
    assert sum(r['path'].endswith('.hold.bin') for r in binaries)==6
    assert (base/'final/README.md').read_text().index('## 한국어')<(base/'final/README.md').read_text().index('## English')
    assert len(list((base/'metadata/cards').glob('*.md')))==13
    original=json.loads((base/'metadata/config/main.original.json').read_text())
    mix=json.loads((base/'metadata'/original['mix']).read_text())
    assert len(mix['holdout'])==726 and len(original['holdout'])==6
    assert set(effective_input_paths(original,mix))=={r['path'] for r in binaries}
