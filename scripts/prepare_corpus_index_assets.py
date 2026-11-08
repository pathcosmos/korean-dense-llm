"""Prepare small reproducibility assets without copying large corpus payloads."""
import hashlib,json,shutil,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT


if __name__=='__main__':
    root=Path(PROJECT_ROOT);base=root/'runs/dataset_upload_v1';assets=base/'index_assets'
    recipe=assets/'recipe';recipe.mkdir(exist_ok=True)
    snapshot=root/'runs/dataset_public_release_v1/recipe_snapshot_v1'
    with tarfile.open(assets/'reproduction-code.tar.gz','w:gz') as tf:
        for p in sorted(snapshot.rglob('*')):
            rel=p.relative_to(snapshot)
            if p.is_file() and rel.parts[0]!='tokenizer':tf.add(p,arcname=str(rel),recursive=False)
        for rel in ['data/release_provenance.py','data/release_split.py','data/release_offsets.py','data/public_corpus_reader.py',
            'scripts/prepare_corpus_upload.py','scripts/upload_public_corpus.py','scripts/supervise_corpus_upload.py',
            'tests/test_corpus_upload.py','tests/test_public_corpus_reader.py','tests/test_release_offsets.py',
            'tests/test_release_split.py','tests/test_release_provenance.py']:
            tf.add(root/rel,arcname='release_tools/'+rel,recursive=False)
    config=json.loads((root/'runs/3b_full_v1/main.json').read_text())
    (recipe/'main.original.json').write_text(json.dumps(config,indent=2)+'\n')
    config['mix']='recipe/mix.portable.json';config['out_dir']='runs/korean-pretrain-consumer'
    config['holdout']=[f'{Path(p).parts[-2]}/tokens/validation/{Path(p).name}' for p in config['holdout']]
    (recipe/'main.portable.json').write_text(json.dumps(config,indent=2)+'\n')
    mix=json.loads((root/'data/tokens/final_v1/mix.json').read_text())
    for key,item in mix['sources'].items():
        item['files']=[f'{key}/tokens/train/{Path(p).name}' for p in item['files']]
        item.pop('expected_exposures',None)
    assert isinstance(mix['holdout'],list)
    mix['holdout']=[f'{Path(p).parts[-2]}/tokens/validation/{Path(p).name}' for p in mix['holdout']]
    (recipe/'mix.portable.json').write_text(json.dumps(mix,indent=2)+'\n')
    decisions=json.loads((root/'runs/dataset_public_release_v1/publication_decisions.json').read_text())
    (recipe/'source_repositories.json').write_text(json.dumps(decisions['source_repository_names'],indent=2)+'\n')
    (recipe/'README.md').write_text('''# 학습 recipe / Training recipe

원래 실행 설정은 main.original.json이며 model=3b/vocab65536/seq4096/global2,097,152tokens/seed1234입니다.
57,000step 설정은 참고용 전체 학습 계약입니다. 자신의 자원과 출력 경로를 확인하고 별도 실행 디렉터리를 사용하세요.

main.portable.json remaps input paths for a data root containing recipe/ and one directory per source:
<source>/tokens/train/ and <source>/tokens/validation/. Preserve the final published source commits.
Set PROJECT_ROOT to that data root when using the native loader. mix.portable.json preserves the
actual weights and token counts. Stale pilot exposure estimates were removed; sampling with replacement
is not a single corpus epoch. The original configuration is retained for provenance.

Source repositories supply manifest.json, splits/, offsets/ and public_corpus_reader.py.
The original text shards contain BOTH train and eval: use an explicit canonical split for text training.
Binary train/eval files are already separated. Each binary document includes its appended EOS.

reproduction-code.tar.gz contains the original training/cleaning code snapshot. The tokenizer is
distributed separately under tokenizer/. Historical snapshot manifests describe their original
snapshot and are not a complete inventory of this expanded archive; release_tools/ are later additions.
Use the outer archive SHA256 for transport integrity. Full standalone training reproduction is not
yet validated; no full training job is started by these asset preparation checks.

No model checkpoint or instruction-tuning dataset is included in this data release.
''')
    files=[{'path':str(p.relative_to(assets)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(assets.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    (base/'index_asset_manifest.json').write_text(json.dumps({'repository':'pathcosmos/korean-pretrain-corpus','files':files},indent=2)+'\n')
    print(json.dumps({'index_files':len(files),'index_bytes':sum(i['bytes'] for i in files)}))
