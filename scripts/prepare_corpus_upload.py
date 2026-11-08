"""Freeze the complete existing corpus as immutable upload allowlists, no copy of large payloads."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    root=Path(PROJECT_ROOT);out=root/out
    if out.exists():raise FileExistsError('Preserve prior package; use a new output path')
    out.mkdir(parents=True)
    inv=json.loads((root/'docs/data_provenance_v1/file_inventory.json').read_text())['files']
    registry=json.loads((root/'docs/data_provenance_v1/source_registry.json').read_text())
    decisions=json.loads((root/'runs/dataset_public_release_v1/publication_decisions.json').read_text())
    counts=json.loads((root/'runs/f6_tokens_v1/counts.json').read_text())['source_counts']
    split={r['original_final_file']:r for r in json.loads((root/'runs/dataset_public_release_v1/split_full_v1/report.json').read_text())['files']}
    sources=[]
    for s in registry['sources']:
        key=s['source'];repo=decisions['source_repository_names'][key];folder=out/'metadata'/key;folder.mkdir(parents=True)
        files=[];documents=[]
        def add(local,remote,digest=None,size=None,kind='metadata'):
            local=Path(local);st=local.stat()
            if size is not None:assert st.st_size==size, str(local)
            digest=digest or sha(local)
            assert len(digest)==64
            files.append({'local_path':str(local.relative_to(root)),'path':remote,'bytes':st.st_size,'sha256':digest,
                          'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'kind':kind})
        for r in inv:
            if r['source']!=key:continue
            f=r['final'];stem=Path(f['path']).stem
            add(root/f['path'],f'text/{stem}.parquet',f['sha256'],f['bytes'],'text')
            sp=root/'runs/dataset_public_release_v1/split_full_v1'/key/(stem+'.parquet');sr=split[f['path']]
            add(sp,f'splits/{stem}.parquet',sr['sha256'],sr['bytes'],'split')
            off=root/'runs/dataset_public_release_v1/token_offsets_v1'/key/(stem+'.parquet')
            od=json.loads(off.with_suffix('.json').read_text());add(off,f'offsets/{stem}.parquet',od['sha256'],od['bytes'],'offset')
            tokenroot=root/'data/tokens/final_v1'/key
            certificate=json.loads((tokenroot/(stem+'.bin.done')).read_text())
            for binary in certificate['outputs']:
                is_eval=binary['name'].endswith('.hold.bin')
                add(tokenroot/binary['name'],f"tokens/{'validation' if is_eval else 'train'}/{binary['name']}",binary['sha256'],binary['bytes'],'token')
            documents.append({'original_final_file':f['path'],'rows':f['rows'],'text_path':f'text/{stem}.parquet',
                'text_sha256':f['sha256'],'assignment_path':f'splits/{stem}.parquet','assignment_sha256':sr['sha256'],
                'offset_path':f'offsets/{stem}.parquet','offset_sha256':od['sha256'],'upstream_shard':r['upstream_shard']})
            if key=='wiki_ko':
                short=stem.split('-of-')[0]
                candidates=[root/'runs/dataset_public_release_v1/provenance/wiki_ko'/short,
                            root/'runs/dataset_public_release_v1/provenance/wiki_ko'/stem]
                provenance=next(p for p in candidates if (p/'provenance.parquet').exists())
                pr=json.loads((provenance/'report.json').read_text())
                add(provenance/'provenance.parquet',f'provenance/{stem}.parquet',pr['sidecar_sha256'],pr['sidecar_bytes'],'provenance')
                add(provenance/'resolution.json',f'provenance/{stem}.resolution.json')
        status='all_wiki_source_candidates_restored' if key=='wiki_ko' else 'repo_path_row_preserved_git_commit_unknown' if key=='github_code_v1' else 'shard_provenance_available_document_urls_not_restored'
        public_source={k:s[k] for k in ['source','repository','dataset_url','pinned_revision','pinned_card_url','selected_subset','final_documents','final_bytes']}
        public_source.update({'provenance_status':status,'license_scope':s['license_scope'],
                              'declared_upstream_license':s['declared_license_interpreted_from_card'],
                              'known_items_pending':s['requirements_or_unresolved_items']})
        (folder/'source.json').write_text(json.dumps(public_source,ensure_ascii=False,indent=2)+'\n')
        shutil.copyfile(root/'data/public_corpus_reader.py',folder/'public_corpus_reader.py')
        shutil.copyfile(root/'runs/dataset_public_release_v1/recipe_snapshot_v1/LICENSE-CODE',folder/'LICENSE-CODE')
        scope=f'''# License scope / 라이선스 범위

Publisher: pathcosmos. Contact: miracle.yang@pathcosmos.com.

Original helper code is MIT; original explanatory prose is CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . Retain attribution and indicate modifications.
These grants do not relicense the corpus text or tokenized content.

Original dataset: {s['repository']}
Pinned revision: {s['pinned_revision']}
Original card and notices: {s['pinned_card_url']}
Scope: {s['license_scope']}

ODC-By 1.0 database conditions: https://opendatacommons.org/licenses/by/1-0/
Common Crawl terms: https://commoncrawl.org/terms-of-use
HPLT CC0 packaging only: https://creativecommons.org/publicdomain/zero/1.0/
Wikipedia attribution and applicable exceptions: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/

Code examples retain original text and in-file notices. Source license labels
are metadata, not verification of every file's original Git revision or complete
standalone LICENSE/NOTICE coverage. Known gaps are retained in source.json.
Do not interpret the source database/packaging license as blanket content rights.
'''
        (folder/'LICENSE-SCOPE.md').write_text(scope)
        c=counts[key]
        (folder/'README.md').write_text(f'''---
pretty_name: Korean Pretraining Corpus — {key}
license: other
license_name: upstream-source-specific-conditions
license_link: https://huggingface.co/datasets/{repo}/blob/main/LICENSE-SCOPE.md
configs:
- config_name: original_documents
  default: true
  data_files:
  - split: all
    path: text/*.parquet
tags:
- pretraining
- reproducibility
---

# Korean Pretraining Corpus — {key}

## 한국어

**전송 진행 중입니다. manifest의 전체 파일과 검증된 release commit이 갖춰질 때까지 전체 공개 완료 상태가 아닙니다.**

원본: [{s['repository']}]({s['pinned_card_url']}). 선택 범위: {s['selected_subset']}.
기존 정제본 {s['final_documents']:,}문서, 학습 {c['train_docs']:,}/검증 {c['holdout_docs']:,}문서입니다.
pathcosmos는 추가 정제·분리·패키징 제작자이며 원문 제작자는 원 출처를 따릅니다.

정제 Parquet의 원래 bytes와 행번호를 보존합니다. `original_documents/all`은 **train과 eval을 함께 포함**합니다.
학습에는 아래 로더로 `split="train"`을 명시하고, 검증에는 `split="validation"`을 사용하세요.
문서별 분리표·token offsets와 전체 train/validation bin을 함께 제공합니다.

출처 복원 상태: `{status}`. 복원하지 않은 URL이나 원 Git commit을 확인된 것으로 표시하지 않습니다.
원본 라이선스의 범위와 미완료 고지는 [source.json](source.json), [LICENSE-SCOPE.md](LICENSE-SCOPE.md)를 확인하세요.

## English

**Upload in progress. This is not a complete release until every manifest entry is present and a verified release commit is published.**

Upstream: [{s['repository']}]({s['pinned_card_url']}); revision `{s['pinned_revision']}`.
Original selection: {s['selected_subset']}. Locally curated documents: {s['final_documents']:,}.
pathcosmos performed additional curation and packaging; upstream authors retain attribution.

`text/` preserves the original cleaned Parquet bytes and row identity.
The `original_documents` config exposes an `all` split containing **both train and eval**.
Use the supplied reader with an explicit split to preserve the canonical holdout.
The corpus is not relabeled under the helper-code license. See source-specific license scope and known gaps.

## Canonical split reader

Download `public_corpus_reader.py` from this repository, then:

```python
from huggingface_hub import HfApi
from public_corpus_reader import download_and_iter
repo = "{repo}"
revision = HfApi(token=False).repo_info(repo, repo_type="dataset").sha
for document in download_and_iter(repo, revision=revision, split="train", max_documents=10):
    print(document["doc_id"], document["text"][:120])
```

Pin the verified release commit when available. `tokens/train/` and `tokens/validation/`
are little-endian uint16, EOS=1, using tokenizer SHA
`050f9b43cf964f0c3b29ae443a1c2eae46cb2ee0022cc010bcfea3d26acf1706`.
Different tokenizers require retokenizing text. Token offsets include the appended EOS.

[Full card and 13 language documents](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus).
Questions/private removal requests: miracle.yang@pathcosmos.com.
''')
        manifest={'status':'immutable_snapshot_upload_in_progress','source':key,'repository':repo,
            'content_inventory_id':registry['content_inventory_id'],'source_documents':documents,
            'counts':c,'provenance_status':status,'text_representation':'original_cleaned_bytes_with_canonical_split_sidecars',
            'token_format':'uint16-little-endian-eos-split-v3','eos_id':1,
            'tokenizer_sha256':'050f9b43cf964f0c3b29ae443a1c2eae46cb2ee0022cc010bcfea3d26acf1706',
            'payload_files':[{k:r[k] for k in ['path','bytes','sha256','kind']} for r in files]}
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        for p in sorted(folder.iterdir()):add(p,p.name)
        assert len({r['path'] for r in files})==len(files)
        sources.append({'source':key,'repository':repo,'files':files,'metadata_folder':str(folder.relative_to(root))})
    spec={'schema_version':1,'account':'pathcosmos','private':False,'free_only':True,'stop_on_quota_error':True,
          'baseline_text_bytes_preserved':True,'preparation_contract':'existing_cleaned_snapshot_with_explicit_provenance_gaps',
          'large_payloads_copied_locally':False,'repositories':sources,'files':sum(len(s['files']) for s in sources),
          'bytes':sum(r['bytes'] for s in sources for r in s['files'])}
    (out/'upload_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in spec.items() if k!='repositories'}))


if __name__=='__main__':prepare(Path('runs/dataset_upload_v1'))
