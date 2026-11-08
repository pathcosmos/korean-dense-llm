"""Consolidate the actual corpus bytes into the existing main public dataset.

Original repositories and immutable v1.0.0 tags are retained. No paid service,
large local copy, training change, deletion or history rewrite is performed.
"""
import argparse,fcntl,hashlib,importlib.util,itertools,json,os,shutil,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import atomic_json,upload,verify,is_quota_error
from scripts.finalize_public_corpus_v2 import discovery_metadata
from huggingface_hub import HfApi,CommitOperationAdd,DatasetCard
import yaml

REPO='pathcosmos/korean-pretrain-corpus'
VERSION='v1.1.0'
BASE=Path(PROJECT_ROOT)/'runs/dataset_unified_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebase_manifest(manifest,key):
    m=json.loads(json.dumps(manifest));m['repository']=REPO;m['layout']='single-repository'
    for item in m['source_documents']:
        for name in ['text_path','assignment_path','offset_path']:
            item[name]=key+'/'+item[name]
    for item in m['payload_files']:item['path']=key+'/'+item['path']
    return m


def record(path,name,kind='metadata'):
    st=path.stat()
    return {'local_path':str(path.relative_to(PROJECT_ROOT)),'path':name,'bytes':st.st_size,
            'sha256':sha(path),'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'kind':kind}


def prepare():
    BASE.mkdir(exist_ok=True,parents=True)
    target=BASE/'upload_spec.json'
    if target.exists():return json.loads(target.read_text())
    old=Path(PROJECT_ROOT)/'runs/dataset_upload_v1'
    previous=json.loads((old/'upload_spec.json').read_text())
    original=old/'final_metadata/index';root=BASE/'root'
    shutil.copytree(original,root,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    helper=(Path(PROJECT_ROOT)/'data/public_corpus_reader.py').read_text()
    (root/'examples/public_corpus_reader.py').write_text(helper)
    stream=(root/'examples/stream_documents.py').read_text()
    stream=stream.replace('def stream(repo,revision,split):','def stream(repo,revision,split,source=None):')
    stream=stream.replace('    response=session.get(',"    manifest_path=source+'/manifest.json' if source else 'manifest.json'\n    response=session.get(",1)
    stream=stream.replace('/resolve/{revision}/manifest.json','/resolve/{revision}/{manifest_path}')
    stream=stream.replace("p.add_argument('--split',required=True", "p.add_argument('--source',required=True);p.add_argument('--split',required=True")
    stream=stream.replace('stream(a.repo,revision,a.split)', 'stream(a.repo,revision,a.split,source=a.source)')
    (root/'examples/stream_documents.py').write_text(stream)
    records=[];keys=[]
    for source in previous['repositories']:
        key=source['source'];keys.append(key);folder=BASE/'metadata'/key;folder.mkdir(parents=True)
        for item in source['files']:
            if item['kind']!='metadata':
                r=dict(item);r['path']=key+'/'+item['path'];records.append(r);continue
            origin=Path(PROJECT_ROOT)/item['local_path']
            if item['path'] in ['README.md','manifest.json']:origin=old/'final_metadata'/key/item['path']
            dest=folder/item['path'];dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(origin,dest)
            if item['path']=='manifest.json':atomic_json(dest,rebase_manifest(json.loads(origin.read_text()),key))
            elif item['path']=='public_corpus_reader.py':dest.write_text(helper)
            elif item['path']=='README.md':
                text=origin.read_text().replace(source['repository'],REPO).replace('v1.0.0',VERSION)
                head,body=text[4:].split('\n---\n',1);meta=yaml.safe_load(head)
                meta['configs'][0]['data_files'][0]['path']=f'{key}/text/*.parquet'
                meta['license_link']=f'https://huggingface.co/datasets/{REPO}/blob/{VERSION}/{key}/LICENSE-SCOPE.md'
                body=body.replace('split="train"','split="train", source="'+key+'"')
                body=body.replace('split="validation"','split="validation", source="'+key+'"')
                body=body.replace('from public_corpus_reader import download_and_iter','from public_corpus_reader import download_and_iter')
                dest.write_text('---\n'+yaml.safe_dump(meta,allow_unicode=True,sort_keys=False)+'---\n'+body)
            records.append(record(dest,key+'/'+item['path']))
        atomic_json(folder/'release.json',{'version':VERSION,'repository':REPO,'source':key,
            'manifest_path':key+'/manifest.json','manifest_sha256':sha(folder/'manifest.json'),
            'layout':'single-repository','original_bytes_preserved':True})
        records.append(record(folder/'release.json',key+'/release.json'))
    staging=BASE/'staging_README.md'
    staging.write_text('''---
pretty_name: Korean Pretraining Corpus
language: [ko, en]
license: other
license_name: source-specific-licenses-see-scope
license_link: https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/LICENSE-SCOPE.md
task_categories: [text-generation]
tags: [text, llm, pretraining, korean-pretrain-corpus, korean, english, code, mathematics]
---
# Korean Pretraining Corpus

## 한국어

**전체 약502.6GB를 이 메인 저장소로 통합 업로드 중입니다.** 링크만 제공하는 방식이 아니라 실제 텍스트·토큰·분리·offset 파일을 소스별 폴더에 넣습니다. 최종 검증 및 v1.1.0 발행 전에는 완료 상태가 아닙니다.

'''+''.join(f'- [{key}/]({key}/)\n' for key in keys)+'''

출처별 원본 조건과 미확인 범위는 각 폴더의 source.json 및 LICENSE-SCOPE.md에 기록합니다.
텍스트 all 뷰에는 검증 문서도 포함되므로 학습에는 canonical split reader의 train 필터가 필수입니다.

## English

The actual complete existing six-source snapshot is being consolidated into this repository, under one directory per source. This is not a links-only index. The release is incomplete until the full payload verification and v1.1.0 publication finish. Preserve source-specific licenses and canonical train/validation splits.

[출처·라이선스 / Sources and license scope](SOURCE_REVIEW.md) · [연락처 / Contact](CONTACT.md)
''')
    DatasetCard(staging.read_text()).validate()
    records.append(record(staging,'README.md'))
    spec={'schema_version':1,'account':'pathcosmos','private':False,'free_only':True,'stop_on_quota_error':True,
        'baseline_text_bytes_preserved':True,'large_payloads_copied_locally':False,
        'batch_max_files':96,'batch_max_bytes':32*2**30,'layout':'single-repository',
        'repositories':[{'source':'all_six_sources','repository':REPO,'files':records,'metadata_folder':str((BASE/'metadata').relative_to(PROJECT_ROOT))}],
        'files':len(records),'bytes':sum(r['bytes'] for r in records)}
    assert len({r['path'] for r in records})==len(records)
    atomic_json(target,spec)
    api=HfApi();assert api.whoami()['name']=='pathcosmos';info=api.repo_info(REPO,repo_type='dataset');assert info.private is False
    atomic_json(BASE/'upload_state.json',{'upload_spec_sha256':sha(target),'status':'starting','verified_files':0,
        'verified_bytes':0,'repositories':{REPO:{'head':info.sha,'source':'all_six_sources'}},
        'started_utc':datetime.now(timezone.utc).isoformat(),'authorized_existing_repository':True})
    atomic_json(BASE/'preparation.json',{'status':'passed','repository':REPO,'version':VERSION,'sources':keys,
        'files':spec['files'],'bytes':spec['bytes'],'metadata_files':sum(r['kind']=='metadata' for r in records),
        'large_payload_copy_bytes':0,'existing_main_commit':info.sha})
    shutil.copytree(root,BASE/'root_template',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    return spec


def consumer(revision):
    root=BASE/'root';sys.path.insert(0,str(root/'examples'))
    module_spec=importlib.util.spec_from_file_location('unified_stream',root/'examples/stream_documents.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    keys=json.loads((BASE/'preparation.json').read_text())['sources'];checks=[]
    for key in keys:
        observed={split:list(itertools.islice(module.stream(REPO,revision,split,source=key),16)) for split in ['train','validation']}
        assert all(len(rows)==16 for rows in observed.values())
        assert all(r['split']==('eval' if split=='validation' else split) for split,rows in observed.items() for r in rows)
        assert not {r['doc_id'] for r in observed['train']}&{r['doc_id'] for r in observed['validation']}
        checks.append({'repository':REPO,'revision':revision,'source':key,'sample_documents':32,'canonical_splits_disjoint':True})
        atomic_json(BASE/'consumer_progress.json',{'sources_passed':len(checks),'total':6})
    from scripts.check_public_corpus_consumer import tiny_cpu_check
    m=json.loads((BASE/'metadata/wiki_ko/manifest.json').read_text())
    result={'status':'passed','anonymous_source_checks':checks,'bounded_cpu_training':tiny_cpu_check(REPO,revision,m),
            'full_clean_room_3B_training_reproduction':False,'layout':'single-repository'}
    atomic_json(BASE/'consumer_validation.json',result);return result


def finalize(spec,state):
    root=BASE/'root';head=state['repositories'][REPO]['head']
    shutil.copytree(BASE/'root_template',root,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    checkfile=BASE/'consumer_validation.json'
    checked=json.loads(checkfile.read_text()) if checkfile.exists() else consumer(head)
    assert checked['status']=='passed'
    keys=json.loads((BASE/'preparation.json').read_text())['sources']
    banners={
        'ko':'전체 데이터는 이 메인 저장소 하나에 들어 있습니다. 별도 데이터 저장소를 받을 필요 없이 소스별 폴더에서 텍스트·토큰·분리·문서 위치를 받으세요.',
        'en':'All actual corpus files are in this single repository. Download text, tokens, splits and offsets from the source directories; no separate data repositories are required.',
        'zh-CN':'所有实际数据文件都在这个主仓库中。请从各来源目录下载文本、词元、划分和偏移，无需访问其他数据仓库。',
        'zh-TW':'所有實際資料檔案都在這個主儲存庫中。請從各來源目錄下載文本、詞元、切分及偏移，無須使用其他資料儲存庫。',
        'ja':'全データファイルをこのメインリポジトリに統合しました。各ソースのフォルダーからテキスト・トークン・分割・文書位置を取得でき、別のデータリポジトリは不要です。',
        'fr':'Tous les fichiers du corpus sont réunis dans ce dépôt principal. Les dossiers par source contiennent textes, tokens, partitions et positions ; aucun autre dépôt de données n’est nécessaire.',
        'de':'Alle Korpusdateien liegen in diesem Hauptrepository. Die Quellordner enthalten Texte, Tokens, Aufteilungen und Dokumentpositionen; separate Datenrepositories sind nicht erforderlich.',
        'es':'Todos los archivos del corpus están en este repositorio principal. Los directorios por fuente contienen textos, tokens, particiones y posiciones; no se requieren otros repositorios.',
        'pt-BR':'Todos os arquivos do corpus estão neste repositório principal. As pastas por fonte contêm textos, tokens, divisões e posições; não é necessário baixar outros repositórios.',
        'hi':'पूरा डेटा इसी मुख्य रिपॉज़िटरी में है। स्रोत फ़ोल्डरों से पाठ, टोकन, विभाजन और दस्तावेज़ स्थितियाँ डाउनलोड करें; अलग डेटा रिपॉज़िटरी की आवश्यकता नहीं है।',
        'ar':'جميع ملفات البيانات موجودة في هذا المستودع الرئيسي. تحتوي مجلدات المصادر على النصوص والرموز والتقسيمات والمواضع؛ لا حاجة إلى مستودعات بيانات منفصلة.',
        'he':'כל קובצי הקורפוס נמצאים במאגר הראשי הזה. תיקיות המקורות כוללות טקסטים, טוקנים, חלוקות ומיקומים; אין צורך במאגרי נתונים נפרדים.',
        'ru':'Все файлы корпуса находятся в этом основном репозитории. Папки источников содержат тексты, токены, разбиения и позиции документов; отдельные репозитории данных не требуются.'}
    for p in [root/'README.md',*sorted((root/'cards').glob('*.md'))]:
        text=p.read_text().replace('v1.0.0',VERSION)
        if p.parent.name=='cards':
            lang=p.name.removeprefix('README.').removesuffix('.md');lines=text.split('\n',1)
            if banners[lang] not in text:text=lines[0]+'\n\n'+banners[lang]+'\n'+lines[1]
        else:
            text=text.replace('## 한국어\n','## 한국어\n\n'+banners['ko']+'\n',1)
            text=text.replace('## English\n','## English\n\n'+banners['en']+'\n',1)
            first=text.index('## Download available material');last=text.index('## Citation',first)
            text=text[:first]+'''## 전체 다운로드 / Download the complete corpus

전체 약502.6GB를 이 저장소 하나에서 받습니다. 아래 코드는 학습을 시작하지 않습니다.
This downloads the complete corpus from this repository, not only preview files. It does not start training.

```python
from huggingface_hub import snapshot_download
path = snapshot_download(
    "pathcosmos/korean-pretrain-corpus", repo_type="dataset",
    revision="v1.1.0", local_dir="korean-pretrain-corpus", token=False,
)
print(path)
```

텍스트 학습에는 canonical train 분리를 적용하세요. 원래 텍스트의 all 뷰에는 validation도 있습니다.
Use the explicit canonical train split for training; the original all-documents view also includes validation.
See [REPRODUCE.md](REPRODUCE.md) for selective downloads and streaming.

'''+text[last:]
            text=text.replace('[6개 공개 데이터 저장소](DATA_REPOSITORIES.md)','[이 저장소의 6개 데이터 폴더](DATA_REPOSITORIES.md)')
            text=text.replace('lists the six verified source commits','lists the six source manifests and verified data commit in this repository')
            header,body=text[4:].split('\n---\n',1);meta=yaml.safe_load(header)
            meta['configs']=[{'config_name':key,'data_files':[{'split':'all','path':key+'/text/*.parquet'}],**({'default':True} if i==0 else {})} for i,key in enumerate(keys)]
            text='---\n'+yaml.safe_dump(meta,allow_unicode=True,sort_keys=False)+'---\n'+body
        p.write_text(text)
    old=json.loads((Path(PROJECT_ROOT)/'runs/dataset_upload_v1/upload_spec.json').read_text())
    for p in root.rglob('*.md'):
        text=p.read_text()
        for source in old['repositories']:
            text=text.replace('https://huggingface.co/datasets/'+source['repository'],
                'https://huggingface.co/datasets/'+REPO+'/tree/'+VERSION+'/'+source['source'])
        text=text.replace('The corpus index links the six separate datasets; all existing repositories remain available.',
            'The complete corpus is stored in the six source directories in this repository.')
        text=text.replace('Corpus index / 통합 안내','Main repository / 메인 저장소')
        p.write_text(text)
    (root/'DATA_REPOSITORIES.md').write_text('# 메인 저장소의 데이터 / Data in this repository\n\n전체 데이터는 이 저장소 하나에 있습니다. All actual files are stored here.\n\n| Source | Files |\n|---|---|\n'+''.join(f'| {key} | [{key}/]({key}/) |\n' for key in keys)+'\n각 폴더: text/, tokens/train/, tokens/validation/, splits/, offsets/, manifest.json, source.json, LICENSE-SCOPE.md.\n')
    (root/'REPRODUCE.md').write_text('''# 재사용·재작업 / Reuse and reproduction

전체 정제 데이터와 토큰은 `pathcosmos/korean-pretrain-corpus` 한 곳에 있습니다.
All actual corpus files are in this single repository. Preserve the v1.1.0 revision.

## 전체 다운로드 / Complete download

```python
from huggingface_hub import snapshot_download
snapshot_download("pathcosmos/korean-pretrain-corpus", repo_type="dataset",
    revision="v1.1.0", local_dir="korean-pretrain-corpus", token=False)
```

전체 약502.6GB입니다. 토큰만 필요하면 `allow_patterns=["*/tokens/*", "tokenizer/*", "recipe/*"]`로 선택하세요.
For text only, use `allow_patterns=["*/text/*", "*/splits/*", "*/manifest.json", "examples/*"]`.
Original text includes both train and validation; apply the canonical split reader before training.

## 로그인 없는 표본 읽기 / Anonymous streaming

```bash
python examples/stream_documents.py --repo pathcosmos/korean-pretrain-corpus --revision v1.1.0 --source wiki_ko --split train --limit 10
```

`--source` selects wiki_ko, fineweb2_ko, hplt_ko, fineweb_edu, owm, or github_code_v1.
Use `--split validation` for evaluation. The reader verifies row identities and content against canonical assignments.

## 학습 recipe / Training recipe

`recipe/main.portable.json` and `recipe/mix.portable.json` reference source directories already present in this download.
Set PROJECT_ROOT to the downloaded root when using the native loader; preserve tokenizer/ and train/validation boundaries.
`main.original.json` is the historical original training configuration. No training starts automatically.
Choose your own resource, output and checkpoint retention settings. Full independent 3B training reproduction remains unverified.

## 원본 및 라이선스 / Upstream and license scope

Each `<source>/manifest.json` retains pinned upstream repo/revision/path/URL/SHA256 and original document identity.
Raw upstream material is referenced rather than duplicated. See SOURCE_REVIEW.md, SOURCES.md and each source.json/ LICENSE-SCOPE.md.
Wikipedia attribution candidates are supplied. Other document-URL, original Git commit and full standalone notice gaps remain documented.
Original source conditions apply; MIT covers helper code and CC BY 4.0 covers original project prose only.
''')
    atomic_json(root/'recipe/source_repositories.json',{key:REPO for key in keys})
    atomic_json(root/'recipe/source_paths.json',{key:key+'/' for key in keys})
    recipe=(root/'recipe/README.md').read_text().replace('Source repositories supply manifest.json, splits/, offsets/ and public_corpus_reader.py.',
        'This single repository supplies <source>/manifest.json, <source>/splits/, <source>/offsets/ and examples/public_corpus_reader.py.')
    (root/'recipe/README.md').write_text(recipe)
    sources={key:{'manifest':key+'/manifest.json','manifest_sha256':sha(BASE/'metadata'/key/'manifest.json'),
        'directory':key+'/','counts':json.loads((BASE/'metadata'/key/'manifest.json').read_text())['counts']} for key in keys}
    release={'version':VERSION,'repository':REPO,'layout':'single-repository','data_commit':head,'sources':sources,
        'verified_source_files':spec['files'],'verified_source_bytes':spec['bytes'],'documents':128856258,'tokens':122402393813,
        'known_provenance_gaps_documented':True,'full_clean_room_3B_training_reproduction':False}
    atomic_json(root/'release.json',release);atomic_json(root/'CONSUMER_VALIDATION.json',checked)
    status=json.loads((root/'PUBLICATION_STATUS.json').read_text());status.update(layout='single-repository',pinned_version=VERSION,
        corpus_text_public=True,token_bins_public=True,portable_splits_public=True,document_offsets_public=True,payload_transfer_status='complete')
    atomic_json(root/'PUBLICATION_STATUS.json',status)
    generated=[record(p,str(p.relative_to(root))) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='CHECKSUMS.json' and '__pycache__' not in p.parts and p.suffix!='.pyc']
    # Staging README is superseded by the final root card. All source paths remain frozen.
    entries={r['path']:{k:r[k] for k in ['path','bytes','sha256']} for r in spec['repositories'][0]['files'] if r['path']!='README.md'}
    entries.update({r['path']:{k:r[k] for k in ['path','bytes','sha256']} for r in generated})
    atomic_json(root/'CHECKSUMS.json',{'version':VERSION,'files':list(entries.values())})
    generated.append(record(root/'CHECKSUMS.json','CHECKSUMS.json'))
    api=HfApi();commit_file=BASE/'final_commit.json'
    if commit_file.exists():revision=json.loads(commit_file.read_text())['commit']
    else:
        current=api.repo_info(REPO,repo_type='dataset').sha
        if current!=head:
            verify(HfApi(token=False),REPO,current,generated,anonymous=True);revision=current
        else:
            commit=api.create_commit(REPO,repo_type='dataset',parent_commit=head,
                commit_message='Publish complete corpus in one repository with canonical splits and v1.1.0 instructions',
                operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=Path(PROJECT_ROOT)/r['local_path']) for r in generated])
            revision=commit.oid
        atomic_json(commit_file,{'commit':revision})
    api.create_tag(REPO,tag=VERSION,revision=revision,repo_type='dataset',exist_ok=True)
    assert api.repo_info(REPO,repo_type='dataset',revision=VERSION).sha==revision
    public=HfApi(token=False);all_records=list(entries.values())+[generated[-1]]
    for i in range(0,len(all_records),50):verify(public,REPO,revision,all_records[i:i+50],anonymous=True)
    listed=set(public.list_repo_files(REPO,repo_type='dataset',revision=revision))
    assert listed-{'.gitattributes'}=={r['path'] for r in all_records},'Unaccounted or missing repository files'
    info=public.repo_info(REPO,repo_type='dataset',revision=revision,files_metadata=True)
    result={'status':'passed','repository':REPO,'version':VERSION,'commit':revision,'layout':'single-repository',
        'files':len(info.siblings),'bytes':sum(r.size for r in info.siblings),'source_files':spec['files']-1,
        'consumer_validation':checked,'paid_services_used':False,'legacy_repositories_preserved':True}
    atomic_json(BASE/'report.json',result);atomic_json(BASE/'release.done',{'report_sha256':sha(BASE/'report.json')})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--rss-limit-gib',type=int,choices=(4,8,16),default=4);a=p.parse_args()
    spec=prepare()
    if not a.run:print(json.dumps({'status':'prepared','files':spec['files'],'bytes':spec['bytes']}));return
    lock=(BASE/'release.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (BASE/'release.pid').write_text(str(os.getpid())+'\n');os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
    stop=threading.Event();phase={'value':'uploading'}
    def monitor():
        while not stop.is_set():
            state=json.loads((BASE/'upload_state.json').read_text())
            rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
            atomic_json(BASE/'release_state.json',{'status':'running','phase':phase['value'],'done':state.get('verified_files',0),
                'total':spec['files'],'verified_bytes':state.get('verified_bytes',0),'target_bytes':spec['bytes'],
                'rss_bytes':rss,'rss_limit_gib':a.rss_limit_gib,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()})
            if rss>a.rss_limit_gib*2**30:
                error={'status':'error','reason':'rss_guard_gib','rss_limit_gib':a.rss_limit_gib,'rss_bytes':rss,'pid':os.getpid()}
                atomic_json(BASE/'error.json',error);atomic_json(BASE/'release_state.json',error)
                os._exit(75)
            stop.wait(30)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        state=upload(BASE/'upload_spec.json');assert state['status']=='all_source_payloads_verified'
        atomic_json(BASE/'payload_report.json',state);phase['value']='validating_and_finalizing'
        result=finalize(spec,state);stop.set();thread.join()
        atomic_json(BASE/'release_state.json',{'status':'complete','phase':'complete','done':spec['files'],'total':spec['files'],'pid':os.getpid()})
        print(json.dumps({k:v for k,v in result.items() if k!='consumer_validation'}),flush=True)
        with (Path(PROJECT_ROOT)/'logs/pipeline_events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'stage':'corpus_single_repository_release','state':'complete','commit':result['commit']})+'\n')
    except Exception as e:
        stop.set();thread.join();result={'status':'failed','error_type':type(e).__name__,
            'http_status':getattr(getattr(e,'response',None),'status_code',None),'quota_error':is_quota_error(e)}
        atomic_json(BASE/'error.json',result);atomic_json(BASE/'release_state.json',{'status':'error',**result})
        print(json.dumps(result),flush=True);raise SystemExit(1)


if __name__=='__main__':main()
