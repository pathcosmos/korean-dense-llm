"""Publish the exact immutable files referenced by the running 3B training mix.

No training process, data file, original configuration or existing HF repository
is changed. Only the new user-selected public dataset is written.
"""
import argparse,fcntl,hashlib,json,os,shutil,sys,threading,time,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import upload,atomic_json,verify,is_quota_error
from huggingface_hub import HfApi,CommitOperationAdd,DatasetCard

BASE=Path(PROJECT_ROOT)/'runs/training_inputs_release_v1'
REPO='pathcosmos/korean-pretrain-training-data'
VERSION='v1.0.0'
MAIN='pathcosmos/korean-pretrain-corpus'
MAIN_COMMIT='b9fd2dae2d00acfea0b1068dd25aab4195a42a1d'
CORE=['model/model.py','train/__init__.py','train/train.py','train/data.py',
      'train/runtime.py','common/paths.py','common/data_contracts.py']
LANGS={
 'ko':('한국어','현재 3B 본학습이 실제 참조하는 토큰 입력입니다. train 726개와 고정 평가용 6개만 원본 bytes 그대로 제공합니다. 전체 검증 코퍼스는 종합 저장소에 있습니다.'),
 'en':('English','These are the exact token inputs referenced by our current 3B training run: 726 train files and six fixed evaluation files, with original bytes preserved. The full validation corpus remains in the main corpus repository.'),
 'zh-CN':('简体中文','本数据集提供当前 3B 预训练实际引用的词元输入：726 个训练文件和 6 个固定评估文件，保留原始字节。完整验证语料位于主语料仓库。'),
 'zh-TW':('繁體中文','本資料集提供目前 3B 預訓練實際引用的詞元輸入：726 個訓練檔及 6 個固定評估檔，保留原始位元組。完整驗證語料位於主語料儲存庫。'),
 'ja':('日本語','現在の3B事前学習が実際に参照するトークン入力です。学習726ファイルと固定評価6ファイルの元のバイト列を保持しています。検証コーパス全体はメインリポジトリにあります。'),
 'fr':('Français','Ce jeu contient les entrées tokenisées réellement référencées par notre entraînement 3B : 726 fichiers d’entraînement et six fichiers d’évaluation fixe, avec les octets originaux préservés. Le corpus complet de validation reste dans le dépôt principal.'),
 'de':('Deutsch','Dies sind die tatsächlichen Token-Eingaben unseres aktuellen 3B-Trainings: 726 Trainingsdateien und sechs Dateien für die feste Evaluation, mit unveränderten Originalbytes. Der vollständige Validierungskorpus liegt im Hauptrepository.'),
 'es':('Español','Son las entradas tokenizadas que utiliza nuestro entrenamiento 3B actual: 726 archivos de entrenamiento y seis de evaluación fija, con sus bytes originales. El corpus completo de validación está en el repositorio principal.'),
 'pt-BR':('Português (Brasil)','Estas são as entradas tokenizadas do nosso treinamento 3B atual: 726 arquivos de treino e seis de avaliação fixa, com os bytes originais preservados. O corpus completo de validação está no repositório principal.'),
 'hi':('हिन्दी','यह मौजूदा 3B प्रशिक्षण में संदर्भित वास्तविक टोकन इनपुट है: 726 प्रशिक्षण फ़ाइलें और निश्चित मूल्यांकन की छह फ़ाइलें, मूल बाइट्स सहित। पूरा सत्यापन कॉर्पस मुख्य रिपॉज़िटरी में है।'),
 'ar':('العربية','هذه مدخلات الرموز الفعلية المستخدمة في تدريب 3B الحالي: 726 ملف تدريب وستة ملفات للتقييم الثابت مع الحفاظ على البايتات الأصلية. مجموعة التحقق الكاملة موجودة في مستودع البيانات الرئيسي.'),
 'he':('עברית','אלו קובצי הטוקנים שאליהם מפנה אימון 3B הנוכחי: 726 קובצי אימון ושישה קובצי הערכה קבועה, עם הבתים המקוריים ללא שינוי. קורפוס האימות המלא נמצא במאגר הראשי.'),
 'ru':('Русский','Это фактические токенизированные входы текущего обучения 3B: 726 обучающих файлов и шесть файлов фиксированной оценки с сохранением исходных байтов. Полный проверочный корпус находится в основном репозитории.')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_input_paths(config,mix):
    """Follow the actual trainer's explicit-config-before-mix evaluation rule."""
    train=[p for source in mix['sources'].values() for p in source['files']]
    holdout=config.get('holdout') or mix.get('holdout',[])
    if len(set(train+holdout))!=len(train+holdout):raise ValueError('Repeated native input')
    return train+holdout


def record(path,name,kind):
    st=path.stat();return {'path':name,'local_path':str(path.relative_to(PROJECT_ROOT)),
        'bytes':st.st_size,'sha256':sha(path),'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'kind':kind}


def readme(complete):
    state_ko='전체 입력 파일 공개·원격 검증 완료.' if complete else '업로드 진행 중입니다. release.json과 v1.0.0 발행 전에는 전체 완료 상태가 아닙니다.'
    state_en='All input files are public and remotely verified.' if complete else 'Upload in progress; the release is incomplete until release.json and v1.0.0 are published.'
    nav=' · '.join(f'[{name}](cards/README.{code}.md)' for code,(name,_) in LANGS.items())
    return f'''---
pretty_name: Korean Pretrain — Actual 3B Training Inputs
language: [ko, en]
license: other
license_name: source-specific-conditions
license_link: https://huggingface.co/datasets/{REPO}/blob/main/LICENSE-SCOPE.md
task_categories: [text-generation]
tags: [text, llm, pretraining, training-data, tokenized, uint16, reproducibility, korean, english, code, mathematics]
---
# Korean Pretrain — Actual 3B Training Inputs

{nav}

## 한국어

**{state_ko}**

{LANGS['ko'][1]}

현재 실행의 `main.json`과 `mix.json`이 참조하는 파일을 고정했습니다. 새로 정제·재토큰화·재혼합하지 않았으며 파일 순서·mix source 순서·가중치·seed·sampler v2를 보존합니다. 학습 설정에 연결된 전체 입력이며, 이미 소비한 토큰만 추출한 로그나 미리 섞은 단일 시퀀스가 아닙니다. 샘플링은 복원추출 방식입니다.

| 항목 | 수량 |
|---|---:|
| train 파일 | 726 |
| 고정 평가 파일 | 6 |
| train 토큰 | 121,174,609,758 |
| 고정 평가 파일의 토큰 | 14,124,617 |
| token bin 합계 bytes | 242,377,468,750 (약242.38GB) |
| vocab / dtype / EOS / seq_len | 65,536 / little-endian uint16 / 1 / 4,096 |

평가6개는 실제 본학습의 고정 probe 입력입니다. 전체 validation 726파일 및 정제 텍스트·split·offset은 [종합 저장소](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0)에서 제공합니다. 카드 번역 언어가 데이터 언어를 뜻하지 않습니다.

동일한 토크나이저, 원본 mix/config 및 실제 native sampler/학습 핵심 코드를 포함합니다. 데이터 원본 조건은 [LICENSE-SCOPE.md](LICENSE-SCOPE.md)와 sources/를 따릅니다. 자체 코드 MIT·자체 문서 CC BY4.0은 원본 데이터의 조건을 대체하지 않습니다. 문서별 원 URL·원 Git commit·독립 고지의 기존 미확인 범위도 유지합니다.

현재 학습의 체크포인트/optimizer/RNG는 포함하지 않습니다. 데이터만으로 진행 중인 학습 상태를 복원할 수는 없으며, 다른 설치 경로에서는 strict checkpoint contract의 절대경로 identity도 달라집니다. 전체 독립3B 학습 재현 검증을 완료했다고 주장하지 않습니다.

## English

**{state_en}**

{LANGS['en'][1]}

The exact native mix, input order, source order, sampling weights, seed and sampler v2 are preserved. Files were not re-filtered, re-tokenized or pre-interleaved. This is the complete configured input set, not a log of tokens already consumed. Training samples with replacement.

There are 121,174,609,758 train tokens plus 14,124,617 tokens in six selected evaluation files: **732 token binaries / 242,377,468,750 bytes**. The full validation set is not implied. Format: little-endian uint16, vocabulary 65,536, EOS=1, sequence length=4,096.

Tokenizer bytes and special IDs match the original. Source terms and known attribution gaps remain applicable; MIT covers helper code and CC BY4.0 covers original prose only. No model checkpoints or optimizer/RNG states are included. Strict checkpoint identity contains absolute paths; moving data does not by itself provide exact checkpoint resume. Full independent 3B training reproduction remains unverified.

## Download / 다운로드

```python
from huggingface_hub import snapshot_download
root = snapshot_download("{REPO}", repo_type="dataset",
    revision="{VERSION}", local_dir="korean-pretrain-training-data", token=False)
```

Use a new download directory. The relative input paths match `config/main.original.json` and the byte-identical native `data/tokens/final_v1/mix.json`.
`config/main.example.json` changes only output location and checkpoint retention for a fresh run. It does not start training.
Verify CHECKSUMS.json; read TRAINING_GUIDE.md before using the native code.

## Tokenizer / 토크나이저

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
cache = snapshot_download("{REPO}", repo_type="dataset",
    revision="{VERSION}", allow_patterns=["tokenizer/*"], token=False)
tokenizer = AutoTokenizer.from_pretrained(Path(cache) / "tokenizer")
```

## Sources and sampling / 출처·가중치

| Source | Weight | Main corpus files | Standalone dataset |
|---|---:|---|---|
| wiki_ko | 0.005 | [wiki_ko](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/wiki_ko) | [Wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia) |
| fineweb2_ko | 0.29464577095297395 | [fineweb2_ko](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/fineweb2_ko) | [FineWeb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2) |
| hplt_ko | 0.20035422904702602 | [hplt_ko](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/hplt_ko) | [HPLT2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2) |
| fineweb_edu | 0.3 | [fineweb_edu](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/fineweb_edu) | [FineWeb-Edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu) |
| github_code_v1 | 0.1 | [github_code_v1](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/github_code_v1) | [Code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code) |
| owm | 0.1 | [owm](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/owm) | [Math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math) |

Publisher: pathcosmos · contact/private removal requests: miracle.yang@pathcosmos.com.
See manifest.json for pinned byte hashes and original source selection. Cite this dataset revision and the upstream sources actually used.
'''


def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'upload_spec.json').exists():return json.loads((BASE/'upload_spec.json').read_text())
    inv=json.loads((BASE/'input_inventory.json').read_text());root=BASE/'metadata';root.mkdir(exist_ok=True)
    project=Path(PROJECT_ROOT);config_path=project/'runs/3b_full_v1/main.json';config=json.loads(config_path.read_text())
    mix_path=project/config['mix'];mix=json.loads(mix_path.read_text());assert set(config['holdout']).issubset(mix['holdout'])
    expected=effective_input_paths(config,mix)
    assert expected==[r['path'] for r in inv['files']] and len(set(expected))==732
    assert sha(config_path)==inv['native_config_sha256'] and sha(mix_path)==inv['native_mix_sha256']
    for name in CORE:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,target)
    for path,name in [(config_path,'config/main.original.json'),(mix_path,config['mix'])]:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    example=dict(config,out_dir='runs/native-inputs-reproduction',keep_ckpts=2)
    atomic_json(root/'config/main.example.json',example)
    corpus=project/'runs/dataset_unified_v1/root'
    for name in ['LICENSE-CODE','LICENSE-DOCS','LICENSE-SCOPE.md','SOURCES.md','CONTACT.md',
                 'metadata/code_repository_labels.jsonl.gz','metadata/source_registry.json',
                 'tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json']:
        src=corpus/name
        if not src.exists() and name=='LICENSE-DOCS':continue
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    assert sha(root/'tokenizer/tokenizer.json')==mix['tokenizer_sha256']
    for key in mix['sources']:
        for name in ['source.json','LICENSE-SCOPE.md']:
            dest=root/'sources'/key/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(project/'runs/dataset_unified_v1/metadata'/key/name,dest)
    environment={p:importlib.metadata.version(p) for p in ['torch','numpy','huggingface_hub','tokenizers','transformers','requests']}
    atomic_json(root/'ENVIRONMENT.json',{'tested_packages':environment,'native_core_files':{n:sha(root/n) for n in CORE}})
    (root/'requirements-reader.txt').write_text('\n'.join(p+'=='+environment[p] for p in ['numpy','huggingface_hub','requests','tokenizers','transformers'])+'\n')
    (root/'TRAINING_GUIDE.md').write_text('''# Native inputs / 실제 학습 입력

Download into a new directory. Native relative paths are preserved under data/tokens/final_v1/.
All 726 train files and exactly six configured fixed-probe validation files are included. Sampling weights and source/file order are unchanged.
The six explicit config holdout files take precedence over the mix fallback list of726 validation files, as in the running native trainer; both original JSON files are preserved byte for byte. The unused720 fallback validation files are not included here: do not clear the supplied config holdout list. The complete validation set remains in the main corpus repository.
The six validation files contain more tokens than the fixed64windows/file probe consumes; no full-validation-set claim is made.

The original native mix is byte-identical, including historical expected_exposures fields. The runtime sampler uses weights/files and does not use those historical exposure estimates.
Native code is the exact current core, not an automatic deployment of the server's operational supervisor/retention policy.
Install a matching PyTorch/CUDA environment using ENVIRONMENT.json as a reference. Reader requirements alone do not install native training dependencies.
For a fresh run use config/main.example.json and choose hardware/world size/output retention deliberately. No training starts during download or verification.

The trainer's strict checkpoint contract includes absolute paths, world size, learning parameters, input hashes and code hashes. A new directory is a fresh-run input package, not a bypass for checkpoint identity or a complete recovery of optimizer/RNG/checkpoints.
Full independent 3B training reproduction remains unverified. No new full training is started here.

Original source conditions and incomplete document URL/Git commit/standalone notice coverage remain documented in sources/ and LICENSE-SCOPE.md.
Project helper code is MIT; original prose is CC BY4.0. Tokenization does not relicense original content.
''')
    nav=' · '.join(f'[{name}](README.{code}.md)' for code,(name,_) in LANGS.items())
    for code,(name,text) in LANGS.items():
        path=root/'cards'/f'README.{code}.md';path.parent.mkdir(exist_ok=True)
        path.write_text(f'# Actual 3B Training Inputs — {name}\n\n{nav}\n\n{text}\n\n'
            '**726 train + 6 validation = 732 files / 242,377,468,750 bytes**\n\n'
            'uint16 little-endian · EOS=1 · vocab=65,536 · sampler=v2 · seq_len=4,096\n\n'
            f'[한국어 / English](../README.md) · [Sources and license scope](../LICENSE-SCOPE.md) · [Training guide](../TRAINING_GUIDE.md) · [Main corpus](https://huggingface.co/datasets/{MAIN}/tree/v1.1.0)\n\n'
            'Code: MIT. Original prose: CC BY4.0. Source-specific conditions and documented attribution gaps remain applicable.\n'
            'No model checkpoints or optimizer/RNG states are included. Full independent 3B reproduction remains unverified.\n'
            'pathcosmos · miracle.yang@pathcosmos.com\n')
    (root/'README.md').write_text(readme(False));DatasetCard((root/'README.md').read_text()).validate()
    manifest={'status':'uploading','repository':REPO,'source_corpus':MAIN,'source_corpus_commit':MAIN_COMMIT,
        'native_config_sha256':inv['native_config_sha256'],'native_mix_sha256':inv['native_mix_sha256'],
        'tokenizer_sha256':mix['tokenizer_sha256'],'source_order':list(mix['sources']),
        'train_files':726,'validation_files':6,'token_bytes':inv['bytes'],
        'train_tokens':121174609758,'evaluation_file_tokens':14124617,'files':inv['files']}
    atomic_json(root/'manifest.json',manifest)
    previous={r['local_path']:r for s in json.loads((project/'runs/dataset_upload_v1/upload_spec.json').read_text())['repositories'] for r in s['files']}
    records=[dict(previous[r['local_path']],path=r['path'],kind=r['kind']) for r in inv['files']]
    records += [record(p,str(p.relative_to(root)),'metadata') for p in sorted(root.rglob('*')) if p.is_file()]
    assert len({r['path'] for r in records})==len(records)
    assert sum(r['kind']=='metadata' for r in records)<96
    spec={'schema_version':1,'account':'pathcosmos','private':False,'free_only':True,'stop_on_quota_error':True,
        'batch_max_files':96,'batch_max_bytes':32*2**30,'files':len(records),'bytes':sum(r['bytes'] for r in records),
        'repositories':[{'source':'actual_native_3b_inputs','repository':REPO,'metadata_folder':str(root.relative_to(PROJECT_ROOT)),'files':records}]}
    atomic_json(BASE/'upload_spec.json',spec);return spec


def validate_public(revision):
    import requests,numpy as np
    from train.data import MixSampler,HoldoutSet
    project=Path(PROJECT_ROOT);config=json.loads((project/'runs/3b_full_v1/main.json').read_text());sampler=MixSampler(config['mix'],config['seq_len'],config['seed'],config['sampler_version'])
    mix=json.loads((project/config['mix']).read_text());session=requests.Session();session.trust_env=False;proofs=[]
    for step,rank,micro in [(0,0,0),(1,7,0),(1234,0,3),(57000,7,3)]:
        expected=sampler.batch(step,rank,micro,4);rng=np.random.default_rng([config['seed'],step,rank,micro])
        for row in range(4):
            source=int(rng.choice(len(sampler.w),p=sampler.w));shard=int(rng.choice(len(sampler.shards[source]),p=sampler.shard_probabilities[source]));mm,n=sampler.shards[source][shard]
            offset=int(rng.integers(0,n-(config['seq_len']+1)+1));path=mix['sources'][sampler.names[source]]['files'][shard]
            start=offset*2;size=(config['seq_len']+1)*2
            with session.get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}',
                headers={'Range':f'bytes={start}-{start+size-1}','Accept-Encoding':'identity'},stream=True,timeout=60) as response:
                response.raise_for_status();assert response.status_code==206,'Ranged verification must not download entire shards'
                data=response.raw.read(size)
            tokens=np.frombuffer(data,dtype='<u2').astype(np.int64)
            assert np.array_equal(tokens[:-1],expected[0][row].numpy()) and np.array_equal(tokens[1:],expected[1][row].numpy())
            proofs.append({'step':step,'rank':rank,'micro':micro,'row':row,'path':path,'token_offset':offset,'window_sha256':hashlib.sha256(data).hexdigest()})
    for path in config['holdout']:
        size=(config['seq_len']+1)*2
        with session.get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}',headers={'Range':f'bytes=0-{size-1}','Accept-Encoding':'identity'},stream=True,timeout=60) as response:
            response.raise_for_status();data=response.raw.read(size)
        with (project/path).open('rb') as local:assert data==local.read(size)
    result={'status':'passed','native_sampler_windows_exact':proofs,'evaluation_files_checked':6,
        'tokenizer_sha256':sha(BASE/'metadata/tokenizer/tokenizer.json'),'cuda_used':False,'full_3B_training_reproduction':False}
    atomic_json(BASE/'consumer_validation.json',result);return result


def finish(spec,state):
    head=state['repositories'][REPO]['head'];consumer_path=BASE/'consumer_validation.json'
    checked=json.loads(consumer_path.read_text()) if consumer_path.exists() else validate_public(head)
    assert checked['status']=='passed';final=BASE/'final';final.mkdir(exist_ok=True)
    (final/'README.md').write_text(readme(True));m=json.loads((BASE/'metadata/manifest.json').read_text());m['status']='complete';atomic_json(final/'manifest.json',m)
    atomic_json(final/'CONSUMER_VALIDATION.json',checked)
    atomic_json(final/'release.json',{'version':VERSION,'repository':REPO,'data_commit':head,'native_mix_sha256':m['native_mix_sha256'],
        'native_config_sha256':m['native_config_sha256'],'tokenizer_sha256':m['tokenizer_sha256'],'token_files':732,
        'token_bytes':242377468750,'consumer_validation':'passed','full_3B_training_reproduction':False})
    records={r['path']:r for r in spec['repositories'][0]['files']}
    updates=[record(p,p.name,'metadata') for p in sorted(final.glob('*')) if p.is_file() and p.name!='CHECKSUMS.json']
    records.update({r['path']:r for r in updates});atomic_json(final/'CHECKSUMS.json',{'files':[{k:r[k] for k in ['path','bytes','sha256']} for r in records.values()]})
    updates.append(record(final/'CHECKSUMS.json','CHECKSUMS.json','metadata'));api=HfApi();saved=BASE/'final_commit.json'
    if saved.exists():revision=json.loads(saved.read_text())['commit']
    else:
        current=api.repo_info(REPO,repo_type='dataset').sha
        if current!=head:verify(HfApi(token=False),REPO,current,updates,anonymous=True);revision=current
        else:
            commit=api.create_commit(REPO,repo_type='dataset',parent_commit=head,commit_message='Finalize byte-identical native 3B training inputs and consumer verification',
                operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=Path(PROJECT_ROOT)/r['local_path']) for r in updates]);revision=commit.oid
        atomic_json(saved,{'commit':revision})
    api.create_tag(REPO,tag=VERSION,revision=revision,repo_type='dataset',exist_ok=True)
    public=HfApi(token=False);assert public.repo_info(REPO,repo_type='dataset',revision=VERSION).sha==revision
    all_records=list(records.values())+[updates[-1]]
    for i in range(0,len(all_records),50):verify(public,REPO,revision,all_records[i:i+50],anonymous=True)
    assert set(public.list_repo_files(REPO,repo_type='dataset',revision=revision))-{'.gitattributes'}=={r['path'] for r in all_records}
    result={'status':'passed','repository':REPO,'version':VERSION,'commit':revision,'token_files':732,'token_bytes':242377468750,
            'files':len(all_records)+1,'consumer_validation':checked,'paid_services_used':False}
    atomic_json(BASE/'report.json',result);atomic_json(BASE/'release.done',{'report_sha256':sha(BASE/'report.json')});return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args();spec=prepare()
    if not args.run:print(json.dumps({'status':'prepared','repository':REPO,'files':spec['files'],'bytes':spec['bytes']}));return
    lock=(BASE/'release.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(BASE/'release.pid').write_text(str(os.getpid())+'\n')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:]);stop=threading.Event();phase={'value':'uploading'}
    def monitor():
        while not stop.is_set():
            f=BASE/'upload_state.json';s=json.loads(f.read_text()) if f.exists() else {}
            rss=int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
            observation={'status':'running','phase':phase['value'],'done':s.get('verified_files',0),'total':spec['files'],
                'verified_bytes':s.get('verified_bytes',0),'target_bytes':spec['bytes'],'rss_bytes':rss,'rss_limit_gib':16,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()}
            atomic_json(BASE/'release_state.json',observation)
            if rss>16*2**30:
                atomic_json(BASE/'error.json',{'status':'error','reason':'rss_guard_16gib'});os._exit(75)
            stop.wait(30)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        state=upload(BASE/'upload_spec.json');assert state['status']=='all_source_payloads_verified';phase['value']='validating_and_finalizing'
        result=finish(spec,state);stop.set();thread.join();atomic_json(BASE/'release_state.json',{'status':'complete','done':spec['files'],'total':spec['files']})
        print(json.dumps({k:v for k,v in result.items() if k!='consumer_validation'}),flush=True)
    except Exception as e:
        stop.set();thread.join();error={'status':'error','error_type':type(e).__name__,'quota_error':is_quota_error(e),'http_status':getattr(getattr(e,'response',None),'status_code',None)}
        atomic_json(BASE/'error.json',error);atomic_json(BASE/'release_state.json',error);print(json.dumps(error),flush=True);raise SystemExit(1)


if __name__=='__main__':main()
