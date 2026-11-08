"""Wait for source transfer, validate public consumption, then publish pinned release.

Runs locally without model calls or outbound messages. Quota/transfer failures
stop this stage; no paid service, deletion, or history rewrite is attempted.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import atomic_json,verify,is_quota_error
from huggingface_hub import HfApi,CommitOperationAdd

FINAL_STATUS={
'ko':'6개 소스의 기존 정제 스냅샷 텍스트, train/eval 토큰, 분리 정보와 문서별 토큰 위치를 공개했습니다. 토크나이저·재현 코드도 제공합니다. 고정 파일과 commit은 release.json을 따릅니다. 문서별 URL·원 Git commit·고지의 미확인 범위는 소스별 source.json에 명시합니다.',
'en':'The complete existing six-source snapshot is public: cleaned text, train/eval tokens, canonical assignments and document offsets, plus tokenizer and reproduction code. Use release.json for pinned commits and files. Remaining document-URL, original-Git-commit and notice gaps are explicitly documented in each source.json.',
'zh-CN':'六个来源的现有完整快照已公开：清洗文本、train/eval 词元、规范划分和文档偏移，以及分词器与复现代码。固定文件和提交见 release.json。文档 URL、原 Git 提交及声明的未确认范围列于各 source.json。',
'zh-TW':'六個來源的既有完整快照已公開：清理文本、train/eval 詞元、規範切分與文件偏移，以及分詞器和重現程式碼。固定檔案與提交請見 release.json。文件 URL、原 Git 提交及聲明的未確認範圍列於各 source.json。',
'ja':'6ソースの既存スナップショット全体を公開しました。整形済みテキスト、train/evalトークン、正規の分割情報、文書位置、トークナイザーと再現コードを含みます。固定commitはrelease.jsonを参照してください。文書URL・元Git commit・告知の未確認範囲は各source.jsonに明記しています。',
'fr':'Le snapshot existant complet des six sources est public : textes nettoyés, tokens train/eval, partitions canoniques, positions des documents, tokeniseur et code de reproduction. Les commits sont fixés dans release.json. Les lacunes restantes concernant URL, commits Git originaux et notices sont précisées dans chaque source.json.',
'de':'Der vollständige vorhandene Snapshot aller sechs Quellen ist öffentlich: bereinigte Texte, train/eval-Tokens, kanonische Zuordnungen, Dokumentpositionen, Tokenizer und Reproduktionscode. Feste Commits stehen in release.json. Offene Fragen zu Dokument-URLs, ursprünglichen Git-Commits und Hinweisen sind in source.json dokumentiert.',
'es':'El snapshot existente completo de las seis fuentes es público: textos limpios, tokens train/eval, asignaciones canónicas, posiciones de documentos, tokenizador y código. Los commits se fijan en release.json. Las lagunas de URL, commits Git originales y avisos se documentan en cada source.json.',
'pt-BR':'O snapshot existente completo das seis fontes está público: textos limpos, tokens train/eval, divisões canônicas, posições de documentos, tokenizador e código. Os commits estão fixados em release.json. Lacunas de URL, commits Git originais e avisos são documentadas em cada source.json.',
'hi':'छह स्रोतों का मौजूदा पूरा स्नैपशॉट सार्वजनिक है: साफ पाठ, train/eval टोकन, canonical विभाजन, दस्तावेज़ स्थितियाँ, टोकनाइज़र और पुनरुत्पादन कोड। निश्चित commit release.json में हैं। दस्तावेज़ URL, मूल Git commit और नोटिस की शेष कमियाँ हर source.json में दर्ज हैं।',
'ar':'أصبحت اللقطة الحالية الكاملة للمصادر الستة عامة: النصوص المنقّحة ورموز train/eval والتقسيمات المرجعية ومواضع الوثائق والمجزّئ وشيفرة إعادة الإنتاج. المراجعات المثبتة في release.json، والفجوات المتبقية في روابط الوثائق وGit والإشعارات موثقة في كل source.json.',
'he':'תמונת המצב הקיימת המלאה של ששת המקורות ציבורית: טקסט מנוקה, טוקני train/eval, חלוקות קנוניות, מיקומי מסמכים, טוקנייזר וקוד שחזור. גרסאות קבועות נמצאות ב-release.json, ופערים בכתובות מסמכים, Git והודעות מתועדים בכל source.json.',
'ru':'Полный существующий снимок шести источников опубликован: очищенный текст, токены train/eval, канонические разбиения, позиции документов, токенизатор и код. Коммиты закреплены в release.json; оставшиеся пробелы URL, исходных Git-коммитов и уведомлений указаны в source.json.'}
COMPAT={
'ko':'AutoTokenizer에서도 원본과 동일한 어휘 및 16개 특수토큰 ID를 검증했습니다. 대규모 3B 학습의 독립 환경 전체 재현은 별도 검증 대상입니다.',
'en':'AutoTokenizer preserves the original vocabulary and all 16 special-token IDs. Full independent 3B training reproduction remains a separate validation task.',
'zh-CN':'AutoTokenizer 已验证保留原词表及全部 16 个特殊词元 ID。独立环境中的完整 3B 训练复现仍需另行验证。',
'zh-TW':'AutoTokenizer 已驗證保留原詞彙表及全部 16 個特殊詞元 ID。獨立環境中的完整 3B 訓練重現仍須另行驗證。',
'ja':'AutoTokenizerでも元の語彙と16個の特殊トークンIDを維持することを検証しました。独立環境での3B学習全体の再現は別途検証が必要です。',
'fr':'AutoTokenizer conserve le vocabulaire et les 16 identifiants spéciaux originaux. La reproduction indépendante complète de l’entraînement 3B reste à valider.',
'de':'AutoTokenizer erhält das ursprüngliche Vokabular und alle 16 Spezialtoken-IDs. Eine unabhängige vollständige Reproduktion des 3B-Trainings ist gesondert zu prüfen.',
'es':'AutoTokenizer conserva el vocabulario y los 16 ID especiales originales. La reproducción independiente completa del entrenamiento 3B sigue pendiente.',
'pt-BR':'AutoTokenizer preserva o vocabulário e os 16 IDs especiais originais. A reprodução independente completa do treinamento 3B ainda exige validação.',
'hi':'AutoTokenizer में मूल शब्दावली और सभी 16 विशेष टोकन ID समान पाए गए। स्वतंत्र परिवेश में पूरे 3B प्रशिक्षण का पुनरुत्पादन अलग सत्यापन है।',
'ar':'يحافظ AutoTokenizer على المفردات ومعرّفات الرموز الخاصة الـ16 الأصلية. تبقى إعادة تدريب 3B كاملة في بيئة مستقلة بحاجة إلى تحقق منفصل.',
'he':'AutoTokenizer שומר על אוצר המילים ועל כל 16 מזהי הטוקנים המיוחדים המקוריים. שחזור עצמאי מלא של אימון 3B דורש אימות נפרד.',
'ru':'AutoTokenizer сохраняет исходный словарь и все 16 ID специальных токенов. Полное независимое воспроизведение обучения 3B требует отдельной проверки.'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(api,repo,head,changes):
    operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=p) for n,p in sorted(changes.items())]
    commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,
        commit_message='Finalize pinned public snapshot and verified consumer instructions',operations=operations)
    records=[{'path':n,'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in changes.items()]
    verify(HfApi(token=False),repo,commit.oid,records,anonymous=True)
    return commit.oid


def tagged(api,repo,revision):
    api.create_tag(repo,tag='v1.0.0',revision=revision,repo_type='dataset',exist_ok=True)
    assert api.repo_info(repo,repo_type='dataset',revision='v1.0.0').sha==revision


def finalize(root):
    base=root/'runs/dataset_upload_v1';spec=json.loads((base/'upload_spec.json').read_text())
    marker=json.loads((base/'supervisor.done').read_text());assert sha(base/'supervisor_report.json')==marker['report_sha256']
    report=json.loads((base/'supervisor_report.json').read_text());assert report['status']=='passed'
    upload=report['upload_state'];assert upload['status']=='all_source_payloads_verified'
    assert upload['verified_files']==spec['files'] and upload['verified_bytes']==spec['bytes']
    assets=json.loads((base/'index_assets_uploaded.json').read_text())
    api=HfApi();assert api.whoami()['name']=='pathcosmos'
    work=base/'final_metadata';work.mkdir(exist_ok=True)
    progress_path=work/'source_commits.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for source in spec['repositories']:
        repo=source['repository'];key=source['source'];parent=upload['repositories'][repo]['head']
        if repo in progress:
            tagged(api,repo,progress[repo]['commit']);continue
        folder=work/key;folder.mkdir(exist_ok=True)
        original=root/source['metadata_folder'];text=(original/'README.md').read_text()
        text=text.replace('**전송 진행 중입니다. manifest의 전체 파일과 검증된 release commit이 갖춰질 때까지 전체 공개 완료 상태가 아닙니다.**',
            '**기존 정제 스냅샷의 전체 텍스트·토큰·분리 정보·문서 위치를 전송하고 검증했습니다. 출처·고지의 미확인 범위는 source.json에 명시합니다.**')
        text=text.replace('**Upload in progress. This is not a complete release until every manifest entry is present and a verified release commit is published.**',
            '**The complete existing source snapshot is public and verified. Remaining attribution/notice gaps are documented in source.json.**')
        text=text.replace('revision = HfApi(token=False).repo_info(repo, repo_type="dataset").sha','revision = "v1.0.0"')
        text=text.replace('Pin the verified release commit when available.','Use the pinned v1.0.0 tag or the commit in the main release index.')
        (folder/'README.md').write_text(text)
        manifest=json.loads((original/'manifest.json').read_text());manifest['status']='public_snapshot_with_documented_provenance_gaps'
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        release={'version':'v1.0.0','source':key,'data_commit':parent,'manifest_sha256':sha(folder/'manifest.json'),
            'counts':manifest['counts'],'provenance_status':manifest['provenance_status'],'payloads_verified':True,
            'original_input_bytes_preserved':True,'content_rights_blanket_clearance_claimed':False}
        (folder/'release.json').write_text(json.dumps(release,indent=2)+'\n')
        current=api.repo_info(repo,repo_type='dataset').sha
        changes={n:folder/n for n in ['README.md','manifest.json','release.json']}
        if current!=parent:
            verify(HfApi(token=False),repo,current,[{'path':n,'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in changes.items()]);revision=current
        else:revision=publish(api,repo,parent,changes)
        tagged(api,repo,revision)
        progress[repo]={'source':key,'commit':revision,'tag':'v1.0.0','manifest_sha256':sha(folder/'manifest.json'),
                        'documents':manifest['counts']['train_docs']+manifest['counts']['holdout_docs'],
                        'provenance_status':manifest['provenance_status']}
        atomic_json(progress_path,progress)
    commits={repo:r['commit'] for repo,r in progress.items()};atomic_json(work/'consumer_commits.json',commits)
    consumer=work/'consumer_validation.json'
    if not consumer.exists():
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        subprocess.run([sys.executable,'scripts/check_public_corpus_consumer.py','--commits',str(work/'consumer_commits.json'),'--output',str(consumer)],
                       cwd=root,env=env,check=True,timeout=600)
    assert json.loads(consumer.read_text())['status']=='passed'
    original=root/'runs/dataset_public_release_v1/public_index_v4';index=work/'index'
    if not index.exists():
        shutil.copytree(original,index)
        for source in [base/'index_assets',base/'index_metadata']:
            for p in source.rglob('*'):
                if p.is_file():dest=index/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    module_path=root/'runs/dataset_public_release_v1/build_multilingual_cards.py'
    module_spec=importlib.util.spec_from_file_location('card_text',module_path);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    for p in [index/'README.md',*(index/'cards').glob('*.md')]:
        text=p.read_text()
        for lang,content in module.LANGS.items():
            text=text.replace(content['status'],FINAL_STATUS[lang])
            old=content['tokenizer'];position=old.rfind('AutoTokenizer')
            cut=max(old.rfind('.',0,position),old.rfind('。',0,position),old.rfind('।',0,position))+1
            text=text.replace(old,old[:cut].strip()+' '+COMPAT[lang])
        if p.name=='README.md' and p.parent==index:
            replacements={
                '| Cleaned text: six sources | No | Local corpus exists; source attribution and public export ongoing |':'| Cleaned text: six sources | Yes | Original cleaned bytes, with explicit canonical split sidecars and documented attribution gaps |',
                '| Train/eval token binaries | No | All 1,452 files rehashed; counts and document boundaries checked |':'| Train/eval token binaries | Yes | All 1,452 files rehashed; counts and document boundaries checked |',
                '| Portable splits and offsets | No | All 128,856,258 assignments and offsets verified locally |':'| Portable splits and offsets | Yes | All 128,856,258 assignments and offsets verified |',
                '| Per-document provenance | No | All 455,397 Wikipedia documents mapped; remaining source work pending |':'| Per-document provenance | Partial | Wikipedia URL candidates supplied; remaining source gaps in source.json |',
                'The code/source/documentation preview is available; complete corpus payloads are not yet hosted.':'The complete existing six-source snapshot is public; attribution scope is documented per source.',
                'These figures describe local artifacts, not bytes already hosted here.':'These figures describe the published existing snapshot.',
                'a final pinned release index will follow.':'the pinned release index is [release.json](release.json).',
                'Source payloads are being transferred in verified batches;':'Source payloads were transferred and verified in batches;',
                'The preview is not\na one-command training package; it does not yet include the full text/token payloads.':'The code archive is not a one-command full-scale training package. Source repositories contain the text/token payloads.',
                'After publication, reuse will support starting from text with a new tokenizer':'Reuse supports starting from text with a new tokenizer',
                'note = {Publication in preparation; cite the exact revision used}':'note = {Snapshot v1.0.0; cite the exact revision used}'}
            for a,b in replacements.items():text=text.replace(a,b)
            text+='\n## Pinned public release / 고정 공개 버전\n\n[release.json](release.json) lists the six verified source commits. [CONSUMER_VALIDATION.json](CONSUMER_VALIDATION.json) records anonymous canonical-split samples and a bounded CPU toy-model save/resume check. It does not certify full 3B training reproduction.\n'
        p.write_text(text)
    release={'version':'v1.0.0','status':'existing_six_source_snapshot_public_with_documented_provenance_gaps',
        'publisher':'pathcosmos','contact':'miracle.yang@pathcosmos.com','repositories':progress,
        'baseline_verified_source_files':spec['files'],'baseline_verified_source_bytes':spec['bytes'],
        'documents':128856258,'tokens':122402393813,
        'tokenizer_sha256':'050f9b43cf964f0c3b29ae443a1c2eae46cb2ee0022cc010bcfea3d26acf1706',
        'consumer_validation_sha256':sha(consumer),'full_clean_room_3B_training_reproduction':False}
    atomic_json(index/'release.json',release);shutil.copyfile(consumer,index/'CONSUMER_VALIDATION.json')
    state=json.loads((index/'PUBLICATION_STATUS.json').read_text());state.update(status=release['status'],
        corpus_text_public=True,token_bins_public=True,tokenizer_public=True,payload_transfer_status='complete',
        portable_splits_public=True,document_offsets_public=True,source_repositories_created=True,pinned_version='v1.0.0')
    atomic_json(index/'PUBLICATION_STATUS.json',state)
    checks=[{'path':str(p.relative_to(index)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(index.rglob('*')) if p.is_file() and p.name!='CHECKSUMS.json']
    atomic_json(index/'CHECKSUMS.json',{'files':checks})
    names=['README.md','PUBLICATION_STATUS.json','CHECKSUMS.json','release.json','CONSUMER_VALIDATION.json']+[str(p.relative_to(index)) for p in sorted((index/'cards').glob('*.md'))]
    repo='pathcosmos/korean-pretrain-corpus';current=api.repo_info(repo,repo_type='dataset').sha
    index_commit_path=work/'index_commit.json'
    if index_commit_path.exists():revision=json.loads(index_commit_path.read_text())['commit']
    else:
        assert current==assets['commit'],'Index changed outside the prepared release'
        revision=publish(api,repo,current,{n:index/n for n in names});atomic_json(index_commit_path,{'commit':revision})
    tagged(api,repo,revision)
    return {'status':'passed','repository':repo,'commit':revision,'version':'v1.0.0','sources':progress,
        'baseline_source_files':spec['files'],'baseline_source_bytes':spec['bytes'],'consumer_validation':json.loads(consumer.read_text()),
        'scope':'all_existing_snapshot_payloads_public_known_attribution_gaps_documented','paid_services_used':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    root=Path(PROJECT_ROOT);base=root/'runs/dataset_upload_v1';lock=(base/'finalizer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (base/'finalizer.pid').write_text(str(os.getpid())+'\n');start=time.monotonic()
    try:
        while not (base/'supervisor.done').exists():
            if not a.wait:raise RuntimeError('Source transfer is not finished')
            atomic_json(base/'finalizer_state.json',{'status':'waiting_for_source_transfer','pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()})
            if time.monotonic()-start>72*3600:raise TimeoutError('Finalization wait budget exceeded')
            time.sleep(30)
        atomic_json(base/'finalizer_state.json',{'status':'finalizing','pid':os.getpid()})
        result=finalize(root);atomic_json(base/'finalizer_report.json',result)
        atomic_json(base/'finalizer.done',{'report_sha256':sha(base/'finalizer_report.json')})
        atomic_json(base/'finalizer_state.json',{'status':'complete','done':1,'total':1})
        with (root/'logs/pipeline_events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'stage':'corpus_public_release','state':'complete','repository':result['repository'],'commit':result['commit']})+'\n')
        print(json.dumps({'status':'passed','commit':result['commit']}),flush=True)
    except Exception as error:
        result={'status':'failed','error_type':type(error).__name__,'http_status':getattr(getattr(error,'response',None),'status_code',None),'quota_error':is_quota_error(error)}
        atomic_json(base/'finalizer_report.json',result);atomic_json(base/'finalizer_state.json',{'status':'error',**result})
        print(json.dumps(result),flush=True);sys.exit(1)
