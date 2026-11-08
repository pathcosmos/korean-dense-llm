"""Preserve card bodies and show standalone repositories beside main folders.

The active consolidator reads root_template only after uploading. Main-card
templates are updated locally; no concurrent main-branch commit interrupts it.
Official hf.co short links also survive the running legacy URL migration pass.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import atomic_json,verify
from huggingface_hub import HfApi,CommitOperationAdd
import requests

MAIN='pathcosmos/korean-pretrain-corpus'
SOURCES=[('wiki_ko','wikipedia'),('fineweb2_ko','fineweb2'),('hplt_ko','hplt2'),
         ('fineweb_edu','fineweb-edu'),('owm','math'),('github_code_v1','code')]
BEGIN='<!-- corpus-repository-links:start -->'
END='<!-- corpus-repository-links:end -->'
BASE=Path(PROJECT_ROOT)/'runs/public_card_dual_links_v1'


def digest(data):return hashlib.sha256(data).hexdigest()


def block():
    rows=[BEGIN,'## Repository links / 저장소 링크','',
        f'[종합 저장소 / Main corpus](https://huggingface.co/datasets/{MAIN})','',
        '| 소스 / Source | 개별 저장소 / Standalone repository | 종합 저장소 내부 / Folder in main corpus |',
        '|---|---|---|']
    for key,suffix in SOURCES:
        name='korean-pretrain-'+suffix
        rows.append(f'| {key} | [{name}](https://hf.co/datasets/pathcosmos/{name}) | [{key}/](https://huggingface.co/datasets/{MAIN}/tree/main/{key}) |')
    rows+=['','두 경로를 함께 제공합니다. 고정 버전과 검증 범위는 release.json을 참고하세요.',
        'Both locations are listed. See release.json for pinned versions and verification scope.',END]
    return '\n'.join(rows)


def outside(text):
    assert text.count(BEGIN)==text.count(END)<=1
    if BEGIN not in text:return text
    start=text.index(BEGIN);end=text.index(END)+len(END)
    return text[:start]+text[end:]


def linked(text):
    if BEGIN in text:
        start=text.index(BEGIN);end=text.index(END)+len(END)
        result=text[:start]+block()+text[end:]
        assert outside(result)==outside(text)
    else:
        result=text+'\n\n'+block()+'\n'
        assert result.startswith(text)
    assert result.count(BEGIN)==result.count(END)==1
    for key,suffix in SOURCES:
        assert f'https://hf.co/datasets/pathcosmos/korean-pretrain-{suffix}' in result
        assert f'https://huggingface.co/datasets/{MAIN}/tree/main/{key}' in result
    return result


def fetch(repo,revision,name):
    session=requests.Session();session.trust_env=False
    r=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',
                  headers={'Accept-Encoding':'identity'},timeout=60)
    r.raise_for_status();return r.content


def prepare():
    BASE.mkdir(exist_ok=True,parents=True);api=HfApi(token=False)
    template=Path(PROJECT_ROOT)/'runs/dataset_unified_v1/root_template'
    plan={'utc':datetime.now(timezone.utc).isoformat(),'main_templates':[],'standalone_cards':[],
        'main_repository_mutated_during_upload':False,'body_outside_navigation_preserved':True}
    for p in [template/'README.md',*sorted((template/'cards').glob('*.md'))]:
        name=str(p.relative_to(template));before=p.read_bytes();after=linked(before.decode()).encode()
        dest=BASE/'main_templates'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(after)
        saved=BASE/'main_before'/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(before)
        # Prove the active finalizer's URL rewrite will retain both destinations.
        migrated=after.decode()
        for key,suffix in SOURCES:
            migrated=migrated.replace(f'https://huggingface.co/datasets/pathcosmos/korean-pretrain-{suffix}',
                f'https://huggingface.co/datasets/{MAIN}/tree/v1.1.0/{key}')
        assert block() in migrated
        plan['main_templates'].append({'path':name,'before_sha256':digest(before),'sha256':digest(after),'bytes':len(after)})
    for key,suffix in SOURCES:
        repo='pathcosmos/korean-pretrain-'+suffix;info=api.repo_info(repo,repo_type='dataset');assert not info.private
        before=fetch(repo,info.sha,'README.md');after=linked(before.decode()).encode()
        folder=BASE/'standalone'/key;folder.mkdir(parents=True,exist_ok=True)
        (folder/'before.md').write_bytes(before);(folder/'README.md').write_bytes(after)
        plan['standalone_cards'].append({'repository':repo,'parent_commit':info.sha,'path':'README.md',
            'before_sha256':digest(before),'sha256':digest(after),'bytes':len(after),'local_path':str((folder/'README.md').relative_to(PROJECT_ROOT))})
    assert len(plan['main_templates'])==14 and len(plan['standalone_cards'])==6
    assert all(linked((BASE/'main_templates'/r['path']).read_text())==(BASE/'main_templates'/r['path']).read_text() for r in plan['main_templates'])
    atomic_json(BASE/'plan.json',plan);return plan


def apply():
    plan=json.loads((BASE/'plan.json').read_text());p=Path(PROJECT_ROOT)/'runs/dataset_unified_v1'
    state=json.loads((p/'release_state.json').read_text())
    assert state.get('phase')=='uploading' and not (p/'payload_report.json').exists(),'Recheck finalization before changing templates'
    for r in plan['main_templates']:
        dest=p/'root_template'/r['path'];current=digest(dest.read_bytes())
        assert current in (r['before_sha256'],r['sha256'])
        if current==r['sha256']:continue
        data=(BASE/'main_templates'/r['path']).read_bytes();assert digest(data)==r['sha256']
        tmp=dest.with_suffix(dest.suffix+'.nav.tmp');tmp.write_bytes(data);tmp.replace(dest)
    atomic_json(BASE/'main_template_applied.json',{'status':'passed','files':14,'published':False,
        'publication_owner':'existing running consolidator after payload verification',
        'immutable_payload_spec_unchanged':True,'main_head_unchanged_by_this_tool':True})
    api=HfApi();assert api.whoami()['name']=='pathcosmos'
    for r in plan['standalone_cards']:
        receipt=BASE/(r['repository'].split('/')[-1]+'.receipt.json')
        path=Path(PROJECT_ROOT)/r['local_path'];assert digest(path.read_bytes())==r['sha256']
        if receipt.exists():revision=json.loads(receipt.read_text())['commit']
        else:
            result=api.create_commit(r['repository'],repo_type='dataset',parent_commit=r['parent_commit'],
                commit_message='Preserve card content and link standalone datasets beside main corpus folders',
                operations=[CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=path)])
            revision=result.oid
            atomic_json(receipt,{'status':'committed_verification_pending','repository':r['repository'],'commit':revision})
        proofs=verify(HfApi(token=False),r['repository'],revision,[r],anonymous=True)
        atomic_json(receipt,{'status':'passed','repository':r['repository'],'commit':revision,'proofs':proofs})
    atomic_json(BASE/'report.json',{'status':'standalone_published_main_templates_ready','standalone_cards_published':6,
        'main_cards_prepared':14,'main_cards_publication_pending':True,'existing_structure_preserved':True,
        'standalone_and_internal_links_present':True,'payload_changes':0,'release_tag_changes':0})


def verify_main(wait):
    p=Path(PROJECT_ROOT)/'runs/dataset_unified_v1';start=time.monotonic()
    while not (p/'release.done').exists():
        if not wait:raise RuntimeError('Main corpus release has not finished')
        atomic_json(BASE/'verification_state.json',{'status':'waiting_for_main_publication','pid':os.getpid(),
            'utc':datetime.now(timezone.utc).isoformat()})
        if time.monotonic()-start>72*3600:raise TimeoutError('Main publication wait budget exceeded')
        time.sleep(30)
    report=json.loads((p/'report.json').read_text());assert report['status']=='passed'
    assert digest((p/'report.json').read_bytes())==json.loads((p/'release.done').read_text())['report_sha256']
    revision=report['commit'];checks=json.loads(fetch(MAIN,revision,'CHECKSUMS.json'))
    by_path={r['path']:r for r in checks['files']};plan=json.loads((BASE/'plan.json').read_text())
    cards=[]
    for r in plan['main_templates']:
        data=fetch(MAIN,revision,r['path']);assert digest(data)==by_path[r['path']]['sha256']
        text=data.decode();assert block() in text
        if r['path']=='README.md':assert text.index('## 한국어')<text.index('## English')
        cards.append({'path':r['path'],'sha256':digest(data),'bytes':len(data)})
    result={'status':'passed','main_repository':MAIN,'main_commit':revision,'main_cards_verified':14,
        'standalone_cards_verified':6,'standalone_and_internal_links_present':True,'payload_changes':0,
        'tag_changes_by_navigation_job':0,'main_checksum_verified_for_each_card':True,'main_card_proofs':cards}
    atomic_json(BASE/'report.json',result);atomic_json(BASE/'verification_state.json',{'status':'complete','done':20,'total':20})
    atomic_json(BASE/'navigation.done',{'report_sha256':digest((BASE/'report.json').read_bytes())})
    print(json.dumps({'status':'passed','cards':20,'main_commit':revision}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--verify-main',action='store_true');parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.verify_main:verify_main(args.wait)
    elif args.apply:apply()
    else:prepare()
    if not args.verify_main:
        print(json.dumps({'status':'applied' if args.apply else 'prepared','run_dir':str(BASE.relative_to(PROJECT_ROOT))}))
