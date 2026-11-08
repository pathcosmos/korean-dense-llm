"""Read-only anonymous verification of the actual final public release."""
import hashlib,json,sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import requests,yaml
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.upload_public_corpus import verify,atomic_json

base=Path(__file__).parent
report=json.loads((base/'finalizer_report.json').read_text())
assert report['status']=='passed'
marker=json.loads((base/'finalizer.done').read_text())
assert hashlib.sha256((base/'finalizer_report.json').read_bytes()).hexdigest()==marker['report_sha256']
api=HfApi(token=False)
repo=report['repository'];revision=report['commit']

def fetch(repo,revision,name):
    session=requests.Session();session.trust_env=False
    r=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',
                  headers={'Accept-Encoding':'identity'},timeout=60)
    r.raise_for_status();return r.content

release=json.loads(fetch(repo,revision,'release.json'))
checks=json.loads(fetch(repo,revision,'CHECKSUMS.json'))['files']
assert not any('__pycache__' in r['path'] or r['path'].endswith('.pyc') for r in checks)
verify(api,repo,revision,checks,anonymous=True)
listed=set(api.list_repo_files(repo,repo_type='dataset',revision=revision))
assert listed-{'.gitattributes','CHECKSUMS.json'}=={r['path'] for r in checks}
assert api.repo_info(repo,repo_type='dataset',revision='v1.0.0').sha==revision
root_card=fetch(repo,revision,'README.md').decode()
assert root_card.index('## 한국어')<root_card.index('## English')
assert len([n for n in listed if n.startswith('cards/README.') and n.endswith('.md')])==13
metadata=yaml.safe_load(root_card.split('---',2)[1])
assert {'llm','pretraining','language-modeling','korean-pretrain-corpus'}<=set(metadata['tags'])
assert metadata['language']==['ko','en'] and metadata['license']=='other'
status=json.loads(fetch(repo,revision,'PUBLICATION_STATUS.json'))
assert status['payload_transfer_status']=='complete'
validation=json.loads(fetch(repo,revision,'CONSUMER_VALIDATION.json'))
assert validation['status']=='passed' and len(validation['anonymous_source_checks'])==6
assert validation['bounded_cpu_training']['save_resume_second_update_exact'] is True

def source_check(item):
    repo,entry=item;api=HfApi(token=False)
    info=api.repo_info(repo,repo_type='dataset',revision='v1.0.0')
    assert info.sha==entry['commit'] and info.private is False
    card=fetch(repo,entry['commit'],'README.md').decode()
    data=yaml.safe_load(card.split('---',2)[1])
    assert 'korean-pretrain-corpus' in data['tags'] and data['license']=='other'
    assert data['configs'][0]['data_files'][0]['split']=='all'
    manifest=fetch(repo,entry['commit'],'manifest.json')
    assert hashlib.sha256(manifest).hexdigest()==entry['manifest_sha256']
    return {'repository':repo,'commit':info.sha,'tag':'v1.0.0','public':True,'tags':data['tags']}

with ThreadPoolExecutor(max_workers=4) as pool:sources=list(pool.map(source_check,release['repositories'].items()))
result={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'repository':repo,
        'commit':revision,'version':'v1.0.0','baseline_source_files':release['baseline_verified_source_files'],
        'baseline_source_bytes':release['baseline_verified_source_bytes'],'root_checksum_entries':len(checks),
        'translation_documents':13,'root_tags':metadata['tags'],'sources':sources,
        'consumer_validation':validation,'all_root_files_accounted_for':True}
atomic_json(base/'final_public_verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['sources','consumer_validation']},ensure_ascii=False))
