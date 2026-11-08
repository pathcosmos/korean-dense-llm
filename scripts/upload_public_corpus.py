"""Publish an allowlisted corpus with bounded retries, immutable receipts, and quota stop.

No billing changes, deletion, visibility changes, or training changes. Public
SDK calls only. Paths are frozen by bytes/SHA/mtime/inode; remote LFS OIDs are
checked instead of downloading the entire corpus a second time.
"""
import os
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER','0')
import argparse,fcntl,hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from huggingface_hub import HfApi,CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError
import requests


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def batches(records,max_files=24,max_bytes=4*2**30):
    batch=[];size=0
    for r in records:
        if batch and (len(batch)>=max_files or size+r['bytes']>max_bytes):
            yield batch;batch=[];size=0
        batch.append(r);size+=r['bytes']
    if batch:yield batch


def is_quota_error(error):
    message=str(error).lower()
    return any(x in message for x in ['storage limit','storage quota','exceeded your storage',
                                      'exceeded the storage','public storage','storage capacity'])


def make_operation(record):
    p=Path(PROJECT_ROOT)/record['local_path'];st=p.stat()
    if (st.st_size,st.st_mtime_ns,st.st_ino)!=(record['bytes'],record['mtime_ns'],record['inode']):
        raise ValueError('Frozen upload input changed')
    op=CommitOperationAdd(path_in_repo=record['path'],path_or_fileobj=p)
    if op.upload_info.sha256.hex()!=record['sha256'] or op.upload_info.size!=record['bytes']:
        raise ValueError('Outgoing bytes do not match certified SHA256')
    return op


def verify(api,repo,revision,records,*,anonymous=False):
    info=api.get_paths_info(repo,[r['path'] for r in records],repo_type='dataset',revision=revision)
    by_path={i.path:i for i in info}
    if set(by_path)!={r['path'] for r in records}:raise ValueError('Remote file list incomplete')
    proofs=[]
    for r in records:
        item=by_path[r['path']]
        if item.size!=r['bytes']:raise ValueError('Remote size differs')
        if item.lfs:
            if item.lfs.sha256!=r['sha256'] or item.lfs.size!=r['bytes']:raise ValueError('Remote LFS SHA256 differs')
            proof='remote_lfs_file_sha256_and_size'
        else:
            # Small Git files: confirm actual public body SHA256 as well.
            session=requests.Session();session.trust_env=False
            response=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{r["path"]}',timeout=60)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=r['sha256']:raise ValueError('Remote Git-file body SHA256 differs')
            proof='anonymous_file_body_sha256'
        proofs.append({'path':r['path'],'bytes':r['bytes'],'sha256':r['sha256'],'proof':proof})
    return proofs


def upload(spec_path,*,max_batches=None):
    root=Path(PROJECT_ROOT);spec_path=Path(spec_path);base=spec_path.parent
    spec_bytes=spec_path.read_bytes();spec=json.loads(spec_bytes);spec_hash=hashlib.sha256(spec_bytes).hexdigest()
    lock=(base/'uploader.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-8:])
    assert spec['account']=='pathcosmos' and spec['private'] is False and spec['free_only'] is True
    api=HfApi();assert api.whoami()['name']==spec['account']
    state_path=base/'upload_state.json';journal=base/'upload_receipts.jsonl'
    if state_path.exists():
        state=json.loads(state_path.read_text());assert state['upload_spec_sha256']==spec_hash,'Upload contract changed'
        if state['status']=='stopped_quota':raise RuntimeError('Quota-stop latched; user decision required before retry')
    else:
        state={'upload_spec_sha256':spec_hash,'status':'starting','verified_files':0,'verified_bytes':0,'repositories':{},
               'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()}
    state['pid']=os.getpid();completed_batches=0
    receipt_dir=base/'receipts';receipt_dir.mkdir(exist_ok=True)
    receipts=[json.loads(p.read_text()) for p in sorted(receipt_dir.glob('*.json'))]
    # Atomic per-batch receipts are the resume authority, including a crash
    # after the remote commit but before the progress-state update.
    for receipt in receipts:
        repo=receipt['repository']
        state['repositories'].setdefault(repo,{})['head']=receipt['revision']
    state['verified_files']=sum(len(r['proofs']) for r in receipts)
    state['verified_bytes']=sum(p['bytes'] for r in receipts for p in r['proofs'])
    def save(**updates):
        state.update(updates);state['utc']=datetime.now(timezone.utc).isoformat();atomic_json(state_path,state)
    def send(repo,records):
        nonlocal completed_batches
        save(status='hashing',active_repository=repo,active_files=len(records),active_bytes=sum(r['bytes'] for r in records))
        with ThreadPoolExecutor(max_workers=4) as pool:ops=list(pool.map(make_operation,records))
        last_error=None;commit=None
        for attempt in range(3):
            try:
                head=api.repo_info(repo,repo_type='dataset').sha
                known=state['repositories'][repo].get('head')
                if known and head!=known:raise ValueError('Remote head changed outside this uploader')
                save(status='uploading',active_attempt=attempt+1)
                # Resumable server content-addressing: already-uploaded blobs are reused.
                api.preupload_lfs_files(repo,ops,repo_type='dataset',num_threads=4,free_memory=False)
                commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,num_threads=4,
                    commit_message=f'Upload verified corpus snapshot batch ({len(records)} files)',operations=ops)
                revision=commit.oid
                proofs=verify(api,repo,revision,records)
                break
            except Exception as e:
                if is_quota_error(e):
                    save(status='stopped_quota',error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),
                         error_reason='storage_quota_or_capacity_rejected_no_paid_transition')
                    raise
                code=getattr(getattr(e,'response',None),'status_code',None)
                if code in (400,401,403,409,413,422) or isinstance(e,(ValueError,AssertionError)):
                    save(status='stopped_error',error_type=type(e).__name__,http_status=code,error_reason='permanent_or_integrity_error')
                    raise
                # Commit may have succeeded despite a lost response: inspect first.
                try:
                    current=api.repo_info(repo,repo_type='dataset').sha
                    proofs=verify(api,repo,current,records);revision=current;break
                except Exception:pass
                last_error=e
                if attempt==2:raise
                time.sleep(min(30,5*(attempt+1)))
                with ThreadPoolExecutor(max_workers=4) as pool:ops=list(pool.map(make_operation,records))
        else:raise last_error
        for r in records:
            st=(root/r['local_path']).stat()
            if (st.st_size,st.st_mtime_ns,st.st_ino)!=(r['bytes'],r['mtime_ns'],r['inode']):raise ValueError('Input changed during transfer')
        receipt={'utc':datetime.now(timezone.utc).isoformat(),'repository':repo,'revision':revision,'proofs':proofs}
        atomic_json(receipt_dir/f'{len(receipts):06d}.json',receipt)
        with journal.open('a') as f:f.write(json.dumps(receipt)+'\n');f.flush();os.fsync(f.fileno())
        receipts.append(receipt);completed_batches+=1
        state['repositories'][repo]['head']=revision
        state['verified_files']=sum(len(x['proofs']) for x in receipts)
        state['verified_bytes']=sum(p['bytes'] for x in receipts for p in x['proofs'])
        save(status='batch_verified')
        print(json.dumps({'state':'batch_verified','repo':repo,'verified_files':state['verified_files'],
                          'verified_bytes':state['verified_bytes'],'commit':revision}),flush=True)
    # Prepare all six source repositories and source/rights notices first.
    for source in spec['repositories']:
        repo=source['repository'];api.create_repo(repo,repo_type='dataset',private=False,exist_ok=True)
        info=api.repo_info(repo,repo_type='dataset');assert info.private is False,'Unexpected private destination'
        if repo not in state['repositories']:
            remote=api.list_repo_files(repo,repo_type='dataset')
            if set(remote)-{'.gitattributes'}:raise ValueError('Unrecognized nonempty destination')
            state['repositories'][repo]={'head':info.sha,'source':source['source']}
        save(status='preparing_repositories')
        done={p['path'] for r in receipts if r['repository']==repo for p in r['proofs']}
        pending=[r for r in source['files'] if r['kind']=='metadata' and r['path'] not in done]
        if pending:send(repo,pending)
    for source in spec['repositories']:
        repo=source['repository'];done={p['path'] for r in receipts if r['repository']==repo for p in r['proofs']}
        existing=[r for r in receipts if r['repository']==repo]
        if existing:
            # On resume, confirm preserved commits still describe the recorded bytes.
            for r in existing:
                expected={p['path']:p for p in r['proofs']}
                verify(api,repo,r['revision'],list(expected.values()))
        pending=[r for r in source['files'] if r['kind']!='metadata' and r['path'] not in done]
        for records in batches(pending,max_files=spec.get('batch_max_files',24),
                               max_bytes=spec.get('batch_max_bytes',4*2**30)):
            if max_batches is not None and completed_batches>=max_batches:
                save(status='bounded_run_finished');return state
            send(repo,records)
        # Anonymous path-info verifies publication visibility and content-addressed OIDs.
        head=state['repositories'][repo]['head']
        public=HfApi(token=False)
        for records in batches(source['files'],max_files=50,max_bytes=16*2**30):verify(public,repo,head,records,anonymous=True)
        state['repositories'][repo]['all_files_verified_public']=True;save(status='source_verified')
    assert state['verified_files']==spec['files'] and state['verified_bytes']==spec['bytes']
    save(status='all_source_payloads_verified',active_repository=None,active_files=0,active_bytes=0)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',default='runs/dataset_upload_v1/upload_spec.json');p.add_argument('--max-batches',type=int);a=p.parse_args()
    try:
        r=upload(Path(PROJECT_ROOT)/a.spec,max_batches=a.max_batches);print(json.dumps({'status':r['status'],'files':r['verified_files'],'bytes':r['verified_bytes']}))
    except Exception as e:
        # Avoid logging signed URLs, headers, tokens or arbitrary response text.
        print(json.dumps({'status':'stopped','error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None),'quota_error':is_quota_error(e)}),flush=True)
        sys.exit(1)
