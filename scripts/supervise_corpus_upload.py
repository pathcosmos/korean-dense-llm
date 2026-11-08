"""30-second local upload observation and bounded memory protection. No messages sent."""
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from scripts.upload_public_corpus import atomic_json


if __name__=='__main__':
    root=Path(PROJECT_ROOT);base=root/'runs/dataset_upload_v1';spec=json.loads((base/'upload_spec.json').read_text())
    if (base/'supervisor.done').exists():raise RuntimeError('Previous supervisor finished; inspect its report')
    (base/'supervisor.pid').write_text(str(os.getpid())+'\n')
    env=dict(os.environ);env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        RAYON_NUM_THREADS='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_XET='1',HF_HUB_ENABLE_HF_TRANSFER='0')
    with (base/'upload.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,'-u','scripts/upload_public_corpus.py'],cwd=root,env=env,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (base/'uploader.pid').write_text(str(child.pid)+'\n');last=None;stopped_memory=False
    print(json.dumps({'upload_supervisor_pid':os.getpid(),'uploader_pid':child.pid,'interval_seconds':30}),flush=True)
    while True:
        p=base/'upload_state.json';state=json.loads(p.read_text()) if p.exists() else {}
        rss=None
        try:
            proc=dict(line.split(':',1) for line in Path(f'/proc/{child.pid}/status').read_text().splitlines() if ':' in line)
            rss=int(proc['VmRSS'].split()[0])*1024
        except (OSError,KeyError):pass
        if rss is not None and rss>4*2**30:
            child.terminate();stopped_memory=True
        status=state.get('status','starting')
        observation={'status':'running','phase':status,'done':state.get('verified_files',0),'total':spec['files'],
            'verified_bytes':state.get('verified_bytes',0),'target_bytes':spec['bytes'],
            'active_repository':state.get('active_repository'),'active_bytes':state.get('active_bytes'),
            'uploader_rss_bytes':rss,'uploader_rss_limit_bytes':4*2**30,'pid':os.getpid(),'uploader_pid':child.pid,
            'utc':datetime.now(timezone.utc).isoformat()}
        returncode=child.poll()
        if returncode is not None:
            success=returncode==0 and status=='all_source_payloads_verified'
            observation['status']='complete' if success else 'error'
            report={'status':'passed' if success else 'failed','scope':'six_source_snapshot_payload_transfer',
                'returncode':returncode,'quota_stopped':status=='stopped_quota','memory_stopped':stopped_memory,
                'upload_state':state,'full_dataset_release_index_finalized':False}
            atomic_json(base/'supervisor_report.json',report)
            atomic_json(base/'supervisor.done',{'report_sha256':hashlib.sha256((base/'supervisor_report.json').read_bytes()).hexdigest()})
            atomic_json(base/'supervisor_state.json',observation)
            print(json.dumps({'status':report['status'],'verified_files':observation['done'],'verified_bytes':observation['verified_bytes'],'quota_stopped':report['quota_stopped']}),flush=True)
            break
        atomic_json(base/'supervisor_state.json',observation)
        key=(status,observation['done'])
        if key!=last:
            with (root/'logs/pipeline_events.jsonl').open('a') as f:
                f.write(json.dumps({'utc':observation['utc'],'stage':'corpus_public_upload','state':status,
                    'done':observation['done'],'total':spec['files'],'verified_bytes':observation['verified_bytes']})+'\n')
            last=key
        time.sleep(30)
