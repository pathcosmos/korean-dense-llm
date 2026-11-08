"""Sequential bounded F2 evidence jobs. Never promotes ready or changes source data."""
import argparse,json,sys,subprocess,os,time,fcntl
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file
from scripts.storage_budget import guard,GIB

def main(out):
    root=Path(PROJECT_ROOT);out=(root/out).resolve();out.relative_to(root/'runs');out.mkdir(exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=['scripts/run_f2_checks.py','data/probe_audit_overlap.py','scripts/measure_split.py','data/split_manifest.py','data/sample_snapshot.py','data/probe_text_similarity.py']
        contract=dict(code={p:sha256_file(root/p) for p in sources},plan_sha256=sha256_file(out/'plan.json'),frame_sha256=sha256_file(out/'frame.json'))
        cp=out/'runner_contract.json'
        if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Execution code/config changed; new run required')
        atomic_json(cp,contract);(out/'pid').write_text(str(os.getpid()));env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2');results=[]
        def state(kind,**kw):
            atomic_json(out/'state.json',dict(status=kind,done=len(results),total=3,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw))
        def run(name,cmd,seconds):
            guard(8*GIB,root,200*GIB);state('running',case=name);print('START',name,flush=True)
            with (out/(name+'.log')).open('a') as log:
                p=subprocess.Popen(['timeout','--kill-after=30s',str(seconds)+'s']+cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    state('running',case=name,child_pid=p.pid);time.sleep(5)
                if p.returncode:state('error',case=name,returncode=p.returncode);raise RuntimeError(name+' failed: '+str(p.returncode))
        try:
            run('full_overlap',[sys.executable,'data/probe_audit_overlap.py','--audit','runs/source_audit_full_v1/report.json','--output',str(out/'overlap/report.json'),'--max-seconds','1800'],2200)
            report=json.loads((out/'overlap/report.json').read_text())
            if not report['full_source_coverage']:raise ValueError('Full overlap coverage incomplete')
            results.append(dict(step='full_overlap',report=str(out/'overlap/report.json'),sha256=sha256_file(out/'overlap/report.json')));print('DONE full_overlap',flush=True)
            run('split_million',[sys.executable,'scripts/measure_split.py','--output',str(out/'scale'),'--sizes','1000000','--max-case-seconds','1800','--max-rss-gib','4','--max-case-gib','8','--min-free-gib','200','--sqlite-cache-mib','64'],1900)
            results.append(dict(step='split_million',report=str(out/'scale/summary.json'),sha256=sha256_file(out/'scale/summary.json')));print('DONE split_million',flush=True)
            snippet="from data.sample_snapshot import sample; from data.probe_text_similarity import run; from pathlib import Path; import sys; p=Path(sys.argv[1]); r=sample(p/'frame.json',p/'samples',per_source=2000,max_bytes=128*1024**2,max_read_bytes=2*1024**3,max_row_groups=24,max_files_per_source=4,row_groups_per_file=2,max_seconds=300); print('SAMPLE',r['sample_status'],r['stats'],flush=True); [(print('SIMILARITY',t,run(p/'samples/manifest.json',p/('similarity_'+str(t)+'.json'),threshold=t,max_candidates=50000,max_seconds=180,max_view_chars=2048)['status'],flush=True)) for t in [.8,.9,.95]]"
            run('web_sample',[sys.executable,'-u','-c',snippet,str(out)],1000)
            results.append(dict(step='web_sample',report=str(out/'samples/manifest.json'),sha256=sha256_file(out/'samples/manifest.json')))
            assert all(sha256_file(root/p)==h for p,h in contract['code'].items())
            atomic_json(out/'summary.json',dict(status='measurement_complete',steps=results,quality_readiness='pending',scope='Full whole/fence equality; bounded synthetic split and web samples. Not universal duplicate-free or ready approval.'))
            atomic_json(out/'.done',dict(summary_sha256=sha256_file(out/'summary.json')));state('completed');print('DONE all evidence jobs',flush=True)
        except Exception:
            state('error',case='see_stage_log');raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();main(Path(a.output))
