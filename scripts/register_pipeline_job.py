"""Register independent jobs without replacing an existing run definition."""
import argparse,json,fcntl,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json
from scripts.job_registry import validate,ensure_watcher
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args();spec=validate(json.loads(Path(a.spec).read_text()));log=Path(PROJECT_ROOT)/'logs';p=log/'independent_jobs.json'
    with (log/'independent_jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(p.read_text()) if p.exists() else dict(schema_version=1,jobs=[])
        if d.get('schema_version')!=1:raise ValueError('Invalid registry schema')
        old=[x for x in d['jobs'] if x['run_id']==spec['run_id']]
        if old and old!=[spec]:raise ValueError('Registration differs; use a new run_id')
        if not old:d['jobs'].append(spec);atomic_json(p,d)
        watcher=ensure_watcher(PROJECT_ROOT)
    print('Registered',spec['run_id'],'watcher',watcher or 'already_running')
