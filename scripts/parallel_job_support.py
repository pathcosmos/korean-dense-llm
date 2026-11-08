"""Bounded independent experiment processes; observe, never modify, the main split."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from common.data_contracts import resolve,atomic_json,sha256_file
from scripts.run_local_split import process_tree_rss,stop


def observe_main():
    root=resolve('runs/f4_full_v2')
    try:
        state=json.loads((root/'state.json').read_text());plan=json.loads((root/'plan.json').read_text())
        builder=next(Path(plan['scratch']).glob('split-*/build.json'))
        files=json.loads(builder.read_text())['contract']['files']
        path=builder.parent/'split.sqlite.building'
        if not path.exists():path=builder.parent/'split.sqlite'
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.1)
        try:p={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM progress')}
        finally:db.close()
        return dict(unix=time.time(),phase=p['phase'],rows=sum(f['rows'] for f in files[:p['file_index']])+p['row'],
                    main_state=state['phase'],main_peak_rss_bytes=state['peak_rss_bytes'])
    except (OSError,ValueError,sqlite3.Error,StopIteration,KeyError) as exc:
        return dict(unix=time.time(),observation_unavailable=type(exc).__name__)


def run_cases(root,cases,*,max_seconds=900,max_rss_gib=256):
    """Run an independent group concurrently, with one combined host RSS budget."""
    root=Path(root);jobs=[];start=time.monotonic();before=observe_main();peak=0
    try:
        for case in cases:
            out=Path(case['out']);out.mkdir(parents=True,exist_ok=False)
            with (out/'console.log').open('w') as stream:
                p=subprocess.Popen(case['cmd'],env=case['env'],cwd=resolve('.'),stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((case,p))
        while any(p.poll() is None for _,p in jobs):
            rss=sum(process_tree_rss(p.pid) for _,p in jobs if p.poll() is None);peak=max(peak,rss)
            elapsed=time.monotonic()-start
            atomic_json(root/'state.json',dict(phase='running',case=[c['name'] for c,_ in jobs],
                        elapsed_seconds=elapsed,peak_case_group_rss_bytes=peak,main=observe_main()))
            if elapsed>max_seconds or rss>max_rss_gib*1024**3:raise RuntimeError('Experiment time/RSS budget exceeded')
            if shutil.disk_usage(root).free<200*1024**3:raise RuntimeError('Durable space reserve')
            for case,p in jobs:
                scratch=case['env'].get('TMPDIR')
                if scratch and shutil.disk_usage(scratch).free<200*1024**3:raise RuntimeError('Local space reserve')
                if p.poll() not in (None,0):raise RuntimeError('Experiment subprocess failure: '+case['name'])
            time.sleep(2)
        failures={c['name']:p.returncode for c,p in jobs if p.returncode}
        if failures:raise RuntimeError('Experiment subprocess failure: '+str(failures))
        after=observe_main()
        summary=dict(cases=[c['name'] for c,_ in jobs],elapsed_seconds=time.monotonic()-start,
                     sampled_peak_rss_bytes=peak,main_before=before,main_after=after)
        if before.get('phase')==after.get('phase')=='ingest':
            summary['main_rows_per_second']=(after['rows']-before['rows'])/(after['unix']-before['unix'])
        atomic_json(root/('group_'+cases[0]['name']+'.json'),summary)
        return summary
    except BaseException as exc:
        for _,p in jobs:
            if p.poll() is None:stop(p)
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)))
        raise


def certificate(root,report,contract):
    root=Path(root)
    for name,expected in contract['code'].items():
        if sha256_file(resolve(name))!=expected:raise ValueError('Experiment dependency changed: '+name)
    atomic_json(root/'report.json',report)
    atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
    atomic_json(root/'state.json',dict(phase='complete'))
