"""Launch future work with fresh hardware limits and stage thread/temp settings."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json
from scripts.hardware_profile import inspect
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['cpu-prep','tokenization','tokenizer-train','sqlite','gpu']);ap.add_argument('--dry-run',action='store_true');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not cmd:ap.error('Provide command after --')
    r=inspect();preset=r['presets']['stages'][a.stage];scratch=Path(r['storage']['scratch_dir']);env=dict(os.environ);env.update(preset['env'])
    # Temporary files and compiler cache use the measured local filesystem, never /dev/shm.
    for name in ['tmp','sqlite_tmp','torchinductor']:(scratch/name).mkdir(parents=True,exist_ok=True,mode=0o700)
    env.update(TMPDIR=str(scratch/'tmp'),SQLITE_TMPDIR=str(scratch/'sqlite_tmp'),TORCHINDUCTOR_CACHE_DIR=str(scratch/'torchinductor'))
    python_script=len(cmd)>1 and Path(cmd[0]).name.startswith('python') and cmd[1].endswith('.py')
    if python_script:
        code="import runpy,sys; from pathlib import Path; import pyarrow as pa; pa.set_cpu_count(int(sys.argv[1])); pa.set_io_thread_count(int(sys.argv[1])); sys.argv=sys.argv[2:]; sys.path.insert(0,str(Path(sys.argv[0]).resolve().parent)); runpy.run_path(sys.argv[0],run_name='__main__')"
        command=[cmd[0],'-c',code,str(preset['arrow_threads'])]+cmd[1:]
    else:command=cmd
    plan=dict(stage=a.stage,hardware=r,preset=preset,applied_environment={k:env[k] for k in [*preset['env'],'TMPDIR','SQLITE_TMPDIR','TORCHINDUCTOR_CACHE_DIR']},python_arrow_limits_applied=python_script,command_program=Path(cmd[0]).name,argument_count=len(cmd)-1,scope='Thread/local-temp defaults only. Producer worker counts, RAM/disk caps, SQLite DB location and training approvals remain explicit per job. Never logs arbitrary command arguments or unrelated environment secrets.')
    target=Path(PROJECT_ROOT)/'logs/resource_launches'/str(time.time_ns());target.mkdir(parents=True);atomic_json(target/'plan.json',plan)
    print(json.dumps(dict(stage=a.stage,plan=str(target/'plan.json'),preset=preset)),flush=True)
    if not a.dry_run:
        if not r['presets']['launch_permitted']:raise SystemExit('Insufficient fresh memory headroom; child not started')
        sys.exit(subprocess.run(command,cwd=PROJECT_ROOT,env=env).returncode)
