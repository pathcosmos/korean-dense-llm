"""Content-bound local reader manifest for a closed, published split.
The canonical manifest remains the logical input. A fresh SplitSelection still
validates actual local DB SHA, integrity, leakage and full row coverage.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from common.data_contracts import resolve,artifact_path,atomic_json,identity,sha256_file


def closed_manifest(path):
    path=resolve(path);report=json.loads(path.read_text());marker=path.parent/'.done'
    if (report.get('status')!='validated' or not marker.exists() or
            json.loads(marker.read_text())!=dict(contract_id=report['contract_id'],manifest_sha256=sha256_file(path)) or
            identity(report['contract'])!=report['contract_id']):raise ValueError('Canonical split is not certified')
    return report


def prepare_reader(canonical,output,*,local_database=None,scratch=None):
    canonical=resolve(canonical);report=closed_manifest(canonical);canonical_sha=sha256_file(canonical)
    out=resolve(output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();copied=False
    original=resolve(report['database']['path']);size=original.stat().st_size;expected=report['database']['sha256']
    if local_database is not None:
        local=resolve(local_database)
        if local.name.endswith('.building') or local.stat().st_size!=size:raise ValueError('Local closed DB size/path differs')
    else:
        if scratch is None:raise ValueError('Need existing local database or measured scratch directory')
        directory=resolve(scratch);directory.mkdir(parents=True,exist_ok=True);local=directory/('reader-'+expected[:24]+'.sqlite')
        if not local.exists():
            partial=Path(str(local)+'.copying');offset=partial.stat().st_size if partial.exists() else 0
            if offset>size:raise ValueError('Reader partial exceeds canonical DB size')
            digest=hashlib.sha256()
            with original.open('rb') as src,partial.open('r+b' if partial.exists() else 'xb') as dst:
                # Resume only an exactly matching prefix; preserve a wrong partial.
                remaining=offset
                while remaining:
                    a=src.read(min(8*1024**2,remaining));b=dst.read(len(a))
                    if not a or a!=b:raise ValueError('Reader partial prefix differs')
                    digest.update(a);remaining-=len(a)
                while chunk:=src.read(8*1024**2):dst.write(chunk);digest.update(chunk)
                dst.flush();os.fsync(dst.fileno())
            if partial.stat().st_size!=size or digest.hexdigest()!=expected:raise ValueError('Reader copy SHA differs')
            partial.rename(local);copied=True
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    if not local.is_file() or local.stat().st_size!=size:raise ValueError('Local DB size differs')
    if any(Path(str(local)+s).exists() for s in ['-journal','-wal']):raise ValueError('Reader DB must be closed, without journal/WAL')
    derived=copy.deepcopy(report);derived['database']['path']=artifact_path(local)
    target=out/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=derived:raise ValueError('Reader contract changed; use a new output')
    if sha256_file(canonical)!=canonical_sha:raise ValueError('Canonical manifest changed during reader preparation')
    if not target.exists():atomic_json(target,derived)
    atomic_json(out/'.done',dict(contract_id=derived['contract_id'],manifest_sha256=sha256_file(target)))
    atomic_json(out/'origin.json',dict(canonical=artifact_path(canonical),canonical_sha256=canonical_sha,local_database=artifact_path(local),database_sha256=expected,
                copied=copied,copy_or_reuse_seconds=time.perf_counter()-start,
                scope='Reader manifest only. Fresh local DB SHA/integrity/coverage validation by VerifiedSplitContext remains required.'))
    return target
