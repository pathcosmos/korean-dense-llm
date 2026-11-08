"""Shared identities and input-path contract for tokenization/tokenizer/mix.
All data artifacts are identified by content, not vocabulary size or filename alone.
"""
import hashlib
import json
from pathlib import Path
from common.paths import PROJECT_ROOT

SCHEMA_VERSION = 1


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else Path(PROJECT_ROOT) / p


def relative(path):
    return str(Path(path).resolve().relative_to(Path(PROJECT_ROOT).resolve()))


def artifact_path(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path(PROJECT_ROOT).resolve()))
    except ValueError:
        return str(p)  # explicit external fixtures/volumes


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def source_files(source, manifest_path=None, input_dir=None, *, return_verified_records=False, verification_workers=1):
    if manifest_path and input_dir:
        raise ValueError('Choose sources manifest or input directory')
    if manifest_path:
        manifest = json.loads(resolve(manifest_path).read_text())
        if manifest.get('status') != 'ready':
            raise ValueError('Source manifest is a draft, not ready for use')
        if manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported source manifest schema')
        entry = manifest['sources'][source]
        if not entry.get('text_policy') or not entry.get('completion_marker'):
            raise ValueError('Source requires text policy and completion evidence')
        if not resolve(entry['completion_marker']).exists():
            raise ValueError('Source snapshot is not complete')
        files = [resolve(p) for p in entry['files']]
        snapshot = identity(manifest)
    else:
        directory = resolve(input_dir or f'data/clean/{source}')
        files = sorted(directory.glob('*.parquet'))
        snapshot = None
    if not files or len(set(map(lambda p: str(p.resolve()), files))) != len(files):
        raise ValueError('Source must contain nonempty, unique input files')
    if len(set(p.stem for p in files)) != len(files):
        raise ValueError('Input basenames must be unique')
    for file in files:
        if not file.is_file() or not Path(str(file) + '.done').exists():
            raise ValueError(f'Input parquet or completion marker missing: {file}')
    bound=None
    if manifest_path:
        from common.source_inventory import verify_bound
        bound=verify_bound(manifest_path,manifest,source,workers=verification_workers)
    if return_verified_records:return files,snapshot,bound
    return files, snapshot
