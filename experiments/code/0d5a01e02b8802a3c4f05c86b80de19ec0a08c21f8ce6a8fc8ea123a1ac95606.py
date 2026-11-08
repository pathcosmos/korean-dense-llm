"""Build local, metadata-only release drafts. Never upload or copy corpus text.

Existing output is refused. Snapshots are not a full clean-room reproduction.
Run from the project root: python scripts/prepare_dataset_release.py
"""
import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT

NAMES = {
    'wiki_ko': 'korean-wikipedia-cleaned',
    'fineweb2_ko': 'korean-fineweb2-cleaned',
    'hplt_ko': 'korean-hplt2-cleaned',
    'fineweb_edu': 'english-fineweb-edu-cleaned',
    'owm': 'open-web-math-cleaned',
    'github_code_v1': 'github-code-filtered',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out):
    root = Path(PROJECT_ROOT)
    if out.exists():
        raise FileExistsError('Preserve the existing draft; use a new --output path')
    registry = json.loads((root / 'docs/data_provenance_v1/source_registry.json').read_text())
    inventory = json.loads((root / 'docs/data_provenance_v1/file_inventory.json').read_text())['files']
    assert set(NAMES) == {x['source'] for x in registry['sources']}
    assert len(inventory) == 726
    out.mkdir(parents=True)
    package_root = out / 'packages'
    index = package_root / 'ko-en-code-math-corpus'
    index.mkdir(parents=True)
    copied = []

    def copy(rel, dest=None):
        src = root / rel
        target = index / (dest or ('snapshot/' + rel))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied.append({'source': rel, 'package_path': str(target.relative_to(index)),
                       'bytes': target.stat().st_size, 'sha256': digest(target)})

    releases = []
    for source in registry['sources']:
        key = source['source']
        folder = package_root / NAMES[key]
        folder.mkdir()
        rows = [x for x in inventory if x['source'] == key]
        assert len(rows) == source['final_files']
        assert sum(x['final']['rows'] for x in rows) == source['final_documents']
        assert sum(x['final']['bytes'] for x in rows) == source['final_bytes']
        write_json(folder / 'source.json', source)
        write_json(folder / 'files.json', {'scope': 'historical final and pinned upstream files; no corpus payload included', 'files': rows})
        (folder / 'README.md').write_text(
            '---\npretty_name: ' + NAMES[key] + '\ntags:\n- reproducibility\n- pretraining\n---\n\n'
            '# ' + NAMES[key] + '\n\n'
            '**Local release draft: metadata only. No text dataset is uploaded.**\n\n'
            f"Upstream: [{source['repository']}]({source['dataset_url']}), revision `{source['pinned_revision']}`.\n\n"
            f"Historical cleaned corpus: {source['final_documents']:,} documents, {source['final_bytes']:,} bytes. "
            'These are not counts of data hosted by this draft.\n\n'
            'Use `files.json` for exact upstream shard URLs, revisions and SHA256 values. '
            'Checksums of historical final files are prior audit evidence, not a fresh corpus rehash. '
            'Source-level licensing evidence and unresolved conditions are in `source.json`. '
            'The upstream database/packaging license does not automatically license every text.\n\n'
            'Reproduction bundle: planned `pathcosmos/ko-en-code-math-corpus`. '
            'Pin the verified release commit once it exists; no remote release commit exists yet. '
            'Original train/eval assignments must be retained during export.\n\n'
            'Publisher: pathcosmos. Proposed public contact: the repository Community tab. '
            'Private takedown contact is not yet configured.\n')
        releases.append({'source': key, 'repo_id': 'pathcosmos/' + NAMES[key],
                         'status': 'local_metadata_draft', 'commit': None,
                         'historical_bytes': source['final_bytes'], 'historical_documents': source['final_documents']})

    # Copy useful, explicit artifacts, never the workspace, logs, credentials or data tree.
    fixed = [
        'docs/DATASET_RELEASE_PLAN.md', 'docs/DATA_SOURCES_AND_LICENSES.md',
        'docs/DATA_RETENTION_AND_RECOVERY.md', 'docs/DATA_CONTRACT.md',
        'docs/data_provenance_v1/source_registry.json', 'docs/data_provenance_v1/file_inventory.json',
        'docs/data_provenance_v1/raw_schema_evidence.json',
        'docs/data_provenance_v1/legal_evidence.json', 'docs/data_provenance_v1/upstream_evidence.json',
        'data/datasets/final_v1.ready.json', 'data/datasets/final_v2.split/manifest.json',
        'data/clean/github_code_v1/policy.json', 'data/clean/github_code_v1/summary.json',
        'data/dedup/exact_v1/manifest.json', 'data/dedup/exact_v1/summary.json',
        'tokenizer/candidate_v1/tokenizer.json', 'data/tokens/final_v1/mix.json',
        'runs/f5_tokenizer_v1/efficiency.json', 'runs/f6_tokens_v1/counts.json',
        'requirements.txt', 'scripts/prepare_dataset_release.py',
    ]
    for rel in fixed:
        copy(rel)
    # Snapshot data/tokenizer modules and statically discover their local imports.
    pending = {str(p.relative_to(root)) for name in ('data', 'tokenizer', 'common')
               for p in (root / name).glob('*.py')}
    pending.update('scripts/' + n + '.py' for n in (
        'hardware_profile', 'run_with_resources', 'run_local_split', 'publish_local_split',
        'run_final_tokenizer', 'run_final_tokens', 'finalize_source_manifest', 'check_data_readiness'))
    done = set()
    while pending:
        rel = pending.pop()
        if rel in done:
            continue
        done.add(rel)
        tree = ast.parse((root / rel).read_text(), filename=rel)
        copy(rel)
        for node in ast.walk(tree):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            for name in names:
                candidate = name.replace('.', '/') + '.py'
                if (root / candidate).is_file() and candidate not in done:
                    pending.add(candidate)

    # Keep the execution-time recovery snapshot separate; extract only regular,
    # allowlisted code/manifests. Do not ship old .done markers as runnable state.
    historical = []
    archive = root / 'docs/recovery/data_retention_v1/recovery_bundle.tar.gz'
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            name = member.name
            if not member.isfile() or Path(name).is_absolute() or '..' in Path(name).parts:
                continue
            selected = name.endswith('.py') or name in ('environment.json', 'requirements.txt')
            selected |= name.startswith('data/raw/') and name.endswith('/download_manifest.json') and 'github_code_pilot' not in name
            selected |= name.startswith('data/clean/') and name.endswith('.stats.json') and 'github_code_pilot' not in name
            if not selected:
                continue
            data = tf.extractfile(member).read()
            target = index / 'historical_recovery' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            historical.append({'path': name, 'sha256': digest(target), 'bytes': len(data)})
    write_json(index / 'historical_recovery/provenance.json', {
        'original_archive_sha256': digest(archive), 'files': historical,
        'scope': 'selected execution-time recovery code/environment/download manifests; old done markers excluded'})
    write_json(index / 'snapshot_manifest.json', {'files': copied,
        'scope': 'current code snapshot; static local imports followed, dynamic dependencies not proven complete'})
    download_lines = []
    for source in registry['sources']:
        key = source['source']
        extra = ' 56' if key == 'fineweb_edu' else ' 64 --spread' if key == 'github_code_v1' else ''
        download_lines.append(f'PROJECT_ROOT="$REBUILD_ROOT" python "$RECIPE_ROOT/historical_recovery/data/download.py" {key}{extra} --revision {source["pinned_revision"]} --workers 4 --verify-sha256')
    (index / 'REBUILD.md').write_text('''# Rebuild instructions for this package

This is a staged recipe, not evidence of a completed full rebuild. First verify every
file against CHECKSUMS.json. Use a separate Python environment matching
historical_recovery/environment.json and requirements.txt; do not upgrade a running
training environment. Read the source licenses before fetching or redistributing data.
Evidence JSON files identify archived web snapshots by their old local paths; those
HTML/card snapshots are not included. Their official URLs and checksums are included.

From this repository root, choose a new local output directory on sufficient storage:

```bash
RECIPE_ROOT="$PWD"
REBUILD_ROOT="/your/local/scratch/new-corpus-rebuild"
test ! -e "$REBUILD_ROOT" || exit 1
mkdir -p "$REBUILD_ROOT/data/raw"
cp -a "$RECIPE_ROOT/historical_recovery/data/raw/." "$REBUILD_ROOT/data/raw/"
```

The following downloads approximately 394 GB of upstream data in total. Start by
appending `--list-only` to each command and checking the exact selected shards.
No download is performed by preparing this package. Worker 4 is a starting candidate;
measure on your machine and respect concurrent job memory/disk reservations.

```bash
''' + '\n'.join(download_lines) + '''
```

Use the historical clean.py with source/language/workers arguments:
wiki_ko/ko, fineweb2_ko/ko, hplt_ko/ko, fineweb_edu/en, owm/math.
Use clean_code.py github_code_v1 WORKERS for code. Set PROJECT_ROOT to
REBUILD_ROOT and use the thread-limiting wrapper in
snapshot/docs/DATA_RETENTION_AND_RECOVERY.md. Read that guide's filtering and
resource sections; its archive-extraction section is replaced by the setup above.
The original .done markers are deliberately not copied to the new output.

Deduplicate web sources in this exact order:

```bash
PROJECT_ROOT="$REBUILD_ROOT" python "$RECIPE_ROOT/historical_recovery/data/dedup.py" \\
  --sources wiki_ko fineweb2_ko hplt_ko fineweb_edu owm --output dedup/rebuilt_v1
```

Code stays outside the web dedup operation. Compare reconstructed final rows,
columns and hashes to each source repository's files.json, remapping the old
dedup/exact_v1 path to dedup/rebuilt_v1. Serializer byte differences require a full
logical-content comparison and a new receipt, not an assertion of matching SHA.
Per-shard historical cleaning .stats.json records are included under
historical_recovery/data/clean for comparison; they are not completion markers.

The exact final tokenizer is included in snapshot/tokenizer/candidate_v1/tokenizer.json.
Mix and full token counts are supplied under snapshot/data/tokens and snapshot/runs.
The canonical split manifest and split implementation are included, but the complete
document assignment database is not. Exporting a portable assignment sidecar or
verifying a logical rebuild is still required for exact train/eval reconstruction.
Do not run old ready manifests against missing historical proof paths: rebuild and
validate new source/audit receipts before split/tokenization. Code snapshots alone
are not a fully wired one-command reproduction of every downstream stage.
''')
    split = json.loads((root / 'data/datasets/final_v2.split/manifest.json').read_text())
    write_json(index / 'release.json', {
        'status': 'local_draft_not_uploaded', 'publisher': 'pathcosmos',
        'cost_policy': 'free_only', 'layout': 'six_source_repositories_and_one_recipe_repository',
        'all_sources_prepared_before_release': True,
        'cross_repository_atomic_visibility': False,
        'repositories': releases, 'split_contract_id': split['contract_id'],
        'full_split_assignments_included': False,
        'full_clean_room_rebuild_verified': False,
        'original_code_license': 'pending_selection_and_third_party_review',
        'public_storage_capacity': 'unconfirmed_account_level',
        'open_items': ['write token configured locally', 'public code license and third-party notices',
                       'split assignment sidecar or verified logical rebuild', 'full provenance recovery',
                       'text redistribution conditions', 'full outgoing privacy review',
                       'free account capacity for text', 'remote upload and integrity verification']})
    (index / 'README.md').write_text(
        '---\npretty_name: Korean-English Code and Math Corpus — Reproduction Draft\ntags:\n- reproducibility\n- pretraining\n---\n\n'
        '# Korean-English Code and Math Corpus\n\n'
        '**Local draft: no remote release and no corpus text included.**\n\n'
        'Six source inventories, pinned upstream revisions and SHA256 values, cleaning/splitting/tokenization '
        'code, final tokenizer, mix settings and recovery instructions are collected here. '
        'See `release.json` for source repositories and outstanding work.\n\n'
        'Start with `snapshot/docs/DATASET_RELEASE_PLAN.md` and '
        '`snapshot/docs/DATA_RETENTION_AND_RECOVERY.md`. Paths in historical manifests identify '
        'the original project artifacts, not files all included in this package. The recovery guide '
        'references the original full recovery archive; this draft instead includes selected files '
        'under `historical_recovery/`. Adapt paths and use a fresh output root. '
        'Do not use old completion markers to skip rebuilding missing payloads.\n\n'
        'The current snapshot and execution-time historical recovery snapshot are distinct. '
        'A clean-room full rebuild has not been verified. The canonical split contract is included, '
        'but the full assignment table is not; exact split recovery remains incomplete.\n\n'
        'License review: `snapshot/docs/DATA_SOURCES_AND_LICENSES.md`. '
        'No blanket content license or original-code open-source license is granted by this draft. '
        'Historical authorization fields record an earlier review, not the latest upload request.\n\n'
        'Publisher: pathcosmos. Proposed contact: HF Community; private takedown contact pending.\n')

    # This is a narrowly scoped automated scan, not a guarantee of no secrets/PII.
    suspicious = []
    secret_pattern = re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    files = sorted(p for p in package_root.rglob('*') if p.is_file())
    for p in files:
        if secret_pattern.search(p.read_bytes()):
            suspicious.append(str(p.relative_to(package_root)))
    if suspicious:
        raise RuntimeError('Credential-like patterns found; inspect file paths only: ' + repr(suspicious))
    for folder in package_root.iterdir():
        entries = [{'path': str(p.relative_to(folder)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                   for p in sorted(folder.rglob('*')) if p.is_file()]
        write_json(folder / 'CHECKSUMS.json', {'files': entries})
    # Few-file transport layout; extracted package paths remain unchanged.
    staging = out / 'upload_staging'
    for folder in package_root.iterdir():
        target = staging / folder.name
        target.mkdir(parents=True)
        if folder == index:
            with tarfile.open(target / 'reproduction.tar.gz', 'w:gz') as tf:
                for p in sorted(folder.rglob('*')):
                    if p.is_file():
                        tf.add(p, arcname=str(p.relative_to(folder)), recursive=False)
            for name in ('README.md', 'REBUILD.md', 'release.json'):
                shutil.copyfile(folder / name, target / name)
            with (target / 'README.md').open('a') as f:
                f.write('\nTransport: verify TRANSPORT_CHECKSUMS.json, then extract reproduction.tar.gz '
                        'into an empty directory and verify its internal CHECKSUMS.json before use.\n')
        else:
            for p in folder.iterdir():
                shutil.copyfile(p, target / p.name)
        write_json(target / 'TRANSPORT_CHECKSUMS.json', {'files': [
            {'path': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(target.iterdir()) if p.is_file()]})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'local_drafts_built',
              'source_repositories': len(releases), 'total_repositories': 7,
              'historical_final_files': len(inventory),
              'historical_documents': sum(x['final']['rows'] for x in inventory),
              'historical_bytes': sum(x['final']['bytes'] for x in inventory),
              'package_files': sum(1 for p in package_root.rglob('*') if p.is_file()),
              'package_bytes': sum(p.stat().st_size for p in package_root.rglob('*') if p.is_file()),
              'transport_files': sum(1 for p in staging.rglob('*') if p.is_file()),
              'transport_bytes': sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()),
              'credential_pattern_matches': suspicious, 'credential_scan_scope': 'HF/GitHub token forms and private-key headers only',
              'network_writes': 0, 'corpus_text_copied': False,
              'full_reproduction_verified': False, 'upload_ready': False}
    write_json(out / 'build_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/dataset_release_plan_v1/distributed_v1')
    args = parser.parse_args()
    path = Path(args.output)
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    print(json.dumps(build(path), ensure_ascii=False, indent=2))
