"""Add reciprocal dataset navigation without changing payloads or release tags."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from scripts.upload_public_corpus import verify

ROOT_REPO = 'pathcosmos/korean-pretrain-corpus'
SOURCES = ['wikipedia', 'fineweb2', 'hplt2', 'fineweb-edu', 'math', 'code']
REPOS = [ROOT_REPO] + ['pathcosmos/korean-pretrain-' + s for s in SOURCES]
LANGUAGES = ['ko', 'en', 'zh-CN', 'zh-TW', 'ja', 'fr', 'de', 'es', 'pt-BR', 'hi', 'ar', 'he', 'ru']
BEGIN = '<!-- corpus-repository-links:start -->'
END = '<!-- corpus-repository-links:end -->'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def linked_card(data):
    text = data.decode('utf-8')
    assert BEGIN not in text and END not in text, 'Navigation already exists; inspect rather than duplicate'
    links = ['\n\n' + BEGIN, '## Repository links / 저장소 링크', '',
             '- [Corpus index / 통합 안내](https://huggingface.co/datasets/' + ROOT_REPO + ')']
    links += [f'- [{s}](https://huggingface.co/datasets/{repo})' for s, repo in zip(SOURCES, REPOS[1:])]
    links += ['', 'The corpus index links the six separate datasets; all existing repositories remain available.', END, '']
    return (text.rstrip() + '\n'.join(links)).encode('utf-8')


def prepare(base):
    assert not base.exists(), 'Preserve an existing preparation/receipt directory'
    base.mkdir(parents=True)
    api = HfApi()
    plan = {'utc': datetime.now(timezone.utc).isoformat(), 'scope': '20 cards plus root CHECKSUMS only',
            'delete_operations': 0, 'tag_changes': 0, 'payload_changes': 0, 'repositories': []}
    for repo in REPOS:
        info = api.repo_info(repo, repo_type='dataset')
        assert not info.private, repo
        names = ['README.md'] + ([f'cards/README.{lang}.md' for lang in LANGUAGES] if repo == ROOT_REPO else [])
        folder = base / repo.rsplit('/', 1)[1]
        files = []
        for name in names:
            original = Path(hf_hub_download(repo, name, repo_type='dataset', revision=info.sha)).read_bytes()
            changed = linked_card(original)
            for expected_repo in REPOS:
                assert f'https://huggingface.co/datasets/{expected_repo}' in changed.decode('utf-8')
            before = folder / 'before' / name
            after = folder / 'after' / name
            before.parent.mkdir(parents=True, exist_ok=True)
            after.parent.mkdir(parents=True, exist_ok=True)
            before.write_bytes(original)
            after.write_bytes(changed)
            files.append({'path': name, 'bytes': len(changed), 'sha256': digest(changed),
                          'before_sha256': digest(original), 'local_path': str(after.relative_to(Path(PROJECT_ROOT)))})
        if repo == ROOT_REPO:
            original = Path(hf_hub_download(repo, 'CHECKSUMS.json', repo_type='dataset', revision=info.sha)).read_bytes()
            checks = json.loads(original)
            by_path = {r['path']: r for r in checks['files']}
            assert all(r['path'] in by_path and by_path[r['path']]['sha256'] == r['before_sha256'] for r in files)
            for record in files:
                by_path[record['path']].update(bytes=record['bytes'], sha256=record['sha256'])
            changed = (json.dumps(checks, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            (folder / 'before' / 'CHECKSUMS.json').write_bytes(original)
            after = folder / 'after' / 'CHECKSUMS.json'
            after.write_bytes(changed)
            files.append({'path': 'CHECKSUMS.json', 'bytes': len(changed), 'sha256': digest(changed),
                          'before_sha256': digest(original), 'local_path': str(after.relative_to(Path(PROJECT_ROOT)))})
        assert sum(r['bytes'] for r in files) < 2 * 2**20
        plan['repositories'].append({'repository': repo, 'parent_commit': info.sha, 'files': files})
    assert sum(len(r['files']) for r in plan['repositories']) == 21
    atomic_json(base / 'plan.json', plan)
    return plan


def publish(base):
    plan = json.loads((base / 'plan.json').read_text())
    api, anonymous = HfApi(), HfApi(token=False)
    for repo in plan['repositories']:
        name = repo['repository']
        receipt = base / (name.rsplit('/', 1)[1] + '.receipt.json')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            verify(anonymous, name, old['commit'], repo['files'], anonymous=True)
            continue
        for item in repo['files']:
            assert digest((Path(PROJECT_ROOT) / item['local_path']).read_bytes()) == item['sha256']
        # Optimistic parent matching refuses concurrent changes; never overwrite tags or history.
        result = api.create_commit(repo_id=name, repo_type='dataset', parent_commit=repo['parent_commit'],
                                   commit_message='Link corpus index and all six existing source datasets in cards',
                                   operations=[CommitOperationAdd(path_in_repo=r['path'],
                                               path_or_fileobj=Path(PROJECT_ROOT) / r['local_path']) for r in repo['files']])
        atomic_json(receipt, {'repository': name, 'parent_commit': repo['parent_commit'], 'commit': result.oid,
                             'status': 'committed_verification_pending'})
        proofs = verify(anonymous, name, result.oid, repo['files'], anonymous=True)
        atomic_json(receipt, {'repository': name, 'parent_commit': repo['parent_commit'], 'commit': result.oid,
                             'status': 'passed_public_sha_verification', 'proofs': proofs})
    atomic_json(base / 'report.json', {'status': 'passed', 'cards': 20, 'files': 21, 'repositories_preserved': REPOS,
                                      'delete_operations': 0, 'tag_changes': 0, 'payload_changes': 0,
                                      'uploaded_bytes': sum(r['bytes'] for repo in plan['repositories'] for r in repo['files']),
                                      'scope': 'Current main-branch card navigation; immutable v1.0.0 snapshots unchanged'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    base = Path(PROJECT_ROOT) / args.run_dir
    if args.publish:
        publish(base)
    else:
        prepare(base)
    print(json.dumps({'run_dir': args.run_dir, 'published': args.publish}))
