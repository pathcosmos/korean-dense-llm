"""Verify the already-authorized public dataset destination, without logging secrets."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

REPO = 'pathcosmos/ko-en-code-math-corpus'
OUT = Path(__file__).resolve().parent


def main():
    api = HfApi()
    who = api.whoami()
    assert who['name'] == 'pathcosmos', 'Unexpected account'
    role = who.get('auth', {}).get('accessToken', {}).get('role')
    assert role in ('write', 'fineGrained'), 'Write-capable token required'
    created = False
    try:
        info = api.repo_info(REPO, repo_type='dataset')
        assert info.private is False, 'Existing repository is private; inspect before changing'
    except RepositoryNotFoundError:
        api.create_repo(REPO, repo_type='dataset', private=False, exist_ok=False)
        created = True
    info = api.repo_info(REPO, repo_type='dataset')
    files = api.list_repo_files(REPO, repo_type='dataset', revision=info.sha)
    if 'README.md' in files:
        raise RuntimeError('README already exists; preserve remote content and review before writing')
    body = '''---
pretty_name: Korean-English Code and Math Corpus
tags:
- reproducibility
- pretraining
---

# Korean-English Code and Math Corpus

This public repository is being prepared by pathcosmos as an index and reproduction
guide for six cleaned pretraining sources: Korean Wikipedia, Korean FineWeb2,
Korean HPLT2, English FineWeb-Edu, OpenWebMath, and filtered GitHub code.

**Publication status: preparation in progress. No corpus text or complete
reproduction bundle has been uploaded yet.** Source-specific attribution,
license conditions, manifests, processing code and verified release commits will
be documented before the full release. No blanket content license is claimed.

Historical local corpus size: 128,856,258 documents across six sources,
approximately 247.75 GB of compressed Parquet before public export metadata.
These figures describe the local corpus and are not the amount hosted here.
'''.encode()
    (OUT / 'public_README.md').write_bytes(body)
    commit = api.upload_file(path_or_fileobj=body, path_in_repo='README.md',
                             repo_id=REPO, repo_type='dataset',
                             parent_commit=info.sha,
                             commit_message='Initialize public corpus release index')
    remote = api.repo_info(REPO, repo_type='dataset', revision=commit.oid)
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{commit.oid}/README.md'
    # A fresh HTTP session with no Authorization header verifies public access.
    response = requests.Session().get(url, timeout=30)
    response.raise_for_status()
    assert response.content == body, 'Public download differs from upload'
    assert remote.private is False
    report = dict(utc=datetime.now(timezone.utc).isoformat(), status='passed',
                  account=who['name'], token_role=role, repository=REPO,
                  private=remote.private, repository_created=created,
                  commit=commit.oid, uploaded_file='README.md',
                  bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                  public_download_without_token_verified=True,
                  corpus_uploaded=False, paid_services_used=False)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('HF_WRITE_RESULT='+json.dumps(report))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Exception text can contain request data: print only safe classification.
        response = getattr(exc, 'response', None)
        print('HF_WRITE_ERROR='+json.dumps(dict(type=type(exc).__name__,
              http_status=getattr(response, 'status_code', None))))
        sys.exit(1)
