"""Verify the exact source-only release manifest (not excluded datasets)."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release/files.json').read_text())
for row in manifest['files']:
    path=root/row['path']
    if path.is_symlink() or not path.is_file():raise ValueError('Missing or linked: '+row['path'])
    data=path.read_bytes()
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Changed: '+row['path'])
print(json.dumps(dict(status='passed',files=len(manifest['files']),bytes=sum(r['bytes'] for r in manifest['files']))))
