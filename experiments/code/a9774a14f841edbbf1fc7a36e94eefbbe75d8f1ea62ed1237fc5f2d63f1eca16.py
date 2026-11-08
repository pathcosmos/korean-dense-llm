"""Freeze a bounded public recipe from the actual training contract. No upload."""
import ast
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT


def build(destination):
    root = Path(PROJECT_ROOT)
    out = root / destination
    if out.exists():
        raise FileExistsError('Use a new snapshot path')
    contract = json.loads((root/'runs/3b_full_v1/contract.json').read_text())
    for rel, expected in contract['code'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected, rel
    out.mkdir(parents=True)
    manifest=[]
    def copy(rel, target=None):
        p=out/(target or 'code/'+rel);p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/rel,p)
        manifest.append({'path':str(p.relative_to(out)),'origin':rel,'bytes':p.stat().st_size,
                         'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    pending=set(contract['code']) | {'data/clean.py','data/clean_code.py','data/filters.py','data/download.py',
        'data/dedup.py','data/release_provenance.py','scripts/prepare_public_recipe.py',
        'tests/test_release_provenance.py'}
    seen=set()
    while pending:
        rel=pending.pop()
        if rel in seen:continue
        seen.add(rel); tree=ast.parse((root/rel).read_text(),filename=rel);copy(rel)
        for n in ast.walk(tree):
            names=[n.module] if isinstance(n,ast.ImportFrom) and n.module else []
            if isinstance(n,ast.Import):names += [a.name for a in n.names]
            for name in names:
                candidate=name.replace('.','/')+'.py'
                if (root/candidate).is_file() and candidate not in seen:pending.add(candidate)
    copy('requirements.txt','code/requirements.txt')
    copy('runs/3b_full_v1/main.json','recipe/main.original.json')
    copy('data/tokens/final_v1/mix.json','recipe/mix.original.json')
    copy('tokenizer/candidate_v1/tokenizer.json','tokenizer/tokenizer.json')
    copy('runs/f6_tokens_v1/counts.json','recipe/token_counts.json')
    for rel in ['source_registry.json','file_inventory.json','raw_schema_evidence.json']:
        copy('docs/data_provenance_v1/'+rel,'sources/'+rel)
    environment={}
    for name in ['torch','numpy','pyarrow','tokenizers','huggingface_hub','xxhash','safetensors','pytest']:
        try:environment[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:environment[name]=None
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    (out/'LICENSE-CODE').write_text('''MIT License

Copyright (c) 2026 pathcosmos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
''')
    (out/'LICENSE-SCOPE.md').write_text('''# License scope

Original project code under code/: MIT, copyright 2026 pathcosmos.
Original project prose documentation: CC BY 4.0, attribution pathcosmos,
https://creativecommons.org/licenses/by/4.0/ . Indicate modifications and retain
the attribution, license link and applicable notices; no warranties are given.
This statement excludes third-party license texts, quoted material and source
data, which retain their own conditions. The tokenizer and derived artifacts
are not blanket-licensed as corpus content by these code/document grants.

The source registry records database/packaging conditions and outstanding
content attribution work. It is not a per-document redistribution clearance.
The historical upload_authorized=false field in that registry records an older
audit; the publisher has since authorized public publication preparation.

Python packages are referenced, not vendored. They retain their own licenses.
No upstream dataset corpus text or model checkpoint is copied into this recipe.

Official MIT text: https://opensource.org/license/mit
CC BY 4.0 legal code: https://creativecommons.org/licenses/by/4.0/legalcode.en

Publisher and private takedown contact: pathcosmos, miracle.yang@pathcosmos.com.
''')
    (out/'CONTACT.md').write_text('''# Contact and correction requests

Publisher: pathcosmos. Email: miracle.yang@pathcosmos.com.
Use this email for private removal requests. Include a source URL, doc_id, or
repository/path and the relevant concern. Do not post private information in
public discussions. No guaranteed response time is promised.

Verified corrections must cover text, tokens, provenance and release manifests
together. Prior pinned versions may remain accessible; history/cache removal
requires a separate operational decision. No automatic remote deletion is run
by this package.
''')
    (out/'README.md').write_text('''# Public reproduction recipe — preparation snapshot

This is a code, tokenizer and source-metadata snapshot. Dataset payload publication
and clean-room end-to-end validation are still pending. The 49 actual training
producer files were verified against the main-training contract before copying.
Additional cleaning/provenance helpers and statically discovered local imports
are included. Dynamic imports/runtime scripts may need further packaging.

Original code: MIT. Original prose: CC BY 4.0. See LICENSE-SCOPE.md.
Contact: miracle.yang@pathcosmos.com.

recipe/main.original.json is the actual selected training configuration.
recipe/mix.original.json contains original relative paths: public download/path
remapping and a bounded smoke entry point are still to be supplied. Do not start
the full 57,000-step training command as a package verification step.

The tokenizer file is the original raw tokenizers JSON. AutoTokenizer support
has not yet been verified. Source metadata checksums refer to historical corpus
proofs; corpus payload is not included here. Portable split assignments and
token offsets are being prepared separately.
''')
    # Narrow credential scan, with filenames only if a match is found.
    token=re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    matches=[]
    for p in out.rglob('*'):
        if p.is_file() and token.search(p.read_bytes()):matches.append(str(p.relative_to(out)))
    assert not matches, matches
    for r in manifest:
        assert hashlib.sha256((out/r['path']).read_bytes()).hexdigest()==r['sha256']
    report={'status':'snapshot_verified_not_clean_room_reproduction','producer_code_files':len(contract['code']),
        'python_modules':len(seen),'files':manifest,'environment':environment,'credential_pattern_matches':matches,
        'original_code_license':'MIT','original_prose_license':'CC-BY-4.0','public_contact':'miracle.yang@pathcosmos.com',
        'corpus_payload_included':False,'fresh_consumer_validation':False,'network_uploads':0}
    (out/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    checks=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'CHECKSUMS.json').write_text(json.dumps({'files':checks},indent=2)+'\n')
    return {'producer_code_files':len(contract['code']),'python_modules':len(seen),'files':len(checks)+1,
            'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'output':str(out.relative_to(root))}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(build(a.output)))
