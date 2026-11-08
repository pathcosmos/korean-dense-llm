"""Canonical source definitions and audited content bindings, independent of status."""
import json
from pathlib import Path
from common.data_contracts import resolve,artifact_path,identity,sha256_file


def definition(inventory):
    if inventory.get('schema_version')!=1 or not isinstance(inventory.get('sources'),dict) or not inventory['sources']:
        raise ValueError('Invalid source inventory')
    sources={};seen=set()
    for source,e in inventory['sources'].items():
        if e.get('text_policy') not in ('raw-utf8-code-v1','nfc-web-exact-v1') or not e.get('files') or not e.get('completion_marker'):
            raise ValueError('Source definition missing policy/files/completion')
        paths=[artifact_path(resolve(p)) for p in e['files']]
        if len(set(Path(p).stem for p in paths))!=len(paths):raise ValueError('Repeated source basename')
        for p in paths:
            if p in seen:raise ValueError('Repeated inventory file')
            seen.add(p)
        sources[source]=dict(files=paths,text_policy=e['text_policy'],origin=e.get('origin',{}),
                             completion_marker=artifact_path(resolve(e['completion_marker'])))
    return dict(version=1,sources=sources)


def content_inventory(source_definition,audited_sources):
    if set(source_definition['sources'])!=set(audited_sources):raise ValueError('Audited source coverage differs')
    sources={}
    for name,e in source_definition['sources'].items():
        files=audited_sources[name]['files']
        if [f['path'] for f in files]!=e['files']:raise ValueError('Audited file coverage/order differs')
        sources[name]=dict(text_policy=e['text_policy'],origin=e['origin'],
            files=[{k:f[k] for k in ['path','bytes','rows','sha256']} for f in files])
    return dict(version=1,sources=sources)


def load_audit(path,require_full=True):
    path=resolve(path);r=json.loads(path.read_text());done=path.parent/'.done'
    if r.get('format')!='source-audit-v1' or r.get('run_status')!='completed' or not done.exists() or json.loads(done.read_text())!=dict(report_sha256=sha256_file(path),contract_id=r['contract_id']):
        raise ValueError('Uncertified source audit')
    if identity(r['contract'])!=r['contract_id']:raise ValueError('Audit contract changed')
    if require_full and (r.get('scope')!='full' or r.get('structural_status')!='passed'):
        raise ValueError('Full audit required; pilot does not certify full inventory')
    if require_full:
        ci=content_inventory(r['definition'],r['sources'])
        if ci!=r.get('content_inventory') or identity(ci)!=r.get('content_inventory_id'):raise ValueError('Audit content inventory mismatch')
    return r


def verify_bound(manifest_path,inventory,source):
    """Return a bound source's records after proof and actual content verification."""
    fields={'content_inventory_id','content_inventory','audit_proof'}
    if not fields.intersection(inventory):return None  # explicit legacy fixtures remain supported
    if not fields.issubset(inventory):raise ValueError('Incomplete audited source binding')
    manifest_path=resolve(manifest_path);done=Path(str(manifest_path)+'.done')
    if not done.exists() or json.loads(done.read_text())!=dict(sha256=sha256_file(manifest_path),content_inventory_id=inventory['content_inventory_id']):
        raise ValueError('Uncertified ready source manifest')
    proof=inventory['audit_proof'];audit_path=resolve(proof['path'])
    if sha256_file(audit_path)!=proof['sha256']:raise ValueError('Bound audit content changed')
    audit=load_audit(audit_path)
    if definition(inventory)!=audit['definition'] or inventory['content_inventory']!=audit['content_inventory'] or inventory['content_inventory_id']!=audit['content_inventory_id']:
        raise ValueError('Ready content differs from audited definition')
    files=inventory['content_inventory']['sources'][source]['files']
    for f in files:
        p=resolve(f['path'])
        if not p.is_file() or p.stat().st_size!=f['bytes'] or sha256_file(p)!=f['sha256']:
            raise ValueError('Ready source content changed: '+f['path'])
    return files
