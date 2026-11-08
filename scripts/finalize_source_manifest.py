"""Create a new audited ready source only after all quality conditions pass.
Does not train, alter data, or promote pending/partial audit results.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,identity,sha256_file,atomic_json,artifact_path
from common.source_inventory import definition,load_audit,verify_bound
from data.quality_report import assess


def finalize(candidate,audit_path,policy,dedup_root,output):
    candidate,audit_path,policy,output=map(resolve,(candidate,audit_path,policy,output))
    raw=json.loads(candidate.read_text());audit=load_audit(audit_path)
    if definition(raw)!=audit['definition']:raise ValueError('Candidate differs from audited source definition')
    readiness=assess(candidate,dedup_root,policy)
    declaration=json.loads(policy.read_text()).get('checks',{}).get('content_audit',{})
    pinned=[(artifact_path(resolve(p['path'])),p['sha256']) for p in declaration.get('evidence',[])]
    if pinned!=[(artifact_path(audit_path),sha256_file(audit_path))]:raise ValueError('Quality assessment must pin the selected full audit')
    if readiness['readiness']!='passed':raise ValueError('Data readiness is not passed')
    for source in audit['content_inventory']['sources'].values():
        for f in source['files']:
            if sha256_file(resolve(f['path']))!=f['sha256']:raise ValueError('Source changed after audit')
    ready=dict(schema_version=1,status='ready',sources=audit['definition']['sources'],
        content_inventory_id=audit['content_inventory_id'],content_inventory=audit['content_inventory'],
        audit_proof=dict(path=artifact_path(audit_path),sha256=sha256_file(audit_path)),
        review=dict(candidate_sha256=sha256_file(candidate),policy_sha256=sha256_file(policy),readiness_contract_id=readiness['contract_id']))
    marker=Path(str(output)+'.done')
    if output.exists() and json.loads(output.read_text())!=ready:raise ValueError('Ready output differs; use a new path')
    if not output.exists():atomic_json(output,ready)
    atomic_json(marker,dict(sha256=sha256_file(output),content_inventory_id=ready['content_inventory_id']))
    for source in ready['sources']:verify_bound(output,ready,source)
    return ready


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ['candidate','audit','policy','dedup-root','output']:ap.add_argument('--'+field,required=True)
    a=ap.parse_args();r=finalize(a.candidate,a.audit,a.policy,a.dedup_root,a.output)
    print(json.dumps(dict(status=r['status'],content_inventory_id=r['content_inventory_id'])))
