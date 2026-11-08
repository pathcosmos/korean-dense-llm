"""Read-only inventory/content audit of the exact approved10+15 namespaces.

Does not instantiate the retention manager: even its read-like methods may
repair certificates or create directories. Snapshot and final acceptance are
different operations. Final acceptance requires actual completed main evidence
and fresh hashes of every resident full-state checkpoint.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.tiered_checkpoint_retention import inventory, records, size_tree
from train.final_resume_probe import file_stats, stat_record

NAME = re.compile(r'ckpt_[0-9]{7}')
GIB = 2**30


def existing_ancestor(path):
    path = Path(path)
    while not path.exists():
        if path.parent == path:
            raise ValueError('No existing storage ancestor')
        path = path.parent
    if any(p.is_symlink() for p in [path,*path.parents]):
        raise ValueError('Storage ancestor symlink rejected')
    return path


class ChangedDuringAudit(RuntimeError):
    pass


class Snapshot:
    def __init__(self):
        self.files = {}; self.directories = {}; self.checkpoints = {}

    def read(self,path,*,lines=False):
        path = Path(path); before = stat_record(path); raw = path.read_bytes()
        if stat_record(path) != before:
            raise ChangedDuringAudit('File changed while reading: '+str(path))
        self.files[str(path)] = (before, __import__('hashlib').sha256(raw).hexdigest())
        if lines:
            if raw and not raw.endswith(b'\n'):
                raise ChangedDuringAudit('Incomplete journal append')
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        return json.loads(raw)

    def children(self,path):
        path = Path(path)
        if any(p.is_symlink() for p in [path,*path.parents]):
            raise ValueError('Audit namespace symlink rejected')
        if not path.exists():
            value = None; children = []
        else:
            children = sorted(path.iterdir()); value = [(p.name,p.is_dir(),p.is_symlink()) for p in children]
            if any(p.is_symlink() for p in children):
                raise ValueError('Audit child symlink rejected')
        self.directories[str(path)] = value
        return children

    def unchanged(self):
        for name,(expected,_) in self.files.items():
            if stat_record(Path(name)) != expected:
                raise ChangedDuringAudit('Audit metadata changed: '+name)
        for name,expected in self.directories.items():
            path = Path(name)
            value = [(p.name,p.is_dir(),p.is_symlink()) for p in sorted(path.iterdir())] if path.exists() else None
            if value != expected:
                raise ChangedDuringAudit('Audit namespace changed: '+name)
        for name,expected in self.checkpoints.items():
            if file_stats(Path(name)) != expected:
                raise ChangedDuringAudit('Checkpoint changed during audit: '+name)


def checkpoint_content(path, proof, parent, config, *, hashes, production, snapshot):
    path = Path(path); before = file_stats(path)
    inventory(path,proof,hashes=hashes,workers=8)
    expected = {row['path']:row for row in proof['files']}
    # Even light snapshots freshly verify small resume metadata and all RNGs;
    # large payload hashes may be reused only in non-final observations.
    for name in ['meta.json','.metadata','.complete',*[f'rng_rank{r}.pt' for r in range(8)]]:
        if not hashes and sha256_file(path/name) != expected[name]['sha256']:
            raise ValueError('Checkpoint resume metadata SHA differs: '+name)
    meta = snapshot.read(path/'meta.json')
    saved_config = dict(config,out_dir=str(resolve(config['out_dir'])))
    if (meta['training_contract'] != parent or meta['config'] != saved_config or
            meta['step'] != proof['step'] or meta['world_size'] != 8 or
            meta['tokens_per_step'] != config['global_tokens']):
        raise ValueError('Checkpoint model/input/code/config lineage differs')
    if production and meta.get('sampler_version') != config['sampler_version']:
        raise ValueError('Checkpoint resume sampler version differs')
    import torch
    import torch.distributed.checkpoint as dcp
    from scripts.supervise_main_training import validate_optimizer_map
    metadata = dcp.FileSystemReader(path).read_metadata(); validate_optimizer_map(metadata)
    if not metadata.storage_data:
        raise ValueError('Checkpoint DCP storage map absent')
    payloads = set()
    for value in metadata.storage_data.values():
        if (value.relative_path not in expected or not value.relative_path.endswith('.distcp') or
                value.offset < 0 or value.length <= 0 or value.offset+value.length > expected[value.relative_path]['bytes']):
            raise ValueError('Checkpoint DCP map is outside certified payload')
        payloads.add(value.relative_path)
    state = metadata.state_dict_metadata
    model_items = {k[6:]:v for k,v in state.items() if k.startswith('model.')}
    tensor_count = sum(hasattr(v,'size') for v in state.values())
    if production:
        from model.model import Transformer, get_config
        with torch.device('meta'):
            model = Transformer(get_config(config['model'],vocab_size=config['vocab_size'],max_seq_len=config['seq_len']))
        wanted = model.state_dict()
        if (len(payloads) != 8 or set(model_items) != set(wanted) or len(model_items) != 255 or tensor_count != 1020):
            raise ValueError('Actual3B8rank full-state coverage differs')
        for name,value in wanted.items():
            item = model_items[name]
            if tuple(item.size) != tuple(value.shape) or item.properties.dtype != value.dtype:
                raise ValueError('Actual3B model tensor geometry differs: '+name)
    for rank in range(8):
        rng = torch.load(path/f'rng_rank{rank}.pt',weights_only=True,map_location='cpu')
        if set(rng) != {'cpu','cuda'} or any(not isinstance(t,torch.Tensor) or t.dtype != torch.uint8 or t.ndim != 1 for t in rng.values()):
            raise ValueError('Checkpoint rank RNG coverage/type differs')
        if rng['cpu'].numel() != torch.get_rng_state().numel() or (production and rng['cuda'].numel() == 0):
            raise ValueError('Checkpoint rank RNG length differs')
    if file_stats(path) != before:
        raise ChangedDuringAudit('Checkpoint changed across content audit')
    snapshot.checkpoints[str(path)] = before
    return dict(step=proof['step'],bytes=proof['bytes'],model_tensors=len(model_items),state_tensors=tensor_count,
                rng_ranks=8,payload_files=len(payloads),full_payload_sha_refreshed=hashes)


def journal_review(events, run_id, certificates, certificate_digests):
    resets = []; retired_nfs = set(); retired_local = set(); intents = set(); copies = set(); recovered = set()
    for index,row in enumerate(events):
        if (row.get('run_id') != run_id or not isinstance(row.get('unix'),(int,float)) or
                not math.isfinite(row['unix']) or row['unix'] < 0):
            raise ValueError('Retention journal owner/time differs')
        event = row['event']
        if event == 'local_namespace_created_or_lost_history_not_reconstructed':
            resets.append(index); continue
        name = row.get('checkpoint')
        if name not in certificates:
            raise ValueError('Journal references unknown certified checkpoint')
        if event == 'copy_intent':
            if row['bytes'] != certificates[name]['bytes']:
                raise ValueError('Copy journal bytes differ')
            copies.add(name)
        elif event == 'local_published':
            if name not in copies or row['certificate_sha256'] != certificate_digests[name+'.json']:
                raise ValueError('Local publication lacks prior copy intent')
        elif event in ('retire_intent','retire_completed'):
            tier = row['tier']
            if tier not in ('nfs','local'):
                raise ValueError('Journal retirement tier differs')
            if event == 'retire_intent':
                intents.add((tier,name))
            else:
                if (tier,name) not in intents:
                    raise ValueError('Retirement completion lacks durable intent')
                if tier == 'nfs':
                    if name not in copies:
                        raise ValueError('NFS retirement lacks archive copy evidence')
                    retired_nfs.add(name)
                else:
                    if name not in retired_nfs:
                        raise ValueError('Local retirement predates NFS retirement')
                    retired_local.add(name)
        else:
            raise ValueError('Unknown retention journal event')
    # A crash after rename/certificate but before local_published can recover
    # without emitting that event. Do not invent an ACK or reject valid recovery.
    published = {r['checkpoint'] for r in events if r['event']=='local_published'}
    recovered = retired_nfs-published
    last_reset = resets[-1] if len(resets)>1 else -1
    after_reset = {r['checkpoint'] for i,r in enumerate(events) if i>last_reset and r['event']=='retire_completed' and r['tier']=='nfs'}
    return dict(namespace_initializations=len(resets),history_loss_epochs=max(0,len(resets)-1),
                nfs_retired=sorted(retired_nfs),local_retired=sorted(retired_local),
                nfs_retired_after_last_loss=sorted(after_reset),
                recovered_without_publication_event=sorted(recovered),
                scope='Journal sequence and current receipts, not a new reconstruction of already deleted payload bytes or certificate publication timestamps.')


def storage_inventory(root, policy, parent, config, *, hashes=False, final=False, production=True):
    root = Path(root).resolve(); local = Path(policy['local_root']); snap = Snapshot()
    owner = dict(version=1,run_id=policy['run_id'],nfs_run=str(root),local_root=str(local))
    if (policy['nfs_run'] != str(root) or policy['nfs_keep'] != 10 or policy['local_keep'] != 15 or
            policy['resume_from_local'] is not False or local.is_relative_to(root) or root.is_relative_to(local)):
        raise ValueError('Unreviewed10/15 namespace/resume policy')
    if snap.read(root/'.retention_owner.json') != owner:
        raise ValueError('NFS owner receipt differs')
    if final and not hashes:
        raise ValueError('Final audit requires fresh full payload SHA')
    certs = {}; cert_digests = {}
    for p in snap.children(root/'checkpoint_certificates'):
        name = p.stem
        if p.suffix != '.json' or not NAME.fullmatch(name) or not p.is_file():
            raise ValueError('Unexpected checkpoint certificate entry')
        proof = snap.read(p)
        if (proof['status'] != 'passed_main_checkpoint_storage' or proof['step'] != int(name[5:]) or
                proof['training_contract_id'] != parent['id'] or
                proof['checkpoint'] != policy['artifact_training_prefix']+'/'+name):
            raise ValueError('Checkpoint certificate identity differs')
        records(proof); certs[name] = proof; cert_digests[p.name] = snap.files[str(p)][1]
    expected_steps = set(range(config['ckpt_every'],config['max_steps']+1,config['ckpt_every'])) | {config['max_steps']}
    actual_steps = {p['step'] for p in certs.values()}
    if actual_steps-expected_steps or (final and actual_steps != expected_steps):
        raise ValueError('Checkpoint certificate schedule coverage differs')
    events = snap.read(root/'retention_journal.jsonl',lines=True)
    journal = journal_review(events,policy['run_id'],certs,cert_digests)
    nfs = {}; archived = {}; residues = []; overlap = []
    def residue(path,why):
        residues.append(dict(path=artifact_path(path),reason=why,bytes=size_tree(path) if path.is_dir() else path.stat().st_size))
    for p in snap.children(root/'training'):
        if p.name in ('training_contract.json','log.jsonl'):
            continue
        if not p.is_dir() or not NAME.fullmatch(p.name) or not (p/'.complete').is_file() or p.name not in certs:
            residue(p,'incomplete_unverified_or_unexpected_nfs_entry'); continue
        nfs[p.name] = checkpoint_content(p,certs[p.name],parent,config,hashes=hashes,production=production,snapshot=snap)
    local_children = snap.children(local)
    local_missing = not local.exists()
    if not local_missing:
        if root.stat().st_dev == local.stat().st_dev or snap.read(local/'.retention_owner.json') != owner:
            raise ValueError('Local owner/device differs')
        allowed = {'.retention_owner.json','checkpoints','certificates','staging','retention_trash'}
        for p in local_children:
            if p.name not in allowed:
                residue(p,'unexpected_local_namespace_entry')
        if any(not (local/name).is_dir() for name in allowed-{'.retention_owner.json'}):
            raise ValueError('Partially missing local namespace')
        for p in snap.children(local/'checkpoints'):
            receipt = local/'certificates'/(p.name+'.json')
            if not p.is_dir() or not NAME.fullmatch(p.name) or p.name not in certs or not receipt.is_file() or not (p/'.complete').is_file():
                residue(p,'incomplete_unverified_local_copy'); continue
            if snap.read(receipt) != certs[p.name]:
                raise ValueError('Local ACK differs from durable NFS certificate')
            archived[p.name] = checkpoint_content(p,certs[p.name],parent,config,hashes=hashes,production=production,snapshot=snap)
        for p in snap.children(local/'certificates'):
            if p.name not in {name+'.json' for name in archived}:
                residue(p,'orphan_or_unverified_local_certificate')
        for name in ['staging','retention_trash']:
            for p in snap.children(local/name):
                residue(p,'local_'+name+'_residue')
    for p in snap.children(root/'retention_trash'):
        residue(p,'nfs_retirement_residue')
    ordered = sorted(certs); expected_nfs = ordered[-10:]; older = ordered[:-10]
    expected_local = older[-15:]
    if journal['history_loss_epochs']:
        expected_local = sorted(set(older)&set(journal['nfs_retired_after_last_loss']))[-15:]
    overlap = sorted(set(nfs)&set(archived))
    issues = []
    if sorted(nfs) != expected_nfs:
        issues.append('NFS is not the latest certified10 (or all if fewer)')
    if not local_missing and sorted(archived) != expected_local:
        issues.append('Local historical suffix differs from archives since the last recorded namespace loss')
    if overlap:
        issues.append('Checkpoint duplicates remain across tiers')
    if residues:
        issues.append('Unfinished or unverified retention residues exist')
    if len(nfs)>10 or len(archived)>15 or len(set(nfs)|set(archived))>25:
        issues.append('Normal retention count exceeds policy')
    if final and set(older) != set(journal['nfs_retired']):
        issues.append('NFS retirement receipts differ from removed older checkpoints')
    if final and set(archived)&set(journal['local_retired']):
        issues.append('A retired local checkpoint is still counted as retained')
    manager = snap.read(root/'retention_state.json')
    for key,value in [('nfs_checkpoints',sorted(nfs)),('local_checkpoints',sorted(archived)),
                      ('distinct_checkpoints',len(set(nfs)|set(archived))),('uncertified_nfs',[])]:
        if manager.get(key) != value and not (local_missing and key in ('local_checkpoints','distinct_checkpoints')):
            issues.append('Manager state differs from observed '+key)
    temp = Path(policy['temp_root'])
    local_device_path = existing_ancestor(local)
    if root.stat().st_dev == local_device_path.stat().st_dev:
        issues.append('NFS and local history are not separate filesystems')
    usage = dict(nfs_used_bytes=size_tree(root),local_used_bytes=size_tree(local),temporary_used_bytes=size_tree(temp),
                 nfs_free_bytes=shutil.disk_usage(root).free,local_free_bytes=shutil.disk_usage(local_device_path).free)
    if (usage['nfs_used_bytes']>policy['nfs_budget_bytes'] or usage['local_used_bytes']>policy['local_budget_bytes'] or
            usage['temporary_used_bytes']>policy['temporary_budget_bytes'] or usage['nfs_free_bytes']<policy['reserve_bytes']+policy['other_nfs_future_bytes'] or
            usage['local_free_bytes']<policy['reserve_bytes']+max(0,policy['temporary_budget_bytes']-usage['temporary_used_bytes'])):
        issues.append('Storage budget or safety/temporary reserve violated')
    snap.unchanged()
    if final and issues:
        raise ValueError('Final retention audit: '+'; '.join(issues))
    return dict(status=('passed_final_retention_content' if production else 'passed_fixture_retention_content') if final else 'observed_retention_snapshot',
                production=production,final=final,full_payload_sha_refreshed=hashes,nfs=nfs,local=archived,
                nfs_resume_candidates=sorted(nfs),expected_nfs=expected_nfs,expected_local=expected_local,
                distinct_checkpoints=len(set(nfs)|set(archived)),overlap=overlap,residues=residues,issues=issues,
                local_namespace_missing=local_missing,local_history_optional_loss=local_missing or bool(journal['history_loss_epochs']),
                certificate_receipts=cert_digests,journal=journal,usage=usage,
                resident_file_stats=snap.checkpoints,
                metadata_evidence={artifact_path(Path(p)):v[1] for p,v in snap.files.items()},
                scope='Read-only physical resident state and durable provenance audit. NFS-only resume, optional local history loss reported; no repair/deletion. Snapshot is not final acceptance.')


def audit_run(source, *, final=False):
    source = resolve(source)
    from scripts.supervise_main_training import require_authorization
    binding_files = [source/'contract.json',source/'main.json',source/'training/training_contract.json']
    before = {str(p):stat_record(p) for p in binding_files}
    c = json.loads((source/'contract.json').read_text()); require_authorization(c)
    for path,digest in {**c['code'],**c['evidence']}.items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Approved main binding changed: '+path)
    config = json.loads((source/'main.json').read_text())
    if c['configurations']['main.json'] != sha256_file(source/'main.json') or config['model']!='3b' or config['max_steps']!=57000:
        raise ValueError('Unreviewed final retention target')
    parent = json.loads((source/'training/training_contract.json').read_text())
    training_audit = None
    if final:
        from scripts.evaluate_final_main import completed_main
        _,_,_,training_audit = completed_main(source)
        completion_files = [source/'report.json',source/'.done']
        binding_files.extend(completion_files)
        before.update({str(p):stat_record(p) for p in completion_files})
    result = storage_inventory(source,c['tiered_retention'],parent,config,hashes=final,final=final,production=True)
    if final:
        report = json.loads((source/'report.json').read_text())
        if report['certificate_receipts'] != result['certificate_receipts']:
            raise ValueError('Final supervisor certificate list differs')
        if 'ckpt_0057000' not in result['nfs_resume_candidates']:
            raise ValueError('Final checkpoint is not a certified NFS resume candidate')
    if {str(p):stat_record(p) for p in binding_files} != before:
        raise ChangedDuringAudit('Main contract/config/completion changed during audit')
    for path,digest in {**c['code'],**c['evidence']}.items():
        if sha256_file(resolve(path)) != digest:
            raise ChangedDuringAudit('Main implementation/evidence changed during audit')
    result.update(utc=datetime.now(timezone.utc).isoformat(),source_run=artifact_path(source),training_audit=training_audit,
                  source_contract_sha256=sha256_file(source/'contract.json'),audit_code_sha256=sha256_file(Path(__file__)))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',default='runs/3b_full_v1');parser.add_argument('--output',required=True)
    parser.add_argument('--final',action='store_true');args=parser.parse_args()
    output=resolve(args.output)
    if output.exists() or output.is_relative_to(resolve(args.source_run)):
        raise ValueError('Use a new independent audit output directory')
    result=audit_run(args.source_run,final=args.final)
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'report.json',result)
    if args.final:
        atomic_json(output/'.done',dict(report_sha256=sha256_file(output/'report.json')))
