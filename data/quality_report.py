"""Read-only candidate readiness assessment. Never edits data or promotes a manifest.
Reports reconcile receipts/footer counts, not full data integrity. Content auditing is
an explicit separate condition. Active dedup SQLite databases are never opened.
"""
import json
import math
from pathlib import Path
import pyarrow.parquet as pq
from common.data_contracts import resolve,artifact_path,identity,sha256_file,atomic_json

POLICY='nfc-sha256-case-and-whitespace-preserving-v1'
DECISIONS=('near_duplicate_policy','code_cross_policy','holdout_policy','tokenizer_byte_mix')


def read_json(path):
    def unique(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate JSON key: '+k)
            d[k]=v
        return d
    return json.loads(Path(path).read_text(),object_pairs_hook=unique)


def integers(counts):
    if not isinstance(counts,dict) or any(type(v) is not int or v<0 for v in counts.values()):
        raise ValueError('Counts must be nonnegative integers')


def assess(manifest,dedup_root,policy_path=None):
    manifest=resolve(manifest);root=resolve(dedup_root);inventory=read_json(manifest)
    if inventory.get('schema_version')!=1 or inventory.get('status') not in ('draft','ready') or not inventory.get('sources'):
        raise ValueError('Require schema-1 candidate source inventory')
    snapshot=identity(inventory);receipts={};checks={};sources={};seen=set();expected_web={}
    listed_paths={artifact_path(resolve(p)) for e in inventory['sources'].values() for p in e['files']}
    def receipt(path):
        path=Path(path);r=read_json(path);receipts[artifact_path(path)]=sha256_file(path);return r
    def condition(name,state,reason):checks[name]=dict(status=state,reason=reason)
    complete_at_start=(root/'.done').is_file()
    dedup=receipt(root/'manifest.json')
    if dedup.get('policy')!=POLICY:raise ValueError('Dedup policy mismatch')
    actual_web={}
    for f in dedup['inputs']:
        key=(f['source'],Path(f['path']).name)
        if key in actual_web:raise ValueError('Duplicate dedup input')
        actual_web[key]=artifact_path(root/'clean'/f['source']/key[1])
    for name,entry in inventory['sources'].items():
        files=[resolve(f) for f in entry['files']]
        if not files:raise ValueError('Empty source file list')
        if len({f.stem for f in files})!=len(files):raise ValueError('Repeated source basename')
        code=entry.get('text_policy')=='raw-utf8-code-v1'
        if not code and entry.get('text_policy')!='nfc-web-exact-v1':raise ValueError('Unsupported source text policy')
        if not entry.get('origin',{}).get('revision'):raise ValueError('Source revision missing')
        origin=entry['origin']
        if origin.get('raw_manifest'):
            raw=receipt(resolve(origin['raw_manifest']))
            if any(raw.get(k)!=origin.get(k) for k in ['repo','revision']):raise ValueError('Source origin revision/repo mismatch')
            if {Path(p).name for p in raw['files']}!={p.name for p in files}:raise ValueError('Origin/source file coverage mismatch')
        totals=dict(input_docs=0,kept=0,exact_duplicates=0);completed=0;details=[]
        for file in files:
            key=artifact_path(file)
            if key in seen:raise ValueError('Repeated source file')
            seen.add(key)
            if not code:expected_web[(name,file.name)]=key
            marker=Path(str(file)+'.done')
            if not marker.exists():
                details.append(dict(path=key,state='pending'));continue
            if not file.is_file():raise ValueError('Committed output missing: '+key)
            mark=receipt(marker)
            if code:
                stats=receipt(Path(str(file)+'.stats.json'));counts=stats['counts'];integers(counts)
                if 'kept' not in counts or not mark.get('policy_hash'):raise ValueError('Code receipt incomplete')
                stats_counts=dict(input_docs=sum(counts.values()),kept=counts['kept'],exact_duplicates=counts.get('exact_duplicate',0))
            else:
                stats_counts=mark;integers(stats_counts)
                if set(stats_counts)!=set(totals) or mark['input_docs']!=mark['kept']+mark['exact_duplicates']:
                    raise ValueError('Dedup input/kept/duplicate count mismatch')
            rows=pq.ParquetFile(file).metadata.num_rows
            if rows!=stats_counts['kept']:raise ValueError('Output footer/receipt count mismatch')
            for k in totals:totals[k]+=stats_counts[k]
            completed+=1;details.append(dict(path=key,state='receipt_verified',rows=rows,bytes=file.stat().st_size,
                policy_hash=mark.get('policy_hash') if code else identity(dedup['policy'])))
        # Completed unlisted files invalidate the inventory; incomplete writer files are ignored.
        for parent in set(f.parent for f in files):
            for extra in parent.glob('*.parquet.done'):
                if artifact_path(Path(str(extra)[:-5])) not in listed_paths:
                    raise ValueError('Unexpected completed source file: '+str(extra))
        summary=dict(expected_shards=len(files),completed_shards=completed,counts=totals,files=details,kind='code' if code else 'web')
        sources[name]=summary
        if code:
            complete=resolve(entry['completion_marker'])
            if complete.exists():
                end=receipt(complete);aggregate=receipt(complete.parent/'summary.json')
                if completed!=len(files) or end.get('shards')!=len(files):raise ValueError('Code completion coverage mismatch')
                if any(d.get('policy_hash')!=end.get('policy_hash') for d in details):raise ValueError('Code policy mismatch')
                if aggregate.get('policy_hash')!=end.get('policy_hash') or aggregate['counts']['kept']!=totals['kept']:
                    raise ValueError('Code aggregate mismatch')
                condition('code_clean:'+name,'passed','Completed receipts, policy and footer counts reconcile; content audit separate')
            else:condition('code_clean:'+name,'pending','Source completion marker absent')
    if expected_web!=actual_web:raise ValueError('Candidate files differ from dedup input/order/source mapping')
    web_totals={k:sum(s['counts'][k] for s in sources.values() if s['kind']=='web') for k in ['input_docs','kept','exact_duplicates']}
    complete=root/'.done'
    if complete_at_start:
        end=receipt(complete);summary=receipt(root/'summary.json')
        if any(s['completed_shards']!=s['expected_shards'] for s in sources.values() if s['kind']=='web') or end!=web_totals or summary!=web_totals:
            raise ValueError('Dedup aggregate/completion mismatch')
        condition('exact_completion','passed','All listed shard receipts and aggregate counts reconcile; content audit separate')
    else:condition('exact_completion','pending','Dedup is still running; no global completion receipt')
    policy=receipt(resolve(policy_path)) if policy_path else None
    if policy is not None and (policy.get('schema_version')!=1 or policy.get('source_snapshot_id')!=snapshot):
        raise ValueError('Quality policy source snapshot mismatch')
    declarations=policy.get('checks',{}) if policy else {}
    if set(declarations)-set(DECISIONS)-{'content_audit'}:raise ValueError('Unknown quality condition')
    for name in (*DECISIONS,'content_audit'):
        entry=declarations.get(name)
        if entry is None:condition(name,'pending','No versioned evidence/decision supplied');continue
        state=entry.get('status');reason=entry.get('reason');proofs=entry.get('evidence',[])
        if state not in ('passed','pending','failed') or not isinstance(reason,str) or not reason.strip():raise ValueError('Invalid quality declaration')
        loaded=[]
        for proof in proofs:
            p=resolve(proof['path'])
            if sha256_file(p)!=proof['sha256']:raise ValueError('Evidence content changed')
            loaded.append(receipt(p))
        if state=='passed':
            if not loaded:raise ValueError('Passed condition requires evidence')
            if name=='content_audit':
                if len(loaded)!=1 or loaded[0].get('structural_status')!='passed':
                    raise ValueError('Content audit does not cover this candidate')
                if loaded[0].get('format')=='source-audit-v1':
                    from common.source_inventory import load_audit,definition
                    checked=load_audit(resolve(proofs[0]['path']))
                    if checked['definition']!=definition(inventory):raise ValueError('Content audit source definition differs')
                elif loaded[0].get('manifest_sha256')!=sha256_file(manifest):
                    raise ValueError('Content audit does not cover this candidate')
                audited=loaded[0].get('sources',{})
                if set(audited)!=set(inventory['sources']):raise ValueError('Content audit sources differ')
                for source,entry_source in inventory['sources'].items():
                    if [f['path'] for f in audited[source]['files']]!=[artifact_path(resolve(p)) for p in entry_source['files']]:raise ValueError('Content audit file coverage differs')
                    if audited[source]['rows']!=sources[source]['counts']['kept']:raise ValueError('Content audit row count differs')
            else:
                decision=entry.get('decision',{})
                if not isinstance(decision.get('scope'),str) or not decision['scope'].strip() or decision.get('action') not in ('keep','adjust_weights','group_split','byte_mix'):
                    raise ValueError('Policy pass requires an explicit scoped decision')
                if name=='holdout_policy' and (decision['action']!='group_split' or type(decision.get('seed')) is not int or decision['seed']<0
                        or type(decision.get('eval_fraction')) not in (int,float) or not 0<decision['eval_fraction']<1 or decision.get('code_repository_grouping') is not True):
                    raise ValueError('Holdout policy requires group split, seed, fraction and repository grouping')
                if name=='tokenizer_byte_mix':
                    weights=decision.get('weights',{})
                    if decision['action']!='byte_mix' or not isinstance(weights,dict) or set(weights)!=set(inventory['sources']) or any(type(w) not in (int,float) or not math.isfinite(w) or w<=0 for w in weights.values()) or decision.get('normalization')!='raw':
                        raise ValueError('Tokenizer policy requires complete positive byte weights and raw normalization')
                # This is an auditable policy decision, not proof of universal no-overlap.
                checks[name]=dict(status=state,reason=reason,decision=decision,evidence_level='declared_policy_with_content_pinned_evidence');continue
        condition(name,state,reason)
    states={c['status'] for c in checks.values()};readiness='failed' if 'failed' in states else 'pending' if 'pending' in states else 'passed'
    contract=dict(version=1,manifest_sha256=sha256_file(manifest),source_snapshot_id=snapshot,
                  receipts=receipts,code_sha256=sha256_file(Path(__file__)),sources=sources)
    return dict(run_status='completed',readiness=readiness,contract=contract,contract_id=identity(contract),checks=checks,
                source_counts={s:{k:r[k] for k in ['expected_shards','completed_shards','counts']} for s,r in sources.items()},
                note='Receipt/footer reconciliation and scoped policy decisions. No manifest promotion; full content integrity requires separate audit.')


def save_report(report,output):
    output=resolve(output);marker=Path(str(output)+'.done')
    if output.exists() and read_json(output)!=report:raise ValueError('Report differs; use a new output path')
    if not output.exists():atomic_json(output,report)
    # The certificate proves report execution/content only, never readiness=passed.
    atomic_json(marker,dict(contract_id=report['contract_id'],sha256=sha256_file(output),readiness=report['readiness']))
