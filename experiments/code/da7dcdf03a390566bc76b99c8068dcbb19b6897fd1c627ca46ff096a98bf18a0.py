"""Build release offsets as independently verified assignment shards appear."""
import hashlib,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import pyarrow as pa
from tokenizers import Tokenizer
from data.release_offsets import export_offsets
from data.release_provenance import sha


def job(args):
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    try:return export_offsets(*args[:5],eos=args[5],tokenizer_sha=args[6])
    except ValueError as e:
        if str(e)=='EOS/document count differs; literal special tokens require replay':
            return {'status':'needs_literal_eos_replay','assignment':str(args[0]),'no_token_payload_changed':True}
        raise


if __name__=='__main__':
    os.sched_setaffinity(0,[84,85,86,87]);start=time.monotonic();base=root/'runs/dataset_public_release_v1'
    m=json.loads((root/'data/datasets/final_v2.split/manifest.json').read_text())
    tok=root/'tokenizer/candidate_v1/tokenizer.json';th=sha(tok);eos=Tokenizer.from_file(str(tok)).token_to_id('<|eos|>')
    out=base/'token_offsets_v1';out.mkdir(exist_ok=True)
    # Reuse the fully checked wiki pilot; immutable files, SHA checked again.
    for p in (base/'token_offsets_pilot/wiki_ko').glob('*'):
        if p.suffix not in ('.json','.parquet'):continue
        target=out/'wiki_ko'/p.name;target.parent.mkdir(exist_ok=True)
        if not target.exists():os.link(p,target)
    todo=list(m['contract']['files']);results=[];ambiguous=[];pending={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        while todo or pending:
            for r in list(todo):
                source=r['source'];stem=Path(r['path']).stem;dest=out/source/(stem+'.parquet');marker=dest.with_suffix('.json')
                if marker.exists():
                    item=json.loads(marker.read_text());assert sha(dest)==item['sha256'];results.append(item);todo.remove(r);continue
                assignment=base/'split_full_v1'/source/(stem+'.parquet')
                if len(pending)>=8:break
                if not assignment.with_suffix('.json').exists():continue
                token_root=root/'data/tokens/final_v1'/source
                args=(assignment,token_root/(stem+'.bin'),token_root/(stem+'.hold.bin'),token_root/(stem+'.bin.done'),dest,eos,th)
                pending[pool.submit(job,args)]=r;todo.remove(r)
            if pending:
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for f in done:
                    r=pending.pop(f);item=f.result()
                    if item['status']=='needs_literal_eos_replay':
                        ambiguous.append({'source':r['source'],'original_final_file':r['path'],**item})
                        (out/'ambiguous_files.json').write_text(json.dumps(ambiguous,indent=2)+'\n')
                    else:results.append(item)
                n=len(results)+len(ambiguous)
                if done and (n%25==0 or not todo and not pending):
                    print(json.dumps({'files_checked':n,'target':len(m['contract']['files']),'offset_files_verified':len(results),'ambiguous_files':len(ambiguous)}),flush=True)
            elif todo:time.sleep(5)
            if time.monotonic()-start>1200:raise TimeoutError('Stop at budget; verified per-file outputs remain resumable')
    report={'status':'all_offsets_verified' if not ambiguous else 'literal_eos_replay_required',
        'verified_files':len(results),'ambiguous_files':ambiguous,'rows_with_verified_offsets':sum(i['rows'] for i in results),
        'tokens_verified_for_offset_files':sum(p['tokens'] for i in results for p in i['files'].values()),
        'offset_bytes':sum(i['bytes'] for i in results),'elapsed_seconds':time.monotonic()-start,
        'tokenizer_sha256':th,'eos_id':eos,'bin_payloads_changed':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
