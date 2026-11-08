"""Native checkpoint PPL and fixed-prompt generation with reproducible JSON reports.
Inputs are JSONL {category:ko|en,text}; prompts also require a unique id.
Default execution is CPU. Raw-text evaluation inputs must come from the chosen holdout snapshot.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed.checkpoint as dcp
from tokenizers import Tokenizer
from common.data_contracts import resolve,identity,sha256_file,atomic_json
from model.model import Transformer,get_config
from eval.perplexity import evaluate_sequences,NLLAccumulator
from eval.generation import generate


def records(path, prompts=False):
    seen=set();found=set();result=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line)
        if r.get('category') not in ('ko','en') or not isinstance(r.get('text'),str) or not r['text']:
            raise ValueError('Require nonempty ko/en text records')
        if prompts:
            if not isinstance(r.get('id'),str) or not r['id'] or r['id'] in seen:
                raise ValueError('Prompt IDs must be nonempty and unique')
            seen.add(r['id'])
        found.add(r['category']);result.append(r)
    if found != {'ko','en'}:raise ValueError('Both Korean and English inputs are required')
    return result


def load_checkpoint(directory, tokenizer_sha256, device='cpu'):
    directory=Path(directory)
    if not (directory/'.complete').is_file():raise ValueError('Checkpoint is incomplete')
    meta=json.loads((directory/'meta.json').read_text());c=meta['config']
    expected=meta.get('training_contract',{}).get('mix',{}).get('tokenizer_sha256')
    if expected!=tokenizer_sha256:raise ValueError('Checkpoint/tokenizer identity mismatch')
    cfg=get_config(c['model'],vocab_size=c['vocab_size'],max_seq_len=c['seq_len'])
    model=Transformer(cfg)
    state=model.state_dict();reader=dcp.FileSystemReader(directory)
    stored=reader.read_metadata().state_dict_metadata
    expected_keys={'model.'+k for k in state}
    if {k for k in stored if k.startswith('model.')}!=expected_keys:
        raise ValueError('Native model checkpoint keys differ')
    for key,value in state.items():
        if tuple(stored['model.'+key].size)!=tuple(value.shape):
            raise ValueError(f'Native model checkpoint shape mismatch: {key}')
    dcp.load({'model':state},storage_reader=reader)
    model.load_state_dict(state,strict=True)
    return model.to(device),meta


def run(checkpoint,tokenizer_path,samples_path,prompts_path,output,*,seq_len=32,stride=16,
        max_new_tokens=16,seed=0,temperature=0.0,device='cpu',holdout_manifest=None,
        reader_manifest=None,reader_cache_mib=None,hash_workers=32,verified_inputs=None):
    if holdout_manifest is None and (reader_manifest is not None or verified_inputs is not None):raise ValueError('Verified reader requires a holdout manifest')
    if reader_manifest is not None and verified_inputs is not None:raise ValueError('Choose a reader manifest or an existing live context')
    if reader_cache_mib is not None and reader_manifest is None:raise ValueError('Reader cache requires a reader manifest')
    checkpoint,tokenizer_path,samples_path,prompts_path,output=map(Path,(checkpoint,tokenizer_path,samples_path,prompts_path,output))
    if not (checkpoint/'.complete').is_file():raise ValueError('Checkpoint is incomplete')
    samples=records(samples_path);prompts=records(prompts_path,True)
    provenance=dict(level='manual-input; split membership not verified')
    if holdout_manifest is not None:
        from data.split_inputs import verify_prepared
        if reader_manifest is not None:
            from eval.prepare_holdout import local_verified_inputs
            metadata=json.loads(resolve(holdout_manifest).read_text())['contract']
            with local_verified_inputs(metadata['source_manifest'],metadata['split_manifest'],reader_manifest,
                                       cache_mib=reader_cache_mib,hash_workers=hash_workers) as inputs:
                prepared=verify_prepared(holdout_manifest,'eval',samples_path,verified_inputs=inputs)
            # Close the large reader cache before allocating the evaluation model.
        else:prepared=verify_prepared(holdout_manifest,'eval',samples_path,verified_inputs=verified_inputs)
        provenance=dict(level='verified-eval-split',manifest_sha256=sha256_file(resolve(holdout_manifest)),
                        prepared_contract_id=prepared['contract_id'],split_contract_id=prepared['contract']['split_contract_id'])
    token_sha=sha256_file(tokenizer_path)
    contract=dict(version=2,input_provenance=provenance,checkpoint={str(p.relative_to(checkpoint)):sha256_file(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()},
                  tokenizer_sha256=token_sha,samples_sha256=sha256_file(samples_path),prompts_sha256=sha256_file(prompts_path),
                  seq_len=seq_len,stride=stride,max_new_tokens=max_new_tokens,seed=seed,temperature=temperature,
                  device=device,torch_version=str(torch.__version__),template='base-plain-text-v1',append_eos_for_ppl=True,
                  code={name:sha256_file(resolve(name)) for name in ['eval/run.py','eval/prepare_holdout.py','eval/perplexity.py','eval/generation.py','model/model.py',
                        'data/split_inputs.py','data/verified_split_context.py','data/local_split_reader.py','data/split_manifest.py','common/source_inventory.py']})
    contract_id=identity(contract);marker=Path(str(output)+'.done')
    if output.exists():
        previous=json.loads(output.read_text())
        if previous.get('contract_id')!=contract_id:raise ValueError('Evaluation identity changed; use a new output path')
        if not marker.exists() or json.loads(marker.read_text())!={'contract_id':contract_id,'sha256':sha256_file(output)}:
            raise ValueError('Existing evaluation report is uncertified')
        return previous
    tok=Tokenizer.from_file(str(tokenizer_path));eos=tok.token_to_id('<|eos|>')
    model,meta=load_checkpoint(checkpoint,token_sha,device)
    if eos is None or max(tok.get_vocab().values())>=model.c.vocab_size:
        raise ValueError('Tokenizer vocabulary/EOS incompatible with model')
    if not 1<=stride<=seq_len<=model.c.max_seq_len:raise ValueError('Invalid evaluation context/stride')
    per_language={};aggregate=NLLAccumulator()
    for category in ['ko','en']:
        sequences=[tok.encode(r['text'],add_special_tokens=False).ids+[eos] for r in samples if r['category']==category]
        result=evaluate_sequences(model,sequences,seq_len,stride,device)
        per_language[category]=result;aggregate.add(result['nll_sum'],result['valid_tokens'])
    generations=[]
    for r in prompts:
        result=generate(model,tok.encode(r['text'],add_special_tokens=False).ids,eos_id=eos,
                        context_length=seq_len,max_new_tokens=max_new_tokens,seed=seed,temperature=temperature,device=device)
        generations.append(dict(id=r['id'],category=r['category'],prompt=r['text'],
                          generated_text=tok.decode(result['generated_ids'],skip_special_tokens=False),**result))
    report=dict(status='passed',contract_id=contract_id,contract=contract,checkpoint_step=meta['step'],
                languages=per_language,aggregate=aggregate.result(),generations=generations,
                scope='Native base-model PPL and fixed prompts on supplied inputs; not benchmark-task or chat evaluation')
    atomic_json(output,report);atomic_json(marker,dict(contract_id=contract_id,sha256=sha256_file(output)))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ['checkpoint','tokenizer','samples','prompts','output']:ap.add_argument('--'+field,required=True)
    ap.add_argument('--seq-len',type=int,default=32);ap.add_argument('--stride',type=int,default=16)
    ap.add_argument('--max-new-tokens',type=int,default=16);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--temperature',type=float,default=0.0);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--holdout-manifest',help='Certified eval-split input manifest from prepare_holdout.py')
    ap.add_argument('--reader-manifest',help='Certified local split reader for fresh holdout validation')
    ap.add_argument('--sqlite-cache-mib',type=int);ap.add_argument('--hash-workers',type=int,default=32)
    a=ap.parse_args()
    result=run(*[resolve(getattr(a,f)) for f in ['checkpoint','tokenizer','samples','prompts','output']],
               seq_len=a.seq_len,stride=a.stride,max_new_tokens=a.max_new_tokens,seed=a.seed,temperature=a.temperature,device=a.device,
               holdout_manifest=a.holdout_manifest,reader_manifest=a.reader_manifest,reader_cache_mib=a.sqlite_cache_mib,hash_workers=a.hash_workers)
    print(json.dumps(dict(status=result['status'],languages=result['languages']),ensure_ascii=False))


if __name__=='__main__':main()
