"""Offline source-only end-to-end demo. Generates synthetic text; never reads corpus data."""
import argparse, copy, json, os, random, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from common.data_contracts import atomic_json,identity,sha256_file
from data import clean
from data.dedup import deduplicate
from data.quality_report import POLICY,DECISIONS,assess
from data.source_audit import run as audit
from scripts.finalize_source_manifest import finalize
from data.split_manifest import build
from data.split_inputs import prepare
from tokenizer.train_tokenizer import train_prepared
from tokenizers import Tokenizer
from data.tokenize_shards import tokenize_source
from data.build_mix import build_mix
from train.data import MixSampler,HoldoutSet
from model.model import Transformer,get_config


def execute(root):
    torch.set_num_threads(2);torch.manual_seed(42)
    rng=random.Random(19)
    # Each synthetic document has unique word sequences so filtering is meaningful.
    vocab=['river','forest','planet','garden','window','science','music','cloud','paper','mountain','ocean','bright','silver','amber','travel','gentle','measure','learning','language','number']
    texts=['Document '+str(i)+'. '+' '.join(rng.choices(vocab,k=90))+'.' for i in range(120)]
    raw=root/'raw';raw.mkdir()
    clean.ROOT=str(root)
    cleaned=[]
    for n,rows in enumerate([texts[:70]+[texts[0],'too short'],texts[60:]+[texts[60]]]):
        p=raw/f'part{n}.parquet';pq.write_table(pa.table({'text':rows}),p)
        clean.work(('synthetic','en',str(p)));cleaned.append(('synthetic',root/'clean/synthetic'/p.name))
    counts=deduplicate(cleaned,root/'dedup',POLICY)
    assert counts['kept']==120 and counts['exact_duplicates']==10,counts
    files=[root/'dedup/clean/synthetic'/p.name for _,p in cleaned]
    candidate=root/'candidate.json'
    atomic_json(candidate,dict(schema_version=1,status='draft',sources={'synthetic':dict(files=list(map(str,files)),text_policy='nfc-web-exact-v1',completion_marker=str(root/'dedup/.done'),origin={'revision':'generated-fixture-v1'})}))
    report=audit(candidate,root/'audit',min_free_gib=0,max_rss_gib=2)
    proof=dict(path=str(root/'audit/report.json'),sha256=sha256_file(root/'audit/report.json'))
    checks={k:dict(status='passed',reason='Generated fixture only; no external corpus or benchmark is present',decision={'action':'keep','scope':'synthetic demonstration only'},evidence=[proof]) for k in DECISIONS}
    checks['holdout_policy']['decision'].update(action='group_split',seed=7,eval_fraction=.2,code_repository_grouping=True)
    checks['tokenizer_byte_mix']['decision'].update(action='byte_mix',weights={'synthetic':1},normalization='raw')
    checks['content_audit']=dict(status='passed',reason='All generated input decoded and hashed',evidence=[proof])
    atomic_json(root/'policy.json',dict(schema_version=1,source_snapshot_id=identity(json.loads(candidate.read_text())),checks=checks))
    quality=assess(candidate,root/'dedup',root/'policy.json');assert quality['readiness']=='passed'
    ready=finalize(candidate,root/'audit/report.json',root/'policy.json',root/'dedup',root/'ready.json')
    split=build(root/'ready.json',root/'split',.2,7,sqlite_cache_mib=16)
    assert split['counts']['train']>0 and split['counts']['eval']>0
    prepare(root/'ready.json',root/'split/manifest.json',root/'tokenizer_input',{'synthetic':10000},mode='train',seed=7)
    train_prepared(root/'tokenizer_input/manifest.json',root/'tokenizer.json',vocab=272)
    tok=Tokenizer.from_file(str(root/'tokenizer.json'))
    totals=tokenize_source('synthetic',root/'tokenizer.json',files,root/'tokens/synthetic',snapshot=identity(ready),split_manifest=root/'split/manifest.json',batch_size=16)
    mix=build_mix({'synthetic':1},root/'tokens',root/'mix.json',seq_len=32)
    sampler=MixSampler(str(root/'mix.json'),32,seed=42)
    hold=HoldoutSet(mix['holdout'],32,n_windows=4)
    config=get_config('micro',vocab_size=tok.get_vocab_size(),max_seq_len=32)
    model=Transformer(config);opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    def update(model,opt,step):
        model.train();x,y=sampler.batch(step,0,0,2);loss=model(x,y);assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True);return loss.item()
    losses=[update(model,opt,i) for i in range(3)]
    checkpoint=root/'tiny_checkpoint.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),step=3),checkpoint)
    expected=update(model,opt,3)
    resumed=Transformer(config);resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=1e-3)
    saved=torch.load(checkpoint,weights_only=True);resumed.load_state_dict(saved['model']);resumed_opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    actual=update(resumed,resumed_opt,3)
    assert expected==actual
    for k,v in model.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]),k
    a,b=opt.state_dict(),resumed_opt.state_dict()
    assert a['param_groups']==b['param_groups']
    for k,values in a['state'].items():
        for field,value in values.items():
            other=b['state'][k][field]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    model.eval()
    with torch.no_grad():
        # Use holdout memmaps directly; every window is from the actual held-out shard.
        assert len(hold)>0
        eval_loss=model(hold.wins[:,:-1].contiguous(),hold.wins[:,1:].contiguous()).item()
    assert np.isfinite(eval_loss)
    result=dict(status='passed',synthetic_only=True,dedup=counts,split=split['counts'],tokens=totals,vocab_size=tok.get_vocab_size(),parameters=model.num_params(),training_losses=losses,next_step_loss=actual,holdout_loss=eval_loss,resume_model_optimizer_exact=True,scope='Offline CPU FP32 micro model; same model/sampler/data modules, simple torch.save demo checkpoint. Full FSDP/DCP and GPU are covered separately, not rerun here.')
    atomic_json(root/'result.json',result);print(json.dumps(result,indent=2));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',help='New output directory inside this checkout');a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];(repo/'runs').mkdir(exist_ok=True)
    if a.output_dir:
        p=Path(a.output_dir).resolve()
        if not p.is_relative_to(repo):ap.error('--output-dir must be inside this checkout for portable mix paths')
        p.mkdir(parents=True,exist_ok=False);execute(p)
    else:
        with tempfile.TemporaryDirectory(prefix='dense-quickstart-',dir=repo/'runs') as tmp:execute(Path(tmp))
if __name__=='__main__':main()
