"""Train a candidate byte-level BPE from an explicit, completed source snapshot.
Usage: python tokenizer/train_tokenizer.py --sources-manifest PATH --mix SRC:WEIGHT ...
       [--mb 3000] [--output tokenizer/candidate/tokenizer.json]
Weights describe sample UTF8 bytes, not pretraining token weights. No missing-source fallback.
"""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import source_files, resolve, identity, atomic_json, sha256_file, artifact_path
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

SPECIAL = ['<|bos|>', '<|eos|>', '<|pad|>', '<|im_start|>', '<|im_end|>'] + [f'<|reserved_{i}|>' for i in range(11)]


def make_tokenizer(normalization='raw'):
    if normalization not in ('raw', 'NFC'):
        raise ValueError('Unsupported normalization policy')
    tok = Tokenizer(models.BPE())
    if normalization == 'NFC':
        tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tok.decoder = decoders.ByteLevel()
    return tok


def sample_texts(files, budget_bytes, seed=0, stats=None):
    files=list(files);random.Random(seed).shuffle(files)
    per_file=max(1,budget_bytes//max(1,len(files)));got=docs=0
    for file in files:
        p=pq.ParquetFile(file);fb=0
        for rg in random.Random(seed).sample(range(p.num_row_groups),p.num_row_groups):
            for batch in p.iter_batches(batch_size=1024,row_groups=[rg],columns=['text']):
                for text in batch.column(0).to_pylist():
                    text=text[:20000];b=len(text.encode())
                    if not b:continue
                    yield text
                    fb+=b;got+=b;docs+=1
                    if fb>=per_file or got>=budget_bytes:break
                if fb>=per_file or got>=budget_bytes:break
            if fb>=per_file or got>=budget_bytes:
                break
        if got>=budget_bytes:
            break
    if stats is not None:
        stats.update(utf8_bytes=got,documents=docs)


def train_prepared(manifest,output,vocab=65536,normalization='raw'):
    from data.split_inputs import verify_prepared
    prepared=verify_prepared(manifest,'train');output=resolve(output);marker=Path(str(output)+'.done')
    if type(vocab) is not int or not 272<=vocab<=65536:raise ValueError('Vocabulary must be 272..65536')
    config=dict(version=2,vocab=vocab,normalization=normalization,special_tokens=SPECIAL,
        prepared_manifest_sha256=sha256_file(resolve(manifest)),prepared_contract_id=prepared['contract_id'],
        split_contract_id=prepared['contract']['split_contract_id'],
        code={p:sha256_file(resolve(p)) for p in ['tokenizer/train_tokenizer.py','data/split_inputs.py']})
    if output.exists():
        if marker.exists():
            prior=json.loads(marker.read_text())
            if prior.get('config_id')==identity(config) and prior.get('tokenizer_sha256')==sha256_file(output):return prior
        raise ValueError('Existing tokenizer differs or is uncertified; use a new output path')
    def texts():
        with resolve(prepared['samples']['path']).open() as stream:
            for line in stream:yield json.loads(line)['text']
    tok=make_tokenizer(normalization)
    tok.train_from_iterator(texts(),trainers.BpeTrainer(vocab_size=vocab,special_tokens=SPECIAL,show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=2))
    if tok.get_vocab_size()!=vocab:raise ValueError('Candidate vocabulary did not reach requested size')
    output.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(output)+'.tmp');tok.save(str(tmp));tmp.replace(output)
    result=dict(config=config,config_id=identity(config),sampled=prepared['stats'],tokenizer_sha256=sha256_file(output),
                special_ids={x:tok.token_to_id(x) for x in SPECIAL})
    atomic_json(marker,result);return result


def main():
    import math
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources-manifest',required=True)
    ap.add_argument('--mix',nargs='+',required=True,help='source:weight by UTF8 bytes')
    ap.add_argument('--mb',type=int,default=3000);ap.add_argument('--vocab',type=int,default=65536)
    ap.add_argument('--output',default='tokenizer/candidate/tokenizer.json')
    ap.add_argument('--normalization',choices=['raw','NFC'],default='raw')
    ap.add_argument('--split-manifest')
    ap.add_argument('--sample-dir',help='Separate certified sample corpus directory')
    ap.add_argument('--experimental-unsplit',action='store_true',help='Explicit legacy experiment only; no verified eval exclusion')
    a=ap.parse_args()
    if a.mb<1 or not 272<=a.vocab<=65536:
        ap.error('Require positive MB and vocabulary 272..65536')
    entries=[x.rsplit(':',1) for x in a.mix]
    if len(set(x[0] for x in entries))!=len(entries):
        ap.error('Repeated source name')
    weights={s:float(w) for s,w in entries}
    if any(not math.isfinite(w) or w<=0 for w in weights.values()) or not math.isfinite(sum(weights.values())):
        ap.error('Weights must be finite and positive')
    if bool(a.split_manifest)==bool(a.experimental_unsplit):ap.error('Require split manifest, or explicit experimental-unsplit mode')
    if a.split_manifest:
        from data.split_inputs import prepare
        budgets={s:int(a.mb*1e6*w/sum(weights.values())) for s,w in weights.items()}
        sample_dir=resolve(a.sample_dir) if a.sample_dir else Path(str(resolve(a.output))+'.samples')
        prepare(a.sources_manifest,a.split_manifest,sample_dir,budgets,mode='train',seed=0,max_chars=20000)
        result=train_prepared(sample_dir/'manifest.json',a.output,a.vocab,a.normalization)
        print('SPLIT-VERIFIED CANDIDATE TOKENIZER',a.output,json.dumps(result['sampled']));return
    sources={};snapshots={}
    for source in weights:
        sources[source],snapshots[source]=source_files(source,a.sources_manifest)
    config=dict(vocab=a.vocab,mb=a.mb,source_manifest_sha256=sha256_file(resolve(a.sources_manifest)),
                sources={s:dict(weight=weights[s],snapshot=snapshots[s],
                         files=[dict(path=artifact_path(f),bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in sources[s]]) for s in sources},
                normalization=a.normalization,digits='individual',special_tokens=SPECIAL,seed=0,max_document_chars=20000)
    output=resolve(a.output);marker=Path(str(output)+'.done')
    if output.exists():
        if marker.exists():
            previous=json.loads(marker.read_text())
            if previous['config_id']==identity(config) and previous['tokenizer_sha256']==sha256_file(output):
                print('REUSED TOKENIZER',output);return
        raise ValueError('Existing tokenizer differs or is uncertified; use a new output path')
    sampled={}
    def iterator():
        for source,weight in weights.items():
            sampled[source]={}
            yield from sample_texts(sources[source],int(a.mb*1e6*weight/sum(weights.values())),stats=sampled[source])
    tok=make_tokenizer(a.normalization)
    trainer=trainers.BpeTrainer(vocab_size=a.vocab,special_tokens=SPECIAL,show_progress=True,
                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=2)
    tok.train_from_iterator(iterator(),trainer)
    if tok.get_vocab_size()!=a.vocab:
        raise ValueError('Candidate vocabulary did not reach requested size; review sample before continuing')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(output)+'.tmp');tok.save(str(tmp));tmp.replace(output)
    atomic_json(marker,dict(config=config,config_id=identity(config),sampled=sampled,
                            tokenizer_sha256=sha256_file(output),special_ids={x:tok.token_to_id(x) for x in SPECIAL}))
    print('CANDIDATE TOKENIZER',str(output),json.dumps(sampled),flush=True)


if __name__=='__main__':
    main()
