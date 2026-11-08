"""Compare RAM-preloaded BPE against the completed, identical streaming experiment.
Includes JSON decoding/preload time; forbids production use of this experimental tokenizer.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file,identity


def worker(a):
    from scripts.benchmark_tokenizer_resources import frame,texts
    from tokenizer.train_tokenizer import make_tokenizer,SPECIAL
    from tokenizers import trainers,pre_tokenizers
    fr=frame(a.frame);tok=make_tokenizer('raw');start=time.perf_counter()
    rows=list(texts([f['path'] for f in fr['shards']]));loaded=time.perf_counter()
    tok.train_from_iterator(iter(rows),trainers.BpeTrainer(vocab_size=65536,min_frequency=2,special_tokens=SPECIAL,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    end=time.perf_counter()
    if tok.get_vocab_size()!=65536:raise ValueError('Vocabulary mismatch')
    for text in rows[:256]:
        if tok.decode(tok.encode(text,add_special_tokens=False).ids,skip_special_tokens=False)!=text:raise ValueError('Raw roundtrip failed')
    out=resolve(a.output);target=out/'experimental_tokenizer.json';tok.save(str(target))
    atomic_json(out/'result.json',dict(status='experimental_performance_only',rayon=int(os.environ['RAYON_NUM_THREADS']),
        documents=len(rows),utf8_bytes=sum(len(x.encode()) for x in rows),preload_seconds=loaded-start,bpe_seconds=end-loaded,
        total_preload_train_seconds=end-start,semantic_id=identity(json.loads(target.read_text())),tokenizer_sha256=sha256_file(target),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Same512MiB experimental input; RAM loading is charged, preceding frame SHA verification excluded as in streaming measurement.'))


def main(a):
    from scripts.hardware_profile import inspect
    from scripts.parallel_job_support import run_cases,certificate
    reference=resolve(a.reference);proof=json.loads((reference/'.done').read_text())
    for file,key in [('report.json','report_sha256'),('contract.json','contract_sha256')]:
        if sha256_file(reference/file)!=proof[key]:raise ValueError('Reference changed')
    ref=json.loads((reference/'report.json').read_text());previous=json.loads((reference/'contract.json').read_text())
    for file,digest in previous['code'].items():
        if sha256_file(resolve(file))!=digest:raise ValueError('Reference producer changed')
    remaining=1200-sum(x['supervisor_seconds'] for x in ref['results'])
    if remaining<60:raise ValueError('Combined20min campaign budget has insufficient time for RAM comparison')
    hw=inspect()
    if hw['memory']['conservative_headroom_gib']<576:raise RuntimeError('Reserve main future RSS plus192GiB')
    root=resolve(a.output);root.mkdir(parents=True,exist_ok=False);n=ref['bpe_candidate_threads']
    stream=next(r for r in ref['results'] if r['case']==f'bpe_{n}')
    contract=dict(reference_report_sha256=proof['report_sha256'],code=dict(previous['code'],**{'scripts/benchmark_bpe_memory.py':sha256_file(Path(__file__))}),
                  rayon=n,hardware=hw,max_seconds=min(remaining,600),max_rss_gib=192,scope='One matched RAM vs streaming observation; production adoption pending.')
    atomic_json(root/'contract.json',contract);scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS=str(n),TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch))
    cmd=['taskset','-c',','.join(map(str,hw['cpu']['affinity'][:64])),sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(root/'ram'),'--frame',str(reference/'prepare/frame.json')]
    try:
        execution=run_cases(root,[dict(name='ram',out=str(root/'ram'),cmd=cmd,env=env)],max_seconds=contract['max_seconds'],max_rss_gib=192)
        r=json.loads((root/'ram/result.json').read_text())
        if r['semantic_id']!=stream['semantic_id']:raise ValueError('RAM/stream tokenizer differs')
        if r['max_rss_kib']*1024>192*2**30:raise ValueError('RAM worker HWM budget')
        certificate(root,dict(status='measurement_complete',ram=r,stream=stream,execution=execution,
            ram_vs_stream_speedup=stream['train_seconds']/r['total_preload_train_seconds'],production_adoption='pending',scope=contract['scope']),contract)
    except BaseException as exc:atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--frame')
    p.add_argument('--reference',default='runs/next_stage_optimization_v1');a=p.parse_args()
    worker(a) if a.worker else main(a)
