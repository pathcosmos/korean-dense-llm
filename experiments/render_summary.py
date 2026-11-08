"""Render measured comparisons from bundled result JSON, without rerunning benchmarks."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(run,name='report.json'):return json.loads((HERE/'results'/run/name).read_text())
def main():
    sha=load('source_hash_sweep32_v1');storage=load('resource_policy_validation_v1','local_vs_nfs.json');batch=load('actual_deterministic_batch_v1');bpe=load('bpe_memory_v1');cache=load('split_cache_sweep_v1');token=load('next_stage_optimization_v1','review.json');compile=load('compile_long_v1','review.json')
    comparisons=[
      ['SHA256 workers16/32',f"{sha['median_seconds']['16']:.3f}s / {sha['median_seconds']['32']:.3f}s",'32 selected files; 109.70GB/pass; repeated warm-cache shared-host observations; matching hashes','source_hash_sweep32_v1/report.json'],
      ['SQLite NFS/local',f"{storage['timing']['nfs_seconds']:.3f}s / {storage['timing']['local_seconds']:.3f}s",'1M synthetic rows; cache512MiB; same logical digest; single shared-host comparison','resource_policy_validation_v1/local_vs_nfs.json'],
      ['SQLite cache512/8192/131072MiB',' / '.join(f"{x['build_and_verify_seconds']:.3f}s" for x in cache['cases']),'3M synthetic rows; build+verify; upper cache limit != resident RAM; no full-corpus optimum claim','split_cache_sweep_v1/report.json'],
      ['Token encoding single/parallel',f"{token['baseline_seconds']:.3f}s / {token['nfs_median_seconds']:.3f}s",'512MiB experimental input; identical encodings; four sources, warm cache; final-input pilot separate','next_stage_optimization_v1/review.json'],
      ['BPE stream/RAM',f"{bpe['stream']['train_seconds']:.3f}s / {bpe['ram']['total_preload_train_seconds']:.3f}s",'Matched512MiB input; RAM includes preload; one observation; RAM candidate slower','bpe_memory_v1/report.json'],
    ]
    for model in ['1b','3b']:
        rows=[x for x in batch['results'] if x['model']==model and not x['repeat']]
        comparisons.append([f'{model} actual-token eager throughput',' / '.join(f"micro{x['micro_bsz']}: {x['aggregate_tok_s']/1000:.2f}k tok/s" for x in rows),'8 GPUs; seq4096; 6 timed updates; evaluation/checkpoint costs excluded; model quality not measured','actual_deterministic_batch_v1/report.json'])
    rows=[x for x in compile['cases'] if x['ranks']==8]
    comparisons.append(['Compile diagnostic 8GPU',' / '.join(f"{x['throughput_change_percent']:+.3f}%" for x in rows),'Two seeds; numerical diagnostic and production adoption are separate; no useful speedup','compile_long_v1/review.json'])
    text='''# Optimization experiments / 최적화 실험

These are **historical measurements**, not new benchmarks of the public package. Reports preserve the original scope/status, including rejected or pending candidates. The original machine had 8 H200 GPUs, container CPU capacity96 and memory limit1920GiB; each contract/hardware report supplies the actual case settings. Do not extrapolate short, warm-cache or synthetic measurements to full-corpus performance.

각 실험은 목적·비교 후보·조건·측정 결과·채택/기각 이유를 함께 읽습니다. 속도 향상만으로 수치 정확성·학습 품질 통과를 뜻하지 않습니다. `index.json`의 원본/공개 SHA와 `code_bindings.json`의 당시 코드 연결을 확인하세요. 개인정보 경로와 본문/sample 필드는 변환·제외했고 수치 자체는 보존했습니다.

## Selected measured comparisons

| Experiment | Measurement | Scope / interpretation | Evidence |
|---|---|---|---|
'''
    for title,value,scope,ref in comparisons:text+=f'| {title} | {value} | {scope} | [JSON](results/{ref}) |\n'
    text+='''
## How to reproduce or extend a comparison

1. Find the campaign below and read its `contract.json`, report, and review. Bind source/input identity, seed, model geometry and software environment.
2. Inspect the corresponding `scripts/benchmark_*`, `probe_*` or `validate_*` CLI and use an explicit new output directory. GPU recipes require the indicated devices and input data, which are not bundled.
3. Measure total elapsed including the same validation/publication boundaries; report warmup/compile cost separately. Preserve restart/output equivalence and compare memory as well as time.
4. Report all measured candidates and unsuccessful attempts. A missing historical source hash is explicitly unresolved in `code_bindings.json`, not silently replaced with current code.

The large GPU/data campaigns are not executed by CI. `python experiments/render_summary.py` rebuilds this table from included JSON only. With `requirements-bench.txt`, it also creates the SVG below.

![Selected measured comparisons](summary.svg)

## Campaign index

| Campaign | Result files | Historical code snapshots |
|---|---:|---:|
'''
    index=json.loads((HERE/'index.json').read_text())
    for c in index['campaigns']:text+=f"| [{c['id']}](results/{c['id']}/) | {len(c['reports'])} | {len(c['code_snapshots'])} |\n"
    text+='\nHistorical code objects are under `code/<sha256>.py`; `index.json` maps each to its original path. They are source evidence, not supported standalone CLI entrypoints. Results can reference excluded input files/DBs/checkpoints; those paths document the original experiment and are not promised downloads.\n'
    (HERE/'README.md').write_text(text)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'korean-dense-v0.1.0'})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6));fig.patch.set_facecolor('#fafafa')
    panels=[('SHA256: 109.70GB/pass',['16 workers','32 workers'],[sha['median_seconds']['16'],sha['median_seconds']['32']],'Seconds (lower is faster)'),('SQLite: 1M synthetic rows',['NFS','Local'],[storage['timing']['nfs_seconds'],storage['timing']['local_seconds']],'Seconds (lower is faster)'),('3B eager: 8 GPUs',['micro2','micro4'],[x['aggregate_tok_s']/1000 for x in batch['results'] if x['model']=='3b' and not x['repeat']],'Thousand tokens/s')]
    for ax,(title,labels,values,xlabel) in zip(axes,panels):
        bars=ax.barh(labels,values,color=['#8ba9c9','#296b8c']);ax.set_title(title);ax.set_xlabel(xlabel);ax.invert_yaxis();ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_xlim(0,max(values)*1.25);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Historical bounded comparisons — see scope and evidence table',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'summary.svg',metadata={'Date':None});plt.close(fig)
if __name__=='__main__':main()
