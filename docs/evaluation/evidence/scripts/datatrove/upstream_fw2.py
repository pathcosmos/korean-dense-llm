"""Reconstruct FineWeb-2 kor_Hang step-5 filters (fineweb-2-pipeline.py @ d0defb2) with datatrove, per document.
Usage (dt venv): upstream_fw2.py IN.jsonl OUT.jsonl [nproc]. Output has NO text. Each stage is evaluated independently
(no short-circuit) so we also get per-stage flags; `first` = first failing stage in pipeline order."""
import os, sys, json, warnings, yaml, multiprocessing as mp
warnings.filterwarnings('ignore')
CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fw2_kor_Hang.yml')

def build():
    from datatrove.data import Document
    from datatrove.pipeline.filters import GopherRepetitionFilter, FineWebQualityFilter, GopherQualityFilter
    from datatrove.utils import word_tokenizers as wt
    # memoise: all three filters tokenise the same text; tokenise once per doc (no behaviour change)
    orig = wt.KiwiTokenizer.word_tokenize
    cache = {}
    def memo(self, text):
        k = hash(text)
        if cache.get('k') != k:
            cache['k'] = k; cache['v'] = orig(self, text)
        return cache['v']
    wt.KiwiTokenizer.word_tokenize = memo
    c = yaml.safe_load(open(CFG)); L = 'kor_Hang'
    rep = GopherRepetitionFilter(language=L, dup_para_frac=0, dup_line_char_frac=0, dup_para_char_frac=0,
        dup_line_frac=c['dup_line_frac'], top_n_grams=c['top_n_grams'], dup_n_grams=c['dup_n_grams'])
    fwq = FineWebQualityFilter(language=L, short_line_thr=999, char_duplicates_ratio=0.1,
        line_punct_thr=c['line_punct_thr'], new_line_ratio=c['new_line_ratio'])
    gq = GopherQualityFilter(language=L, max_avg_word_length=c['max_avg_word_length'], min_avg_word_length=c['min_avg_word_length'],
        stop_words=c['stopwords'], max_non_alpha_words_ratio=c['max_non_alpha_words_ratio'], min_stop_words=2)
    return Document, [('gopher_rep', rep), ('fineweb_qual', fwq), ('gopher_qual', gq)]

def work(args):
    global G
    if 'G' not in globals(): G = build()
    Document, stages = G
    idx, text = args
    res = {}; first = None
    for name, f in stages:
        try:
            r = f.filter(Document(text=text, id=str(idx)))
        except Exception as e:
            r = (False, 'EXC:' + type(e).__name__)
        ok, why = (r, None) if isinstance(r, bool) else r
        res[name] = None if ok else why
        if not ok and first is None: first = (name, why)
    return idx, res, first

if __name__ == '__main__':
    inp, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    def gen():
        for i, l in enumerate(open(inp, encoding='utf-8')): yield i, json.loads(l)['text']
    with mp.Pool(n) as p, open(out, 'w') as fo:
        for k, (idx, res, first) in enumerate(p.imap(work, gen(), chunksize=20)):
            fo.write(json.dumps(dict(i=idx, stages=res, first=first)) + '\n')
            if k % 5000 == 0: print(k, flush=True)
