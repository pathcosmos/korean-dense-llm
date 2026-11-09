"""Opt-in FineWeb-2-style Korean quality filter, built on datatrove.

Why: on sources FineWeb-2 never saw (HPLT, Wikipedia) data/filters.check_ko is far more permissive than the
FineWeb-2 recipe (measured: docs/evaluation/IMPROVEMENTS.md section 7). This module lets a cleaning run apply the
same three datatrove filters, in the same order and with the same arguments as fineweb-2-pipeline.py, using a
language config such as configs/kor_Hang.yml supplied by the user.

It is NOT part of the default pipeline: nothing imports it unless a policy asks for it (data/clean.py --policy).
Costs and caveats, all measured or stated:
  * heavy optional dependencies (pip install -r requirements-fw2.txt); they are imported lazily
  * Korean word tokenisation by Kiwi makes it much slower than check_ko (see the throughput note in IMPROVEMENTS)
  * it rejects ~7/67 labelled-good documents on the 100-document gold (check_ko: 2/67) in exchange for catching
    10/27 labelled semantic-junk documents (check_ko: 3/27). Whether that trade helps a model is NOT known
    (needs a training ablation).
  * it runs after check_ko, on the norm_ws text, without the upstream steps that precede step 5 in FineWeb-2
    (URL filter, extraction, language ID) or follow it (FTFY, PII). Which datatrove version FineWeb-2 itself used
    is unknown (UNCERTAIN); the reconstruction was checked argument by argument against fineweb-2-pipeline.py
    at commit d0defb24.

This file must stay importable by bare name (data/clean.py puts data/ on sys.path): stdlib only at import time.
"""
import hashlib
import importlib.metadata as _md
import importlib.util as _util
import os

REQUIREMENTS = 'datatrove==0.10.1 kiwipiepy==0.21.0 kiwipiepy_model==0.21.0 PyYAML (see requirements-fw2.txt)'
STAGES = ('gopher_rep', 'fineweb_qual', 'gopher_qual')
REQUIRED_KEYS = ('dup_line_frac', 'top_n_grams', 'dup_n_grams', 'line_punct_thr', 'new_line_ratio',
                 'max_avg_word_length', 'min_avg_word_length', 'stopwords', 'max_non_alpha_words_ratio')


def available():
    return all(_util.find_spec(m) is not None for m in ('datatrove', 'kiwipiepy', 'yaml'))


def require():
    if not available():
        raise ImportError('The FineWeb-2-style quality filter needs optional packages: pip install ' + REQUIREMENTS)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def environment(spec):
    """What the verdicts depend on, for the cleaning signature: config content, library versions, this module."""
    require()
    return dict(config_sha256=_sha256(spec['config']), datatrove=_md.version('datatrove'),
                kiwipiepy=_md.version('kiwipiepy'), module_sha256=_sha256(os.path.abspath(__file__)))


def parse_spec(raw):
    """Validate the policy fragment {"engine": "fw2", "config": path, "language": "kor_Hang"}."""
    unknown = set(raw) - {'engine', 'config', 'language'}
    if unknown:
        raise ValueError(f'unknown quality policy fields: {sorted(unknown)}')
    if raw.get('engine') != 'fw2':
        raise ValueError('quality.engine must be "fw2"')
    if not raw.get('config'):
        raise ValueError('quality.config (path to a FineWeb-2 language config such as kor_Hang.yml) is required')
    return dict(engine='fw2', config=os.path.abspath(raw['config']), language=raw.get('language', 'kor_Hang'))


class Fw2Quality:
    """check(text) -> None if kept, else 'fw2:<stage>:<reason>' (first failing stage, in pipeline order)."""

    def __init__(self, config_path, language='kor_Hang'):
        require()
        import yaml
        from datatrove.pipeline.filters import FineWebQualityFilter, GopherQualityFilter, GopherRepetitionFilter
        from datatrove.utils import word_tokenizers as wt
        with open(config_path, encoding='utf-8') as f:
            c = yaml.safe_load(f)
        missing = [k for k in REQUIRED_KEYS if k not in c]
        if missing:
            raise ValueError(f'FineWeb-2 config lacks keys: {missing}')
        # The three filters tokenise the same text with Kiwi; tokenise once per document (no change in behaviour).
        if not getattr(wt.KiwiTokenizer, '_dataqa_memo', False):
            orig, last = wt.KiwiTokenizer.word_tokenize, {}

            def memo(self, text):
                if last.get('text') is not text:
                    last['text'], last['tokens'] = text, orig(self, text)
                return last['tokens']
            wt.KiwiTokenizer.word_tokenize = memo
            wt.KiwiTokenizer._dataqa_memo = True
        # Same arguments as fineweb-2-pipeline.py (commit d0defb24), step 5.
        self.stages = [
            ('gopher_rep', GopherRepetitionFilter(
                language=language, dup_para_frac=0, dup_line_char_frac=0, dup_para_char_frac=0,
                dup_line_frac=c['dup_line_frac'], top_n_grams=c['top_n_grams'], dup_n_grams=c['dup_n_grams'])),
            ('fineweb_qual', FineWebQualityFilter(
                language=language, short_line_thr=999, char_duplicates_ratio=0.1,
                line_punct_thr=c['line_punct_thr'], new_line_ratio=c['new_line_ratio'])),
            ('gopher_qual', GopherQualityFilter(
                language=language, max_avg_word_length=c['max_avg_word_length'],
                min_avg_word_length=c['min_avg_word_length'], stop_words=c['stopwords'],
                max_non_alpha_words_ratio=c['max_non_alpha_words_ratio'], min_stop_words=2))]
        from datatrove.data import Document
        self._document = Document

    def check(self, text):
        for name, stage in self.stages:
            try:
                result = stage.filter(self._document(text=text, id='0'))
            except Exception as e:   # a pathological document must not kill a multi-hour shard; make it visible in counts
                return f'fw2:{name}:error:{type(e).__name__}'
            ok, why = (result, None) if isinstance(result, bool) else result
            if not ok:
                return f'fw2:{name}:{why}'
        return None


_CACHE = {}


def get(spec):
    """Per-process cached filter for a validated spec (workers build their own)."""
    key = (spec['config'], spec['language'])
    if key not in _CACHE:
        _CACHE[key] = Fw2Quality(spec['config'], spec['language'])
    return _CACHE[key]
