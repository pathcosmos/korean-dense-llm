"""Pinned, randomised real-document sampling for label review. Real text is never committed.

  sample       draw N documents from random shards / row groups of a pinned source revision and
               write (a) a text-free manifest (committable) and (b) a LOCAL review sheet
  materialize  join manifest + labels, re-fetch the texts, verify sha256, write a LOCAL gold file

Committable artefacts: manifest (repo, revision, file, row_group, row, sha256) and labels.
The review directory holds raw web text, which can contain real personal data: keep it out of
git (add `dataqa/review/` to .git/info/exclude) and do not publish it anywhere.

Which shards the original pipeline ingested is only known as a count (docs/data_provenance_v1/
source_registry.json says "25 shards"); data/download.py takes the first N of the sorted listing,
so this module samples from the first N sorted shards. That equivalence is an ASSUMPTION (불확실).

Network: huggingface.co only; one Parquet row group (~1000 docs) is read per sampled document via
range requests, no full-shard download.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from data import filters as F
from dataqa import gold, ko_rules, pii_rules

REGISTRY = Path(__file__).resolve().parents[1] / 'docs' / 'data_provenance_v1' / 'source_registry.json'
# Only fineweb2_ko is verified (columns text,id). hplt_ko / wiki_ko have other column layouts and are NOT supported yet.
SUBSETS = {'fineweb2_ko': 'data/kor_Hang/train'}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def pinned(source):
    for entry in json.loads(REGISTRY.read_text(encoding='utf-8'))['sources']:
        if entry['source'] == source:
            return entry['repository'], entry['pinned_revision']
    raise SystemExit(f'source not in registry: {source}')


def plan(shards, rg_counts, n, seed):
    """Deterministic plan: n distinct (shard, row_group) draws with a row index inside the group.

    rg_counts: {shard: number_of_row_groups}. Pure function, unit-tested without network.
    """
    rng = random.Random(seed)
    seen, out = set(), []
    if n > sum(rg_counts[s] for s in shards):
        raise ValueError('more documents requested than row groups available')
    while len(out) < n:
        shard = rng.choice(shards)
        rg = rng.randrange(rg_counts[shard])
        if (shard, rg) in seen:
            continue
        seen.add((shard, rg))
        out.append(dict(file=shard, row_group=rg, row_draw=rng.random()))
    return out


class Reader:
    def __init__(self, repo, revision):
        from huggingface_hub import HfFileSystem
        self.fs, self.prefix = HfFileSystem(), f'datasets/{repo}@{revision}/'
        self.files, self.cache = {}, {}

    def parquet(self, path):
        import pyarrow.parquet as pq
        if path not in self.files:
            self.files[path] = pq.ParquetFile(self.fs.open(self.prefix + path, 'rb'))
        return self.files[path]

    def row_groups(self, path):
        return self.parquet(path).metadata.num_row_groups

    def group(self, path, rg):
        if (path, rg) not in self.cache:
            self.cache[(path, rg)] = self.parquet(path).read_row_group(rg, columns=['text', 'id'])
        return self.cache[(path, rg)]


def list_shards(repo, revision, subset, limit):
    from huggingface_hub import HfApi
    items = HfApi().list_repo_tree(repo, repo_type='dataset', revision=revision, path_in_repo=subset)
    return sorted(i.path for i in items if i.path.endswith('.parquet'))[:limit]


def fetch_text(reader, file, rg, row):
    table = reader.group(file, rg)
    return table.column('text')[row].as_py(), table.column('id')[row].as_py()


def review_entry(rid, text, label_hint=None):
    reason = ko_rules.filter_reason(text)
    _, base = pii_rules.apply(text, pii_rules.PRESETS['baseline'])
    _, loose = pii_rules.apply(text, pii_rules.PRESETS['all_candidates_loose'])
    return dict(id=rid, current_filter=reason, baseline_pii=base, candidate_pii=loose)


def render_sheet(entries, texts, head=1200, tail=300):
    out = ['# real-sample review sheet (LOCAL ONLY — contains raw web text, do not commit or publish)', '']
    for e in entries:
        t = texts[e['id']]
        out += [f"## {e['id']}  ({len(t)} chars)", '',
                f"- current filter: `{e['current_filter']}`",
                f"- baseline PII detections: {[(d['type'], t[d['start']:d['end']]) for d in e['baseline_pii']]}",
                f"- candidate (loose) PII detections: {[(d['type'], t[d['start']:d['end']]) for d in e['candidate_pii']]}",
                '', '```', t[:head] + ('\n…[중략]…\n' + t[-tail:] if len(t) > head + tail else t[head:]), '```', '']
    return '\n'.join(out)


def cmd_sample(a):
    repo, revision = pinned(a.source)
    reader = Reader(repo, revision)
    shards = list_shards(repo, revision, SUBSETS[a.source], a.shards)
    counts = {s: reader.row_groups(s) for s in shards}
    manifest, entries, texts = [], [], {}
    for item in plan(shards, counts, a.n, a.seed):
        group = reader.group(item['file'], item['row_group'])
        row = int(item['row_draw'] * group.num_rows)
        raw, doc_id = fetch_text(reader, item['file'], item['row_group'], row)
        text = F.norm_ws(raw)
        rid = 'real-' + sha(f"{repo}@{revision}/{item['file']}#{item['row_group']}.{row}")[:12]
        manifest.append(dict(id=rid, source=a.source, repo=repo, revision=revision, file=item['file'],
                             row_group=item['row_group'], row=row, doc_id=doc_id,
                             raw_sha256=sha(raw), text_sha256=sha(text), n_chars=len(text),
                             sampling=dict(seed=a.seed, shards_considered=len(shards))))
        entries.append(review_entry(rid, text))
        texts[rid] = text
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text('\n'.join(json.dumps(m, ensure_ascii=False, sort_keys=True) for m in manifest) + '\n', encoding='utf-8')
    sheet = Path(a.sheet_dir)
    sheet.mkdir(parents=True, exist_ok=True)
    (sheet / 'review.md').write_text(render_sheet(entries, texts), encoding='utf-8')
    (sheet / 'sheet.jsonl').write_text('\n'.join(json.dumps(dict(e, text=texts[e['id']]), ensure_ascii=False) for e in entries) + '\n', encoding='utf-8')
    print(f'{len(manifest)} documents -> manifest {a.manifest}; LOCAL sheet in {sheet}/ (keep out of git)')


def cmd_materialize(a):
    manifest = [json.loads(l) for l in Path(a.manifest).read_text(encoding='utf-8').splitlines() if l.strip()]
    labels = {r['id']: r for r in (json.loads(l) for l in Path(a.labels).read_text(encoding='utf-8').splitlines() if l.strip())}
    readers, records = {}, []
    for m in manifest:
        if m['id'] not in labels:
            continue
        reader = readers.setdefault((m['repo'], m['revision']), Reader(m['repo'], m['revision']))
        raw, _ = fetch_text(reader, m['file'], m['row_group'], m['row'])
        text = F.norm_ws(raw)
        if sha(raw) != m['raw_sha256'] or sha(text) != m['text_sha256']:
            raise SystemExit(f"sha256 mismatch for {m['id']}: pinned revision content changed or sampling differs")
        lab = labels[m['id']]
        records.append(dict(id=m['id'], text=text, label=lab['label'], reject_reason=lab.get('reject_reason'),
                            source_kind='real_reviewed', label_source=lab.get('label_source', 'unknown'),
                            tags=lab.get('tags', []), pii_spans=lab.get('pii_spans', []),
                            ambiguous=bool(lab.get('ambiguous', False)),
                            **({'policy_assumption': lab['policy_assumption'],
                                'policy_reject_reason': lab.get('policy_reject_reason')} if lab.get('policy_assumption') else {})))
    gold.dump(records, a.out)
    print(f'{len(records)} labelled documents -> {a.out} (LOCAL; contains raw web text)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--source', choices=sorted(SUBSETS), default='fineweb2_ko')
    s.add_argument('--n', type=int, default=20)
    s.add_argument('--seed', type=int, default=1)
    s.add_argument('--shards', type=int, default=25, help='first N sorted shards (assumption, see module doc)')
    s.add_argument('--manifest', default='dataqa/real/pilot_manifest_v1.jsonl')
    s.add_argument('--sheet-dir', default='dataqa/review')
    s.set_defaults(fn=cmd_sample)
    m = sub.add_parser('materialize')
    m.add_argument('--manifest', default='dataqa/real/pilot_manifest_v1.jsonl')
    m.add_argument('--labels', default='dataqa/real/pilot_labels_v1.jsonl')
    m.add_argument('--out', default='dataqa/review/real_gold.jsonl')
    m.set_defaults(fn=cmd_materialize)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
