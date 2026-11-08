"""Audit the isolated GitHub Code pilot without changing any dataset contents."""
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT, DATA
import pyarrow.parquet as pq

ALLOWED = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'cc0-1.0', 'unlicense'}
LANG = {'.py': 'Python', '.js': 'JavaScript', '.jsx': 'JavaScript', '.ts': 'TypeScript',
        '.tsx': 'TypeScript', '.c': 'C', '.h': 'C/C++ header', '.cpp': 'C++', '.cc': 'C++',
        '.hpp': 'C++', '.java': 'Java', '.go': 'Go', '.rs': 'Rust', '.sh': 'Shell',
        '.sql': 'SQL', '.html': 'HTML', '.css': 'CSS', '.md': 'Markdown'}


def main():
    root = Path(DATA) / 'raw/github_code_pilot'
    files = sorted(root.rglob('*.parquet'))
    manifest = json.loads((root / 'download_manifest.json').read_text())
    licenses, languages, rejected, selected_languages = (collections.Counter() for _ in range(4))
    docs = size = selected = selected_bytes = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=['path', 'content', 'license']):
            for row in b.to_pylist():
                text = row['content']
                license_name = (row['license'] or '').lower()
                language = LANG.get(Path(row['path']).suffix.lower(), 'Other')
                licenses[license_name] += 1; languages[language] += 1; docs += 1
                if not isinstance(text, str):
                    rejected['non_string'] += 1
                    continue
                nbytes = len(text.encode()); size += nbytes
                if license_name not in ALLOWED:
                    rejected['license_not_selected'] += 1
                elif len(text) < 20:
                    rejected['short'] += 1
                elif len(text) > 200000:
                    rejected['long'] += 1
                elif '\x00' in text:
                    rejected['nul'] += 1
                else:
                    selected += 1; selected_bytes += nbytes; selected_languages[language] += 1
    report = dict(repo=manifest['repo'], revision=manifest['revision'], files=len(files),
                  documents=docs, utf8_bytes=size, licenses=dict(licenses), languages=dict(languages),
                  candidate_policy=dict(allowed_license_metadata=sorted(ALLOWED), min_chars=20, max_chars=200000),
                  candidate_docs=selected, candidate_bytes=selected_bytes,
                  candidate_languages=dict(selected_languages), rejected=dict(rejected),
                  limitations='One early shard only; neither representative nor deduplicated. Metadata license labels are not per-file legal verification. Source code whitespace/case left unchanged.')
    output = Path(PROJECT_ROOT) / 'logs/github_code_pilot_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
