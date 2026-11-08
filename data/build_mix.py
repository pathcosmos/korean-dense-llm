"""Build a checked mix from completed v2 token sources, never from stale globs."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve, atomic_json
from data.tokenize_shards import commit_outputs, TOKEN_FORMAT, GROUPED_TOKEN_FORMAT
from train.data import SAMPLER_VERSION


def portable(path):
    return str(Path(path).resolve().relative_to(Path(PROJECT_ROOT).resolve()))


def build_mix(weights, token_root, output, seq_len=4096, target_tokens=None):
    if type(seq_len) is not int or seq_len < 1 or not weights:
        raise ValueError('Require sources and a positive sequence length')
    sources, holdout = {}, []
    common = None
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Source weights must be finite and positive')
        root = Path(token_root) / name
        plan = json.loads((root / 'manifest.json').read_text())
        done = json.loads((root / '.done').read_text())
        if plan['source'] != name or done['contract_id'] != plan['contract_id'] or done['shards'] != len(plan['inputs']):
            raise ValueError('Incomplete token source')
        current = dict(tokenizer_sha256=plan['tokenizer_sha256'], eos_id=plan['eos_id'],
                       dtype=plan['dtype'], format=plan['format'])
        if current['dtype'] != 'uint16' or current['format'] not in (TOKEN_FORMAT, GROUPED_TOKEN_FORMAT):
            raise ValueError('Unsupported token format')
        if common is not None and current != common:
            raise ValueError('Cannot mix different tokenizer/EOS/dtype contracts')
        common = current
        files = [];count = 0;hold_names = set()
        for inp in plan['inputs']:
            filename = Path(inp['path']).stem + '.bin'
            record = json.loads((root / (filename + '.done')).read_text())
            if record['contract_id'] != plan['contract_id']:
                raise ValueError('Token shard contract mismatch')
            commit_outputs(root, record, publish_done=False)
            if (root / filename).stat().st_size != record['tokens'] * 2:
                raise ValueError('Token shard has inconsistent count')
            if record['tokens']:
                if record['tokens'] < seq_len + 1: raise ValueError('Token shard is short')
                files.append(portable(root / filename));count += record['tokens']
            elif record['train_docs'] != 0:
                raise ValueError('Empty shard with training documents')
            if record['holdout_tokens']:
                hold_name = record.get('holdout_file') or 'hold.bin'
                if hold_name in hold_names or (root / hold_name).stat().st_size != record['holdout_tokens'] * 2 or record['holdout_tokens'] < seq_len + 1:
                    raise ValueError('Invalid holdout token file')
                hold_names.add(hold_name);holdout.append(portable(root / hold_name))
        if not files: raise ValueError('Source has no usable training tokens')
        sources[name] = dict(weight=float(weight), files=files, tokens=count, contract_id=plan['contract_id'])
    denominator = sum(v['weight'] for v in sources.values())
    if not math.isfinite(denominator):
        raise ValueError('Weight sum is not finite')
    if target_tokens is not None and (type(target_tokens) is not int or target_tokens < 1):
        raise ValueError('target_tokens must be a positive integer')
    for value in sources.values():
        value['weight'] /= denominator
        if target_tokens is not None:
            value['expected_exposures'] = target_tokens * value['weight'] / value['tokens']
    result = dict(schema_version=2, sampler_version=SAMPLER_VERSION, seq_len=seq_len,
                  **common, sources=sources, holdout=holdout)
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError('Existing mix differs; choose a new output path')
    else:
        atomic_json(output, result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sources',nargs='+',help='source:weight')
    ap.add_argument('--token-root',default='data/tokens');ap.add_argument('--output',default='data/tokens/mix.json')
    ap.add_argument('--seq-len',type=int,default=4096);ap.add_argument('--target-tokens',type=int)
    a=ap.parse_args();pairs=[x.rsplit(':',1) for x in a.sources]
    if len(set(x[0] for x in pairs)) != len(pairs):
        ap.error('Source names must be unique')
    result=build_mix({name:float(w) for name,w in pairs},resolve(a.token_root),resolve(a.output),a.seq_len,a.target_tokens)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
