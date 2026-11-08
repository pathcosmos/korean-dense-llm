"""Read only a fully published split; never inspect an in-progress builder DB."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file
from data.verified_split_context import VerifiedSplitContext,distribution


def report(sources,split,output):
    out=resolve(output)
    if out.exists():raise ValueError('Preserve existing report; use a new output')
    with VerifiedSplitContext(sources,split) as context:result=distribution(context);context.assert_control_files()
    atomic_json(out,result);atomic_json(str(out)+'.done',dict(sha256=sha256_file(out)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',required=True);p.add_argument('--split',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report(a.sources,a.split,a.output)
