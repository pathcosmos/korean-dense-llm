"""Read-only readiness report; exits 0 passed, 2 pending, 1 failed/invalid evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.quality_report import assess,save_report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--sources-manifest',required=True)
    ap.add_argument('--dedup-root',required=True);ap.add_argument('--policy');ap.add_argument('--output',required=True);a=ap.parse_args()
    try:
        r=assess(a.sources_manifest,a.dedup_root,a.policy);save_report(r,a.output)
        print(json.dumps(dict(readiness=r['readiness'],source_counts=r['source_counts'],checks=r['checks']),ensure_ascii=False))
        sys.exit({'passed':0,'pending':2,'failed':1}[r['readiness']])
    except (ValueError,OSError,KeyError) as exc:
        print('READINESS ERROR:',str(exc),file=sys.stderr);sys.exit(1)
