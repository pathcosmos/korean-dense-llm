"""Resumable full-column candidate audit. Exit 2 means a resumable budget pause."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.source_audit import run,AuditBudgetReached

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--pilot',action='store_true');ap.add_argument('--batch-size',type=int,default=1024)
    ap.add_argument('--max-seconds',type=int,default=7200);ap.add_argument('--max-rss-gib',type=float,default=16)
    ap.add_argument('--min-free-gib',type=int,default=200);ap.add_argument('--events',default='logs/pipeline_events.jsonl');a=ap.parse_args()
    try:
        r=run(a.manifest,a.output_dir,pilot=a.pilot,batch_size=a.batch_size,max_seconds=a.max_seconds,
              max_rss_gib=a.max_rss_gib,min_free_gib=a.min_free_gib,event_log=a.events)
        print(json.dumps({k:r[k] for k in ['scope','structural_status','files_verified','rows_verified','input_bytes','elapsed_this_run_seconds','max_rss_kib']}),flush=True)
    except AuditBudgetReached as exc:print(str(exc),flush=True);sys.exit(2)
