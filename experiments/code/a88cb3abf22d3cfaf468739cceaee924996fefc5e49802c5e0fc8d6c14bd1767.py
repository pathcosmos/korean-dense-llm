from pathlib import Path
import json,os,sys
from common.data_contracts import atomic_json,sha256_file
from data.sample_snapshot import sample
from data.probe_text_similarity import run
p=Path("runs/f2_sampling_v2");(p/"pid").write_text(str(os.getpid()))
def state(phase):atomic_json(p/"state.json",dict(phase=phase,pid=os.getpid()))
try:
 state("sampling")
 r=sample("runs/f2_execution_v1/frame.json",p/"samples",per_source=2000,max_bytes=128*1024**2,max_read_bytes=2*1024**3,max_row_groups=24,max_files_per_source=4,row_groups_per_file=2,max_seconds=300)
 print("SAMPLE",r["sample_status"],r["stats"],flush=True)
 if r["sample_status"]!="complete_for_defined_frame":raise ValueError("Sampling target not met")
 summaries=[]
 for t in [.8,.9,.95]:
  state("similarity_"+str(t));report=run(p/"samples/manifest.json",p/("similarity_"+str(t)+".json"),threshold=t,max_candidates=50000,max_seconds=180,max_view_chars=2048);summaries.append(dict(threshold=t,status=report["status"],verified=report["verified_candidate_pairs"],matches=report["matched_view_pairs"]));print(summaries[-1],flush=True)
 atomic_json(p/"summary.json",dict(status="diagnostic_complete",sample_manifest_sha256=sha256_file(p/"samples/manifest.json"),similarity=summaries));atomic_json(p/".done",dict(summary_sha256=sha256_file(p/"summary.json")));state("complete")
except Exception:
 state("error");raise
