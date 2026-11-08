from pathlib import Path
import subprocess,time,os,json
from datetime import datetime,timezone
from common.data_contracts import atomic_json,sha256_file
p=Path("runs/f3_finalize_v1");(p/"pid").write_text(str(os.getpid()));ready=Path("data/datasets/final_v1.ready.json")
def state(phase,**kw):atomic_json(p/"state.json",dict(phase=phase,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw))
state("running")
try:
 with (p/"finalize.log").open("w") as log:child=subprocess.Popen(["timeout","--kill-after=30s","3600s","python","scripts/finalize_source_manifest.py","--candidate","data/datasets/final_v1.candidate.json","--audit","runs/source_audit_full_v1/report.json","--policy","runs/f2_execution_v1/final_policy.json","--dedup-root","data/dedup/exact_v1","--output",str(ready)],stdout=log,stderr=subprocess.STDOUT)
 while child.poll() is None:state("running",child_pid=child.pid);time.sleep(5)
 if child.returncode:raise RuntimeError("Finalizer exit "+str(child.returncode))
 doc=json.loads(ready.read_text());marker=json.loads(Path(str(ready)+".done").read_text());assert marker["sha256"]==sha256_file(ready) and doc["status"]=="ready"
 atomic_json(p/"report.json",dict(status="passed",ready_manifest=str(ready),ready_manifest_sha256=sha256_file(ready),content_inventory_id=doc["content_inventory_id"],scope="Audited input and declared scoped policies only. No actual corpus split, tokenizer or model evaluation approval."));atomic_json(p/".done",dict(report_sha256=sha256_file(p/"report.json")));state("complete")
except Exception:
 state("error");raise
