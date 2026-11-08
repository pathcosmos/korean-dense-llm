import os,json,tempfile
from pathlib import Path
import pyarrow as pa
p=Path("runs/resource_policy_validation_v1/result.json")
r=dict(omp=os.environ["OMP_NUM_THREADS"],rayon=os.environ["RAYON_NUM_THREADS"],numexpr=os.environ["NUMEXPR_NUM_THREADS"],arrow_cpu=pa.cpu_count(),arrow_io=pa.io_thread_count(),temp_dir=tempfile.gettempdir(),sqlite_tmp=os.environ["SQLITE_TMPDIR"])
p.write_text(json.dumps(r));print(json.dumps(r))
