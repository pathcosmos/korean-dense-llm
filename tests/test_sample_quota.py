import json,tempfile,unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from data.sample_snapshot import sample,load_samples
class SampleQuotaTest(unittest.TestCase):
    def test_small_tail_capacity_is_redistributed_without_exceeding_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'rows.parquet';pq.write_table(pa.table({'text':['document '+str(i) for i in range(110)]}),f,row_group_size=100);Path(str(f)+'.done').touch()
            frame=p/'frame.json';frame.write_text(json.dumps(dict(schema_version=1,sources={'s':dict(kind='web',revision='test',files=[str(f)])})))
            for seed in [1,2]:
                r=sample(frame,p/str(seed),per_source=100,max_files_per_source=1,row_groups_per_file=2,max_row_groups=2,max_read_bytes=100000,max_bytes=100000,seed=seed)
                self.assertEqual(r['sample_status'],'complete_for_defined_frame');self.assertEqual(r['stats']['s']['documents'],100);self.assertLessEqual(r['row_groups_read'],2)
                _,rows=load_samples(p/str(seed)/'manifest.json');self.assertEqual(len({x['row'] for x in rows}),100)
