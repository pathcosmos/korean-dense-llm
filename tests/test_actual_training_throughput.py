import copy
import unittest
from scripts.benchmark_actual_training import summarize,choose


class ActualThroughputTest(unittest.TestCase):
    def rows(self):
        return [dict(step=i,loss=11-i*.01,gnorm=1.,mfu=.1,tok_s=100. if i%2 else 200.,
                     max_allocated_gib=10.,max_reserved_gib=12.,tokens_b=i*2097152/1e9) for i in range(1,9)]

    def test_equal_token_steps_aggregate_by_total_time_not_mean_rates(self):
        r=summarize(self.rows())
        self.assertAlmostEqual(r['aggregate_tok_s'],400/3)
        self.assertEqual(r['timed_tokens'],6*2097152);self.assertEqual(r['timed_steps'],6)
        self.assertEqual(r['median_tok_s'],150.)

    def test_missing_nonfinite_and_wrong_geometry_never_certify_throughput(self):
        for mutate in [lambda r:r.pop(),lambda r:r[3].update(step=3),lambda r:r[3].update(loss=float('nan')),
                       lambda r:r[3].update(tok_s=0),lambda r:r[3].update(tokens_b=1.),lambda r:r[3].update(max_reserved_gib=9.)]:
            rows=self.rows();mutate(rows)
            with self.assertRaises(ValueError):summarize(rows)

    def test_oom_and_fastest_without_checkpoint_headroom_not_selected(self):
        cases=[dict(status='cuda_oom',micro_bsz=32),dict(status='passed',micro_bsz=16,aggregate_tok_s=200.,max_reserved_gib=130.),
               dict(status='passed',micro_bsz=8,aggregate_tok_s=150.,max_reserved_gib=100.),dict(status='passed',micro_bsz=4,aggregate_tok_s=147.,max_reserved_gib=70.)]
        self.assertEqual(choose(cases,140.)['micro_bsz'],4)
        cases[2]['aggregate_tok_s']=160.
        self.assertEqual(choose(cases,140.)['micro_bsz'],8)
        with self.assertRaises(ValueError):choose(cases[:2],140.)
