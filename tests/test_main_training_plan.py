import unittest

from scripts.plan_main_training import GIB, forecast, geometry


class MainPlanTest(unittest.TestCase):
    def settings(self, **kw):
        return dict(dict(seq_len=4096, micro_bsz=4, global_tokens=2097152,
                         max_steps=57000, ckpt_every=250, eval_every=250,
                         keep_ckpts=2, save_final=True), **kw)

    def test_exact_geometry_rejects_silent_rounding(self):
        report = geometry(self.settings())
        self.assertEqual(report['accumulation'],16)
        self.assertEqual(report['target_tokens'],119537664000)
        with self.assertRaisesRegex(ValueError,'divisible'):
            geometry(self.settings(micro_bsz=6))
        changed = geometry(self.settings(micro_bsz=6, global_tokens=1966080))
        self.assertEqual(changed['accumulation'],10)
        self.assertNotEqual(changed['target_tokens'],report['target_tokens'])

    def test_storage_coexistence_and_boundary_require_full_future_reserve(self):
        c = self.settings(); checkpoint = 38666917916
        f = forecast(c,[138511.05573562847,138461.70009862378],checkpoint,500*GIB,140*GIB)
        self.assertEqual(f['checkpoints_written'],228)
        self.assertEqual(f['retained_checkpoint_count'],2)
        self.assertEqual(f['peak_checkpoint_count_including_staging'],3)
        required = f['proposed_new_peak_and_reserve_bytes']
        self.assertTrue(forecast(c,[138500],checkpoint,required,140*GIB)['fits_with_other_future_peaks'])
        self.assertFalse(forecast(c,[138500],checkpoint,required-1,140*GIB)['fits_with_other_future_peaks'])
        preserve = forecast(self.settings(keep_ckpts=57001),[138500],checkpoint,500*GIB,140*GIB)
        self.assertEqual(preserve['peak_checkpoint_count_including_staging'],228)
        self.assertFalse(preserve['fits_with_other_future_peaks'])
        final_tail = forecast(self.settings(max_steps=501),[138500],checkpoint,500*GIB)
        self.assertEqual(final_tail['checkpoints_written'],3)
        with self.assertRaises(ValueError):
            forecast(c,[float('nan')],checkpoint,500*GIB)


if __name__ == '__main__':
    unittest.main()
