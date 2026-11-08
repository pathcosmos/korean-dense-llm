import json
from pathlib import Path
import tempfile
import unittest

from scripts.observe_main_training import summarize, progress_state, rows_from_trace


CFG=dict(max_steps=57000,global_tokens=2097152,log_every=10,eval_every=250,ckpt_every=500)


def train(step, rate=100000.):
    return dict(step=step,loss=4.,gnorm=.2,lr=.0003,tok_s=rate,tokens_b=step*2097152/1e9)


class MainObserverTest(unittest.TestCase):
    def test_true_progress_stall_and_worker_identity(self):
        summary=summarize([train(5010)],CFG)
        args=dict(supervisor_live=True,worker_live=True,phase='running',worker_pid=99)
        state,history=progress_state(summary,{},100,**args)
        self.assertEqual(state,'running')
        self.assertEqual(progress_state(summary,history,1301,**args)[0],'suspected_stall')
        summary=summarize([train(5010),train(5020)],CFG)
        self.assertEqual(progress_state(summary,history,1302,**args)[0],'running')
        self.assertEqual(progress_state(summary,history,1302,**dict(args,worker_live=False))[0],'unexpected_exit')
        self.assertEqual(progress_state(summary,history,1302,**dict(args,phase='error'))[0],'error')
        self.assertEqual(progress_state(summary,history,1302,**dict(args,phase='complete',worker_live=False))[0],'completed_waiting_proof')

    def test_rates_exclude_overhead_and_replay_resets_latest_progress(self):
        rows=[train(s,r) for s,r in [(10,10),(20,100000),(30,110000),(40,120000),(250,90),(260,1),(270,100000),(500,80),(510,2)]]
        result=summarize(rows,CFG)
        self.assertEqual(result['steady_tokens_s'],105000.)
        result=summarize(rows+[train(500,2)],CFG)
        self.assertEqual(result['logged_step'],500)
        self.assertEqual(result['log_step_regressions'],1)
        self.assertIsNone(result['steady_tokens_s'])
        with self.assertRaisesRegex(ValueError,'finite'):summarize([dict(train(10),loss=float('nan'))],CFG)

    def test_partial_append_is_pending_but_complete_bad_line_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'log'
            path.write_text(json.dumps(train(10))+'\n{"step":')
            self.assertEqual(rows_from_trace(path),[train(10)])
            path.write_text(json.dumps(train(10))+'\n{broken}\n')
            with self.assertRaises(json.JSONDecodeError):rows_from_trace(path)


if __name__=='__main__':
    unittest.main()
