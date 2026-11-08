from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from common.data_contracts import atomic_json,sha256_file
from scripts.assemble_pretrain_review import final_checks,pending_gates


class ReviewAssemblyTest(unittest.TestCase):
    def fixture(self):
        rehearsal=dict(status='passed_actual_rehearsal_training',counts=dict(committed_tokens=10485760000),
                       curves=dict(trained_tokens=10485760000,loss_decreased=True))
        native=dict(rehearsal_status=rehearsal['status'],languages={k:dict(valid_tokens=n,mean_nll=3.,perplexity_overflow=False) for k,n in [('ko',663905),('en',705634)]})
        monitored=dict(counts=dict(committed_tokens=8388608))
        validation=dict(results=[dict(model=m,checkpoint_comparison=dict(exact=True,max_abs=0.),trajectory=dict(loss_max_abs=0.)) for m in ['1b','3b']])
        return rehearsal,native,monitored,validation

    def test_partial_training_and_diagnostic_native_cannot_satisfy_full_goal(self):
        values=self.fixture();self.assertEqual(final_checks(*values),[])
        bad=copy.deepcopy(values);bad[0]['counts']['committed_tokens']=1048576000
        with self.assertRaisesRegex(ValueError,'not reached'):final_checks(*bad)
        bad=copy.deepcopy(values);bad[1]['languages']['en']['valid_tokens']-=1
        with self.assertRaisesRegex(ValueError,'coverage'):final_checks(*bad)
        bad=copy.deepcopy(values);bad[3]['results'][1]['checkpoint_comparison']['exact']=False
        with self.assertRaisesRegex(ValueError,'resume proof'):final_checks(*bad)
        bad=copy.deepcopy(values);bad[0]['curves']['loss_decreased']=False
        self.assertTrue(final_checks(*bad))  # Assembled evidence is not automatic quality acceptance.

    def test_only_content_bound_complete_artifacts_clear_wait_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[Path(tmp)/str(i) for i in range(3)]
            for r in roots:r.mkdir()
            a=Namespace(rehearsal_run=str(roots[0]),final_native_run=str(roots[1]),monitored_run=str(roots[2]))
            self.assertEqual(len(pending_gates(a)),3)
            statuses=['passed_actual_rehearsal_training','passed_actual_trained_native_evaluation','passed_actual_monitored_3b']
            for r,status in zip(roots,statuses):
                atomic_json(r/'report.json',dict(status=status));atomic_json(r/'contract.json',dict(code={}))
                atomic_json(r/'.done',dict(report_sha256=sha256_file(r/'report.json'),contract_sha256=sha256_file(r/'contract.json')))
            self.assertFalse(pending_gates(a))
            (roots[0]/'report.json').write_text('{}')
            with self.assertRaises((ValueError,KeyError)):pending_gates(a)


if __name__=='__main__':unittest.main()
