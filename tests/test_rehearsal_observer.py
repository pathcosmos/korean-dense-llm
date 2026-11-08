import copy
import json
from pathlib import Path
import tempfile
import unittest

from common.data_contracts import atomic_json, sha256_file
from scripts.observe_rehearsal import read_rows, summarize, producer_certificate, progress_state


def train(step,rate):
    return dict(step=step,loss=4.,gnorm=.2,lr=.0006,tok_s=rate,tokens_b=step*2097152/1e9)


class ObserverTest(unittest.TestCase):
    def test_stable_rate_excludes_resumed_and_post_evaluation_intervals(self):
        result=summarize([train(10,700000),train(20,360000),train(30,370000),train(260,200000),train(510,100000)])
        self.assertEqual(result['stable_median_tokens_s'],365000)
        self.assertEqual(result['last_train']['step'],510)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            summarize([dict(train(30,360000),loss=float('nan'))])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'log';p.write_text(json.dumps(train(20,360000))+'\n{"step":')
            self.assertEqual(len(read_rows(p)),1)
            p.write_text('{broken}\n'+json.dumps(train(20,360000)))
            with self.assertRaises(json.JSONDecodeError):read_rows(p)

    def test_weighted_probe_counts_and_true_progress_stall(self):
        sources={s:dict(valid_tokens=262144,nll_sum=262144.,holdout_loss=1.) for s in ['wiki','fw','hplt','en','math','code']}
        cats={s:dict(valid_tokens=786432 if s=='ko' else 262144,nll_sum=786432. if s=='ko' else 262144.,holdout_loss=1.) for s in ['ko','en','math','code']}
        evaluation=dict(step=250,valid_tokens=1572864,nll_sum=1572864.,holdout_loss=1.,by_source=sources,by_category=cats)
        summary=summarize([train(250,360000),evaluation]);summary['checkpoints']={}
        state,history=progress_state(summary,{},100,source_phase='running',live=True)
        self.assertEqual(state,'running')
        state,history=progress_state(summary,history,1400,source_phase='running',live=True)
        self.assertEqual(state,'suspected_stall')  # Parent heartbeat alone does not reset this.
        summary['last_train']=train(260,360000)
        state,history=progress_state(summary,history,1401,source_phase='running',live=True)
        self.assertEqual(state,'running')
        self.assertEqual(progress_state(summary,history,1402,source_phase='error',live=True)[0],'error')
        self.assertEqual(progress_state(summary,history,1402,source_phase='running',live=False)[0],'unexpected_exit')
        self.assertEqual(progress_state(summary,history,1402,source_phase='complete',live=False)[0],'completed_waiting_proof')
        bad=copy.deepcopy(evaluation);bad['by_category']['ko']['valid_tokens']-=1
        with self.assertRaisesRegex(ValueError,'target sum'):summarize([bad])

    def test_producer_certificate_metadata_binding_without_payload_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'training/ckpt_0000500';checkpoint.mkdir(parents=True)
            atomic_json(checkpoint/'meta.json',dict(step=500,training_contract=dict(id='fixture')))
            for name in ['.complete','.metadata',*[f'rng_rank{r}.pt' for r in range(8)]]:(checkpoint/name).touch()
            records=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256_file(p)) for p in checkpoint.iterdir()]
            proof=root/'ckpt_0000500.json';atomic_json(proof,dict(status='passed_checkpoint_storage',step=500,
                checkpoint=str(checkpoint),training_contract_id='fixture',bytes=sum(r['bytes'] for r in records),files=records))
            self.assertEqual(producer_certificate(root,proof,'fixture')['step'],500)
            atomic_json(checkpoint/'meta.json',dict(step=501,training_contract=dict(id='fixture')))
            with self.assertRaisesRegex(ValueError,'metadata changed'):producer_certificate(root,proof,'fixture')


if __name__=='__main__':unittest.main()
