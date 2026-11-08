import copy
import json
from pathlib import Path
import tempfile
import unittest

from common.data_contracts import identity
from scripts.audit_main_training import aggregate, audit_trace, Snapshot


def ledger(rank, steps, *, success=True, generations=None, counts=None):
    return dict(rank=rank, success=success, contract_id='contract', optimizer_initializations=1,
                steps=[dict(step=s, source_sequences=counts or {'ko':1, 'en':2}) for s in steps],
                generations=generations or [])


class LedgerAuditTest(unittest.TestCase):
    def arguments(self, **overrides):
        return dict(dict(world=2, through_step=4, target_steps=4, seq_len=8,
                         global_tokens=48, sources=['ko','en'], generation_steps=[], contract_id='contract'), **overrides)

    def test_exact_union_and_replayed_successful_updates_are_separate(self):
        records=[('early',ledger(r,[1,2,3],success=False)) for r in range(2)]
        records += [('resumed',ledger(r,[3,4])) for r in range(2)]
        result=aggregate(records,**self.arguments())
        self.assertEqual(result['committed_tokens'],192)
        self.assertEqual(result['source_tokens'],{'ko':64,'en':128})
        self.assertEqual(result['replayed_tokens_through_prefix'],48)
        self.assertEqual(result['persisted_attempt_tokens_through_prefix'],240)
        self.assertEqual(result['optimizer_initializations_excluded'],4)
        self.assertEqual(result['successful_target_ranks'],[0,1])
        result=aggregate(records,**self.arguments(through_step=2))
        self.assertEqual(result['committed_tokens'],96)
        self.assertEqual(result['persisted_attempt_tokens_beyond_prefix'],144)

    def test_missing_step_rank_and_duplicate_updates_fail(self):
        records=[('main',ledger(r,[1,2,3,4])) for r in range(2)]
        with self.assertRaisesRegex(ValueError,'coverage'):
            aggregate(records[:1],**self.arguments())
        bad=copy.deepcopy(records);bad[1][1]['steps'].pop(1)
        with self.assertRaisesRegex(ValueError,'Noncontiguous'):
            aggregate(bad,**self.arguments())
        bad=copy.deepcopy(records);bad[1][1]['steps'].append(bad[1][1]['steps'][-1])
        with self.assertRaisesRegex(ValueError,'Noncontiguous'):
            aggregate(bad,**self.arguments())
        with self.assertRaisesRegex(ValueError,'Duplicate attempt/rank'):
            aggregate(records+records[:1],**self.arguments())

    def test_replay_counts_contract_and_geometry_must_match(self):
        records=[('main',ledger(r,[1,2,3,4])) for r in range(2)]
        bad=ledger(0,[4],counts={'ko':2,'en':1})
        with self.assertRaisesRegex(ValueError,'Replayed source'):
            aggregate(records+[('resume',bad)],**self.arguments())
        for field,value in [('contract_id','other'),('rank',True)]:
            bad=copy.deepcopy(records);bad[0][1][field]=value
            with self.assertRaises(ValueError):aggregate(bad,**self.arguments())
        bad=copy.deepcopy(records);bad[0][1]['steps'][0]['source_sequences']={'ko':3,'en':1}
        with self.assertRaisesRegex(ValueError,'geometry'):aggregate(bad,**self.arguments())

    def test_full57000_horizon_uses_contracted_generation_schedule(self):
        schedule=[250,2500,5000,*range(10000,55001,5000),57000]
        records=[('main',ledger(r,range(1,57001),generations=[dict(step=s,sha256='a'*64) for s in schedule])) for r in range(2)]
        result=aggregate(records,**self.arguments(through_step=57000,target_steps=57000,generation_steps=schedule))
        self.assertEqual(result['committed_step_rank_pairs'],114000)
        self.assertEqual(result['generation_sha256_by_step'],{str(s):'a'*64 for s in schedule})
        records[1][1]['generations'].pop()
        with self.assertRaisesRegex(ValueError,'Generation rank coverage'):
            aggregate(records,**self.arguments(through_step=57000,target_steps=57000,generation_steps=schedule))

    def test_wrong_rank_generation_and_unscheduled_generation_fail(self):
        records=[('main',ledger(r,[1,2,3,4],generations=[dict(step=4,sha256='a'*64)])) for r in range(2)]
        records[1][1]['generations'][0]['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'agreement'):aggregate(records,**self.arguments(generation_steps=[4]))
        with self.assertRaisesRegex(ValueError,'generation receipt'):aggregate(records,**self.arguments())


class TraceAuditTest(unittest.TestCase):
    def fixture(self):
        config=dict(max_steps=4,global_tokens=48,log_every=1,eval_every=2,seq_len=8,vocab_size=65536)
        probe=dict(groups=[dict(source='ko',category='ko',count=1),dict(source='en',category='en',count=2)],target_tokens=24)
        prompts=[dict(id='p',text='question',category='ko')]
        generation=[dict(id='p',prompt='question',category='ko',generated_ids=[1],text='answer',stopped_on_eos=True)]
        train=[dict(step=s,loss=4.,gnorm=1.,lr=.001,tok_s=100.,tokens_b=s*48/1e9) for s in range(1,5)]
        sources={'ko':dict(valid_tokens=8,nll_sum=16.,holdout_loss=2.),'en':dict(valid_tokens=16,nll_sum=64.,holdout_loss=4.)}
        evaluations=[dict(step=s,contract_id='contract',by_source=copy.deepcopy(sources),by_category=copy.deepcopy(sources),valid_tokens=24,nll_sum=80.,holdout_loss=80/24) for s in [2,4]]
        evaluations[-1]['fixed_generations']=generation
        kw=dict(config=config,through_step=4,contract_id='contract',probe=probe,prompts=prompts,generation_hashes={'4':identity(generation)})
        return train+evaluations,kw

    def test_source_weighted_nll_and_generations_bind_to_rank_ledgers(self):
        rows,kw=self.fixture();result=audit_trace(rows,**kw)
        self.assertEqual(result['evaluation_points'],2)
        bad=copy.deepcopy(rows);bad[-1]['holdout_loss']=3. # Unweighted mean is wrong.
        with self.assertRaisesRegex(ValueError,'weighted NLL'):audit_trace(bad,**kw)
        bad=copy.deepcopy(rows);bad[-1]['fixed_generations'][0]['generated_ids']=[2]
        with self.assertRaisesRegex(ValueError,'receipts'):audit_trace(bad,**kw)

    def test_missing_train_eval_generation_and_nonfinite_cannot_pass(self):
        rows,kw=self.fixture()
        for bad in [rows[1:],rows[:-2]+rows[-1:],rows[:-1]]:
            with self.assertRaisesRegex(ValueError,'Missing'):audit_trace(bad,**kw)
        bad=copy.deepcopy(rows);del bad[-1]['fixed_generations']
        with self.assertRaisesRegex(ValueError,'generation trace'):audit_trace(bad,**kw)
        bad=copy.deepcopy(rows);bad[0]['gnorm']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):audit_trace(bad,**kw)

    def test_replay_timing_can_differ_but_deterministic_values_cannot(self):
        rows,kw=self.fixture()
        replay=dict(rows[0],tok_s=50.)
        self.assertEqual(audit_trace(rows+[replay],**kw)['repeated_train_records'],1)
        replay['loss']=4.01
        with self.assertRaisesRegex(ValueError,'replay trace'):audit_trace(rows+[replay],**kw)

    def test_snapshot_rejects_incomplete_trace_and_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace';path.write_text('{"step":1}\n{"step":')
            with self.assertRaisesRegex(ValueError,'Incomplete trace'):Snapshot().read(path,lines=True,require_newline=True)
            path.write_text('{"step":1}\n');snapshot=Snapshot();snapshot.read(path,lines=True)
            path.write_text('{"step":2}\n')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):snapshot.unchanged()


if __name__=='__main__':
    unittest.main()
