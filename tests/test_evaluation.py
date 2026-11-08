import json
import math
from pathlib import Path
import unittest
import torch
from eval.perplexity import scoring_windows, NLLAccumulator, evaluate_sequences
from eval.generation import generate

CASES={x['id']:x for x in json.loads((Path(__file__).resolve().parents[1]/'docs/reviews/frankenstallm/acceptance_cases.json').read_text())['cases']}


class UniformModel(torch.nn.Module):
    def forward(self, ids):
        return torch.zeros((*ids.shape, 16), device=ids.device)


class EvalTest(unittest.TestCase):
    def test_E01_unique_target_coverage(self):
        c=CASES['E01'];args=c['inputs']
        windows=list(scoring_windows(args['tokens'],args['seq_len'],args['stride']))
        positions=[p for w in windows for p in w.scored_positions]
        self.assertEqual(positions,c['expected']['scored_target_positions'])
        for w in windows:
            self.assertLessEqual(len(w.input_ids),args['seq_len'])
            actual=[p for p,t in zip(w.target_positions,w.targets.tolist()) if t!=-100]
            self.assertEqual(actual,w.scored_positions)
        for n in range(2,25):
            for seq in range(1,10):
                for stride in range(1,seq+1):
                    self.assertEqual([p for w in scoring_windows(list(range(n)),seq,stride) for p in w.scored_positions],list(range(1,n)))

    def test_E02_token_weighted_aggregation(self):
        c=CASES['E02'];acc=NLLAccumulator()
        for b in c['inputs']['batches']:acc.add(b['nll_sum'],b['valid_tokens'])
        result=acc.result()
        for key in ['nll_sum','valid_tokens','mean_nll']:self.assertEqual(result[key],c['expected'][key])

    def test_E03_no_target_is_error(self):
        for seq in CASES['E03']['inputs']['token_sequences']:
            self.assertEqual(list(scoring_windows(seq,8,2)),[])
            with self.assertRaisesRegex(ValueError,'no next-token targets'):evaluate_sequences(UniformModel(),[seq],8,2)

    def test_model_eval_loss_and_training_state(self):
        m=UniformModel();m.train()
        result=evaluate_sequences(m,[list(range(13))],8,2)
        self.assertEqual(result['valid_tokens'],12)
        self.assertAlmostEqual(result['mean_nll'],math.log(16),places=5)
        self.assertTrue(m.training)
        with self.assertRaises(ValueError):evaluate_sequences(m,[[100,100]],8,2)
        self.assertTrue(m.training)

    def test_actual_native_model_cpu_interface(self):
        from model.model import Transformer, ModelConfig
        model=Transformer(ModelConfig(vocab_size=16,n_layers=1,dim=16,n_heads=2,n_kv_heads=1,ffn_dim=32,max_seq_len=8))
        report=evaluate_sequences(model,[list(range(9))],8,2)
        self.assertEqual(report['valid_tokens'],8)
        self.assertTrue(math.isfinite(report['mean_nll']))
        sample=generate(model,[1,2],eos_id=15,context_length=8,max_new_tokens=3)
        self.assertLessEqual(len(sample['generated_ids']),3)

    def test_generation_seed_eos_and_state(self):
        m=UniformModel();m.train()
        a=generate(m,[1,2],eos_id=15,context_length=4,max_new_tokens=10,seed=7,temperature=1)
        b=generate(m,[1,2],eos_id=15,context_length=4,max_new_tokens=10,seed=7,temperature=1)
        self.assertEqual(a,b);self.assertTrue(m.training)
        eos=generate(m,[1],eos_id=0,context_length=4,max_new_tokens=10)
        self.assertEqual(eos['generated_ids'],[0]);self.assertTrue(eos['stopped_at_eos'])
