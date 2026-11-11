import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file
from scripts.ollama_gguf_tensor_values import compare_values


class ValueTests(unittest.TestCase):
    def test_permutation_partial_chunk_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            suffixes = {'input_layernorm.weight': ('attn_norm.weight', (4,)),
                'post_attention_layernorm.weight': ('ffn_norm.weight', (4,)),
                'self_attn.q_proj.weight': ('attn_q.weight', (4,4)),
                'self_attn.k_proj.weight': ('attn_k.weight', (2,4)),
                'self_attn.v_proj.weight': ('attn_v.weight', (2,4)),
                'self_attn.o_proj.weight': ('attn_output.weight', (4,4)),
                'mlp.gate_proj.weight': ('ffn_gate.weight', (8,4)),
                'mlp.up_proj.weight': ('ffn_up.weight', (8,4)),
                'mlp.down_proj.weight': ('ffn_down.weight', (4,8))}
            pairs = {'lm_head.weight': ('output.weight', (3,4)),
                'model.embed_tokens.weight': ('token_embd.weight', (3,4)),
                'model.norm.weight': ('output_norm.weight', (4,))}
            pairs.update({f'model.layers.0.{k}': (f'blk.0.{v}', s) for k,(v,s) in suffixes.items()})
            weights = {k: np.arange(np.prod(s), dtype=np.float32).reshape(s) for k,(_,s) in pairs.items()}
            save_file(weights, str(Path(tmp) / 'model.safetensors'))
            tensors = []
            for k,(name,shape) in pairs.items():
                arr = weights[k].copy()
                h = 2 if k.endswith('q_proj.weight') else 1 if k.endswith('k_proj.weight') else None
                if h: arr = arr.reshape(h,2,shape[0]//h//2,shape[1]).swapaxes(1,2).reshape(shape)
                tensors.append(SimpleNamespace(name=name, data=arr))
            r = SimpleNamespace(tensors=tensors)
            cfg = {'num_hidden_layers':1, 'num_attention_heads':2, 'num_key_value_heads':1}
            self.assertEqual(compare_values(r, tmp, cfg, rows_per_chunk=3)['tensor_count'], 12)
            next(t for t in tensors if t.name == 'blk.0.attn_q.weight').data[0,0] += 1
            with self.assertRaisesRegex(ValueError, 'Tensor values differ'):
                compare_values(r, tmp, cfg, rows_per_chunk=3)


if __name__ == '__main__':
    unittest.main()
