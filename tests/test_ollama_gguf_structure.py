import unittest
from types import SimpleNamespace

from scripts.ollama_gguf_structure import verify_structure


class StructureTests(unittest.TestCase):
    def test_missing_head_and_malformed_qkv_are_rejected(self):
        c = dict(model_type='llama', tie_word_embeddings=False, attention_bias=False,
                 mlp_bias=False, rope_scaling=None, hidden_size=4, num_hidden_layers=1,
                 num_attention_heads=2, num_key_value_heads=1, intermediate_size=8,
                 vocab_size=3, max_position_embeddings=16, rope_theta=10000.0,
                 rms_norm_eps=1e-5, bos_token_id=0, eos_token_id=1, pad_token_id=2)
        fields = {'general.architecture': 'llama', 'llama.embedding_length': 4,
            'llama.block_count': 1, 'llama.attention.head_count': 2,
            'llama.attention.head_count_kv': 1, 'llama.feed_forward_length': 8,
            'llama.context_length': 16, 'llama.rope.freq_base': 10000.0,
            'llama.attention.layer_norm_rms_epsilon': 1e-5, 'tokenizer.ggml.bos_token_id': 0,
            'tokenizer.ggml.eos_token_id': 1, 'tokenizer.ggml.padding_token_id': 2,
            'tokenizer.ggml.tokens': ['a', 'b', 'c']}
        shapes = {'token_embd.weight': (4,3), 'output.weight': (4,3), 'output_norm.weight': (4,),
            'blk.0.attn_norm.weight': (4,), 'blk.0.ffn_norm.weight': (4,),
            'blk.0.attn_q.weight': (4,4), 'blk.0.attn_k.weight': (4,2),
            'blk.0.attn_v.weight': (4,2), 'blk.0.attn_output.weight': (4,4),
            'blk.0.ffn_gate.weight': (4,8), 'blk.0.ffn_up.weight': (4,8),
            'blk.0.ffn_down.weight': (8,4)}
        tensors = [SimpleNamespace(name=k, shape=v, tensor_type=0) for k, v in shapes.items()]
        reader = SimpleNamespace(tensors=tensors,
            get_field=lambda name: SimpleNamespace(contents=lambda: fields[name]) if name in fields else None)
        self.assertEqual(verify_structure(reader, c)['tensor_count'], 12)
        reader.tensors = [t for t in tensors if t.name != 'output.weight']
        with self.assertRaisesRegex(ValueError, 'inventory'):
            verify_structure(reader, c)
        reader.tensors = list(tensors)
        reader.tensors[6] = SimpleNamespace(name='blk.0.attn_k.weight', shape=(4,4), tensor_type=0)
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            verify_structure(reader, c)


if __name__ == '__main__':
    unittest.main()
