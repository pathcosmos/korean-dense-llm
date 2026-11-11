"""Read-only GGUF structure check for this project's untied Llama export.

The caller supplies an already authenticated GGUFReader and HF config. This
checks metadata and tensor shapes, not tensor values, tokenizer semantics,
numeric agreement, or runtime operation.
"""
import math


def verify_structure(reader, config):
    c = config
    if (c.get('model_type') != 'llama' or c.get('tie_word_embeddings') is not False
            or c.get('attention_bias') is not False or c.get('mlp_bias') is not False
            or c.get('rope_scaling') is not None):
        raise ValueError('Unsupported HF model configuration')
    d, n, heads, kv, ff, vocab = (int(c[k]) for k in ('hidden_size', 'num_hidden_layers',
        'num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'vocab_size'))
    if min(d, n, heads, kv, ff, vocab) <= 0 or d % heads or heads % kv:
        raise ValueError('Invalid model dimensions')
    expected_meta = {'general.architecture': 'llama', 'llama.embedding_length': d,
        'llama.block_count': n, 'llama.attention.head_count': heads,
        'llama.attention.head_count_kv': kv, 'llama.feed_forward_length': ff,
        'llama.context_length': c['max_position_embeddings'], 'llama.rope.freq_base': c['rope_theta'],
        'llama.attention.layer_norm_rms_epsilon': c['rms_norm_eps'],
        'tokenizer.ggml.bos_token_id': c['bos_token_id'],
        'tokenizer.ggml.eos_token_id': c['eos_token_id'],
        'tokenizer.ggml.padding_token_id': c['pad_token_id']}
    for name, expected in expected_meta.items():
        field = reader.get_field(name)
        value = field.contents() if field is not None else None
        ok = math.isclose(value, expected, rel_tol=1e-6, abs_tol=0) if isinstance(expected, float) and isinstance(value, (float, int)) else value == expected
        if not ok:
            raise ValueError('GGUF metadata differs: ' + name)
    tokens = reader.get_field('tokenizer.ggml.tokens')
    if tokens is None or len(tokens.contents()) != vocab:
        raise ValueError('GGUF vocabulary size differs')
    expected = {'token_embd.weight': (d, vocab), 'output.weight': (d, vocab),
                'output_norm.weight': (d,)}
    for i in range(n):
        prefix = f'blk.{i}.'
        expected.update({prefix + name: shape for name, shape in {
            'attn_norm.weight': (d,), 'ffn_norm.weight': (d,),
            'attn_q.weight': (d, d), 'attn_k.weight': (d, d * kv // heads),
            'attn_v.weight': (d, d * kv // heads), 'attn_output.weight': (d, d),
            'ffn_gate.weight': (d, ff), 'ffn_up.weight': (d, ff),
            'ffn_down.weight': (ff, d)}.items()})
    tensors = list(reader.tensors)
    actual = {t.name: (tuple(int(v) for v in t.shape), int(t.tensor_type)) for t in tensors}
    if len(tensors) != len(actual) or set(actual) != set(expected):
        raise ValueError('GGUF tensor inventory differs')
    for name, shape in expected.items():
        if actual[name] != (shape, 0):  # GGML F32
            raise ValueError('GGUF tensor shape or dtype differs: ' + name)
    return {'status': 'structure_only', 'tensor_count': len(actual),
            'numerical_equivalence_verified': False, 'ollama_verified': False}
