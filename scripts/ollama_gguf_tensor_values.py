"""Exact F32 HF/GGUF tensor comparison, separate from inference validation.

Caller must authenticate the HF package, GGUF and reader, check structure and
resource admission, and reauthenticate inputs after this read-only operation.
"""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from safetensors import safe_open


def compare_values(reader, hf_dir, config, rows_per_chunk=128):
    if rows_per_chunk < 1:
        raise ValueError('Invalid comparison chunk size')
    files = sorted(Path(hf_dir).glob('*.safetensors'))
    if not files or any(p.is_symlink() for p in files):
        raise ValueError('Missing or symlinked HF weights')
    mapping = {'lm_head.weight': 'output.weight', 'model.embed_tokens.weight': 'token_embd.weight',
               'model.norm.weight': 'output_norm.weight'}
    suffixes = {'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
        'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
        'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight'}
    for i in range(config['num_hidden_layers']):
        mapping.update({f'model.layers.{i}.{k}': f'blk.{i}.{v}' for k, v in suffixes.items()})
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors) or set(tensors) != set(mapping.values()):
        raise ValueError('Unexpected GGUF tensor inventory')
    seen = set()
    with ExitStack() as stack:
        for path in files:
            f = stack.enter_context(safe_open(path, framework='np'))
            for key in f.keys():
                if key in seen or key not in mapping:
                    raise ValueError('Unexpected or duplicate HF tensor: ' + key)
                seen.add(key)
                view = f.get_slice(key)
                out = tensors[mapping[key]].data
                shape = tuple(view.get_shape())
                if shape != out.shape or out.dtype != np.float32:
                    raise ValueError('Shape or dtype differs: ' + key)
                heads = (config['num_attention_heads'] if key.endswith('q_proj.weight') else
                         config['num_key_value_heads'] if key.endswith('k_proj.weight') else None)
                if heads:
                    if len(shape) != 2 or shape[0] % (2 * heads):
                        raise ValueError('Invalid Q/K shape: ' + key)
                    # Compare one head at a time to bound memory and preserve RoPE layout.
                    height = shape[0] // heads
                    for h in range(heads):
                        a = np.asarray(view[h*height:(h+1)*height])
                        a = a.reshape(2, height//2, shape[1]).swapaxes(0, 1).reshape(height, shape[1])
                        if a.dtype != np.float32 or not np.array_equal(a, out[h*height:(h+1)*height]):
                            raise ValueError('Tensor values differ: ' + key)
                else:
                    for start in range(0, shape[0], rows_per_chunk):
                        stop = min(start + rows_per_chunk, shape[0])
                        a = np.asarray(view[start:stop])
                        if a.dtype != np.float32 or not np.array_equal(a, out[start:stop]):
                            raise ValueError('Tensor values differ: ' + key)
    if seen != set(mapping):
        raise ValueError('HF tensor inventory incomplete')
    return {'status': 'exact_f32_tensor_values_only', 'tensor_count': len(seen),
            'logits_verified': False, 'ollama_verified': False}
