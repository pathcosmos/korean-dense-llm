"""Llama-style dense decoder: RMSNorm(pre) + RoPE + GQA + SwiGLU. Plain PyTorch."""
import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    n_layers: int = 28
    dim: int = 3072
    n_heads: int = 24
    n_kv_heads: int = 8
    ffn_dim: int = 8192
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False


PRESETS = {
    "micro": dict(n_layers=2, dim=64, n_heads=4, n_kv_heads=2, ffn_dim=128),       # bounded correctness tests
    "tiny": dict(n_layers=6, dim=512, n_heads=8, n_kv_heads=4, ffn_dim=1408),       # ~60M smoke test
    "100m": dict(n_layers=12, dim=768, n_heads=12, n_kv_heads=4, ffn_dim=2048),
    "1b":   dict(n_layers=18, dim=2048, n_heads=16, n_kv_heads=8, ffn_dim=5632, tie_embeddings=True),
    "3b":   dict(n_layers=28, dim=3072, n_heads=24, n_kv_heads=8, ffn_dim=8192),
}


def get_config(name, **kw):
    return ModelConfig(**{**PRESETS[name], **kw})


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def precompute_rope(head_dim, seq_len, theta, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return torch.cos(ang), torch.sin(ang)  # (T, hd/2)


def apply_rope(x, cos, sin):  # x: (B,T,H,hd)
    T = x.shape[1]
    cos, sin = cos[:T, None, :], sin[:T, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_heads, c.n_kv_heads, c.dim // c.n_heads
        self.wq = nn.Linear(c.dim, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(c.dim, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(c.dim, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, c.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd)
        k = self.wk(x).view(B, T, self.nkv, self.hd)
        v = self.wv(x).view(B, T, self.nkv, self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w3 = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.w2 = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(c.dim, c.norm_eps), RMSNorm(c.dim, c.norm_eps)
        self.attn, self.ffn = Attention(c), FeedForward(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim, c.norm_eps)
        self.out = nn.Linear(c.dim, c.vocab_size, bias=False)
        if c.tie_embeddings: self.out.weight = self.tok.weight
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.init_weights()

    def init_weights(self):
        c = self.c
        for m in self.modules():
            if isinstance(m, RMSNorm): nn.init.ones_(m.weight)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
        # GPT-2 style: scale residual-output projections by 1/sqrt(2*L)
        for b in self.blocks:
            nn.init.normal_(b.attn.wo.weight, std=0.02 / math.sqrt(2 * c.n_layers))
            nn.init.normal_(b.ffn.w2.weight, std=0.02 / math.sqrt(2 * c.n_layers))
        self.cos, self.sin = precompute_rope(c.dim // c.n_heads, c.max_seq_len, c.rope_theta)

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        cos, sin = self.cos.to(x.device), self.sin.to(x.device)
        for b in self.blocks: x = b(x, cos, sin)
        logits = self.out(self.norm(x))
        if targets is None: return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        return n  # tied weights counted once by parameters()

    def flops_per_token(self, seq_len):
        c = self.c
        n_nonemb = self.num_params() - self.tok.weight.numel() * (1 if c.tie_embeddings else 2) + self.out.weight.numel()
        return 6 * n_nonemb + 12 * c.n_layers * c.dim * seq_len  # matmul params (incl. output head) + attention
