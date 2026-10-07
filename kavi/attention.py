"""Causal multi-head self-attention, with optional rotary position embeddings (RoPE).

docs/05-attention.md derives everything here. Shapes:
    B batch, T sequence length, C model width, H heads, D = C/H head width.
"""
import math

import numpy as np

from . import backend as B
from .layers import Dropout, Linear
from .module import Module


def softmax(x, axis=-1):
    """Numerically stable softmax: subtracting the row max changes nothing mathematically
    (it cancels in the ratio) but keeps exp() from overflowing."""
    e = B.xp.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def rotate_half(x):
    """[x1, x2] -> [-x2, x1] on the last axis (a 90° rotation of every (x1_i, x2_i) pair)."""
    half = x.shape[-1] // 2
    return B.xp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class RoPE:
    """Rotary position embedding (Su et al., 2021).          docs/05-attention.md §RoPE

    Instead of *adding* a position vector to the token, rotate q and k: feature pair i at
    position t is rotated by angle t·θ_i, with θ_i = base^(-2i/D). Then q_m·k_n depends
    only on (m - n): attention sees *relative* position for free, with no parameters.

        forward:  y  = x ⊙ cos + rotate_half(x) ⊙ sin
        backward: dx = dy ⊙ cos - rotate_half(dy ⊙ sin)
    (A rotation's inverse is its transpose; rotate_half is antisymmetric, Rᵀ = -R.)
    """

    def __init__(self, head_dim, max_len, base=10000.0):
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
        angles = np.outer(np.arange(max_len, dtype=np.float64), inv_freq)   # (T, D/2)
        angles = np.concatenate([angles, angles], axis=-1)                   # (T, D)
        self.cos = B.asarray(np.cos(angles))
        self.sin = B.asarray(np.sin(angles))
        self.max_len = max_len

    def forward(self, x, offset=0):
        T = x.shape[-2]
        cos, sin = self.cos[offset:offset + T], self.sin[offset:offset + T]
        return x * cos + rotate_half(x) * sin

    def backward(self, dy, offset=0):
        T = dy.shape[-2]
        cos, sin = self.cos[offset:offset + T], self.sin[offset:offset + T]
        return dy * cos - rotate_half(dy * sin)


class CausalSelfAttention(Module):
    """softmax( Q Kᵀ/√D + mask ) V, per head, then an output projection.

    The causal mask sets every score where key position > query position to -inf, so
    after softmax a token can only attend to itself and the past — it cannot peek at the
    token it is being trained to predict.
    """

    def __init__(self, dim, n_head, max_len, rng, bias=True, dropout=0.0, rope=False,
                 rope_base=10000.0, proj_std=0.02):
        assert dim % n_head == 0
        self.dim, self.n_head, self.head_dim = dim, n_head, dim // n_head
        self.qkv = Linear(dim, 3 * dim, rng, bias=bias)
        self.proj = Linear(dim, dim, rng, bias=bias, std=proj_std)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)
        self.rope = RoPE(self.head_dim, max_len, rope_base) if rope else None
        self.max_len = max_len
        self.cache = None    # KV cache for incremental generation (see forward_cached)

    # -- helpers -----------------------------------------------------------------------
    def _split_heads(self, t):          # (B,T,C) -> (B,H,T,D)
        Bsz, T, _ = t.shape
        return t.reshape(Bsz, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, t):          # (B,H,T,D) -> (B,T,C)
        Bsz, _, T, _ = t.shape
        return t.transpose(0, 2, 1, 3).reshape(Bsz, T, self.dim)

    # -- training path -----------------------------------------------------------------
    def forward(self, x):
        xp = B.xp
        T = x.shape[1]
        q, k, v = xp.split(self.qkv(x), 3, axis=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        if self.rope is not None:
            q, k = self.rope.forward(q), self.rope.forward(k)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale                  # (B,H,T,T)
        causal = xp.tril(xp.ones((T, T), dtype=bool))
        scores = xp.where(causal, scores, -xp.inf)
        probs = softmax(scores)
        probs_d = self.attn_drop(probs)
        out = self._merge_heads(probs_d @ v)
        self.q, self.k, self.v, self.probs, self.probs_d, self.scale = q, k, v, probs, probs_d, scale
        return self.resid_drop(self.proj(out))

    def backward(self, dy):
        q, k, v, P, Pd, scale = self.q, self.k, self.v, self.probs, self.probs_d, self.scale
        dout = self._split_heads(self.proj.backward(self.resid_drop.backward(dy)))
        # out = Pd @ v
        dPd = dout @ v.transpose(0, 1, 3, 2)
        dv = Pd.transpose(0, 1, 3, 2) @ dout
        dP = self.attn_drop.backward(dPd)
        # softmax backward, row-wise: dS = P ⊙ (dP - Σ_j dP_j P_j). Masked entries have
        # P = 0, so they get exactly zero gradient — the mask needs no special handling.
        dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True)) * scale
        dq = dS @ k
        dk = dS.transpose(0, 1, 3, 2) @ q
        if self.rope is not None:
            dq, dk = self.rope.backward(dq), self.rope.backward(dk)
        dqkv = B.xp.concatenate(
            [self._merge_heads(dq), self._merge_heads(dk), self._merge_heads(dv)], axis=-1)
        return self.qkv.backward(dqkv)

    # -- inference path: KV cache ------------------------------------------------------
    def reset_cache(self, batch):
        shape = (batch, self.n_head, self.max_len, self.head_dim)
        self.cache = {"k": B.xp.zeros(shape, dtype=B.dtype),
                      "v": B.xp.zeros(shape, dtype=B.dtype), "len": 0}

    def forward_cached(self, x):
        """Process only the *new* tokens x (B,t,C), reusing keys/values of earlier tokens.

        Without a cache, generating token n re-runs attention over all n tokens: O(n²)
        work per token. With it, each step computes q/k/v for one token and attends over the
        stored keys — the standard trick that makes LLM inference affordable.
        """
        xp = B.xp
        c = self.cache
        start, t = c["len"], x.shape[1]
        assert start + t <= self.max_len, "KV cache full — caller must re-prefill"
        q, k, v = xp.split(self.qkv(x), 3, axis=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        if self.rope is not None:
            q, k = self.rope.forward(q, offset=start), self.rope.forward(k, offset=start)
        c["k"][:, :, start:start + t] = k
        c["v"][:, :, start:start + t] = v
        c["len"] = end = start + t
        keys, vals = c["k"][:, :, :end], c["v"][:, :, :end]
        scores = (q @ keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)  # (B,H,t,end)
        # new token i sits at absolute position start+i and may see keys 0..start+i
        allowed = xp.arange(end)[None, :] <= (start + xp.arange(t))[:, None]
        probs = softmax(xp.where(allowed, scores, -xp.inf))
        return self.proj(self._merge_heads(probs @ vals))
