"""FlashAttention (Dao et al., 2022), the algorithm without the CUDA.    docs/12-flash-attention.md

Naive attention materialises the full (T, T) score matrix per head: O(T²) memory, and
on a GPU most of the time goes to moving that matrix to and from slow memory. FlashAttention
computes *exactly the same output* tile by tile and never stores the matrix:

Forward — for each block of queries, stream over blocks of keys keeping, per query row,
  m  the running max of the scores seen so far
  l  the running softmax denominator  Σ exp(s - m)
  acc the running numerator           Σ exp(s - m) · v
When a new tile raises the max from m to m', everything accumulated so far is rescaled by
exp(m - m') ("online softmax"). At the end, out = acc / l, and we keep only the per-row
log-sum-exp  L = m + log l  (O(T) memory) for the backward pass.

Backward — recompute each tile's probabilities from L instead of storing them:
  P = exp(S - L),  dV += Pᵀ dO,  dP = dO Vᵀ,  dS = P ⊙ (dP - Δ),  Δ = rowsum(dO ⊙ O)
  dQ += dS K · scale,  dK += dSᵀ Q · scale
(Δ = rowsum(dO ⊙ O) equals Σ_j dP_ij P_ij — the softmax-backward term — but costs O(T·D)
to compute instead of needing the full P.)

Recomputing is cheaper than remembering: that trade is the whole idea. In NumPy it is
slower than the naive version (Python loops over tiles); the point here is to understand
and verify the algorithm. Tests prove forward and backward match naive attention.
"""
import math

from . import backend as B


def _causal_mask(qs, qe, ks, ke):
    xp = B.xp
    return xp.arange(ks, ke)[None, :] <= xp.arange(qs, qe)[:, None]


def flash_forward(q, k, v, causal=True, block=32):
    """q, k, v: (B, H, T, D) -> (out (B,H,T,D), lse (B,H,T,1))."""
    xp = B.xp
    T, D = q.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    out = xp.empty_like(q)
    lse = xp.empty(q.shape[:-1] + (1,), dtype=q.dtype)
    for qs in range(0, T, block):
        qe = min(qs + block, T)
        qi = q[..., qs:qe, :]
        m = xp.full(qi.shape[:-1] + (1,), -xp.inf, dtype=q.dtype)
        l = xp.zeros_like(m)
        acc = xp.zeros_like(qi)
        for ks in range(0, qe if causal else T, block):   # causal: later keys are all masked
            ke = min(ks + block, T)
            s = (qi @ k[..., ks:ke, :].transpose(0, 1, 3, 2)) * scale
            if causal:
                s = xp.where(_causal_mask(qs, qe, ks, ke), s, -xp.inf)
            m_new = xp.maximum(m, s.max(axis=-1, keepdims=True))
            p = xp.exp(s - m_new)
            alpha = xp.exp(m - m_new)          # rescales what was accumulated so far
            l = alpha * l + p.sum(axis=-1, keepdims=True)
            acc = alpha * acc + p @ v[..., ks:ke, :]
            m = m_new
        out[..., qs:qe, :] = acc / l
        lse[..., qs:qe, :] = m + xp.log(l)
    return out, lse


def flash_backward(q, k, v, out, lse, dout, causal=True, block=32):
    """Gradients of flash_forward w.r.t. q, k, v, recomputing every probability tile."""
    xp = B.xp
    T, D = q.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    dq, dk, dv = xp.zeros_like(q), xp.zeros_like(k), xp.zeros_like(v)
    delta = (dout * out).sum(axis=-1, keepdims=True)        # (B,H,T,1)
    for qs in range(0, T, block):
        qe = min(qs + block, T)
        qi, doi = q[..., qs:qe, :], dout[..., qs:qe, :]
        for ks in range(0, qe if causal else T, block):
            ke = min(ks + block, T)
            kj, vj = k[..., ks:ke, :], v[..., ks:ke, :]
            s = (qi @ kj.transpose(0, 1, 3, 2)) * scale
            if causal:
                s = xp.where(_causal_mask(qs, qe, ks, ke), s, -xp.inf)
            p = xp.exp(s - lse[..., qs:qe, :])                # exact softmax probs, recomputed
            dv[..., ks:ke, :] += p.transpose(0, 1, 3, 2) @ doi
            dp = doi @ vj.transpose(0, 1, 3, 2)
            ds = p * (dp - delta[..., qs:qe, :]) * scale
            dq[..., qs:qe, :] += ds @ kj
            dk[..., ks:ke, :] += ds.transpose(0, 1, 3, 2) @ qi
    return dq, dk, dv
