"""Phase 7 gate: tiled FlashAttention is exactly naive attention, forward and backward."""
import math

import numpy as np
import pytest

from kavi import backend as B
from kavi.attention import CausalSelfAttention, softmax
from kavi.flash import flash_backward, flash_forward
from kavi.gradcheck import check_module


def naive(q, k, v):
    T = q.shape[-2]
    s = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
    p = softmax(np.where(np.tril(np.ones((T, T), bool)), s, -np.inf))
    return p @ v, p


def naive_grads(q, k, v, dout):
    out, p = naive(q, k, v)
    dp = dout @ v.transpose(0, 1, 3, 2)
    dv = p.transpose(0, 1, 3, 2) @ dout
    ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(q.shape[-1])
    return ds @ k, ds.transpose(0, 1, 3, 2) @ q, dv


@pytest.mark.parametrize("T,block", [(1, 4), (7, 4), (16, 4), (33, 8), (20, 64)])
def test_flash_matches_naive(T, block):
    rng = np.random.default_rng(T)
    q, k, v, dout = (rng.normal(size=(2, 3, T, 8)) for _ in range(4))
    out, lse = flash_forward(q, k, v, block=block)
    np.testing.assert_allclose(out, naive(q, k, v)[0], rtol=1e-12, atol=1e-12)
    got = flash_backward(q, k, v, out, lse, dout, block=block)
    for g, n in zip(got, naive_grads(q, k, v, dout)):
        np.testing.assert_allclose(g, n, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rope", [False, True])
def test_flash_module_gradcheck(rope):
    m = CausalSelfAttention(8, 2, 40, np.random.default_rng(0), rope=rope, proj_std=0.3,
                            flash=True)
    x = B.asarray(np.random.default_rng(1).normal(size=(2, 37, 8)))   # 37: ragged last tile
    errs = check_module(m, x, n_samples=40)
    assert max(errs.values()) < 1e-6, errs
