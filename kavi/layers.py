"""The basic Lego bricks. Every backward pass here is derived in docs/ (chapter per brick).

Shape convention: activations are (..., C) — usually (B, T, C) — and every brick works on
the last axis, so it doesn't care how many leading batch axes there are.
"""
import math

from . import backend as B
from .module import Module, Param


def _sum_to_last(a):
    """Sum over every axis except the last: (..., C) -> (C,). Used for bias/gain grads."""
    return a.reshape(-1, a.shape[-1]).sum(axis=0)


# --------------------------------------------------------------------------------------
class Linear(Module):
    """y = x W + b.            docs/03-linear-and-backprop.md

    backward:  dW = xᵀ dy  (summed over every batch position)
               db = Σ dy
               dx = dy Wᵀ
    """

    def __init__(self, n_in, n_out, rng, bias=True, std=0.02):
        self.n_in, self.n_out = n_in, n_out
        self.weight = Param(B.normal(rng, (n_in, n_out), std))
        self.bias = Param(B.xp.zeros(n_out, dtype=B.dtype), decay=False) if bias else None

    def forward(self, x):
        self.x = x
        y = x @ self.weight.data
        if self.bias is not None:
            y = y + self.bias.data
        return y

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.n_in)
        dy2 = dy.reshape(-1, self.n_out)
        self.weight.grad += x2.T @ dy2
        if self.bias is not None:
            self.bias.grad += dy2.sum(axis=0)
        return dy @ self.weight.data.T


# --------------------------------------------------------------------------------------
class Embedding(Module):
    """out = W[ids] — a lookup table, i.e. a Linear layer applied to one-hot vectors.

    backward: row W[i] receives the sum of the output gradients at every position where
    token i appeared. Duplicates must accumulate -> scatter_add, not fancy assignment.
    There is no gradient w.r.t. the integer ids, so backward returns None.
    """

    def __init__(self, n_rows, dim, rng, std=0.02):
        self.dim = dim
        self.weight = Param(B.normal(rng, (n_rows, dim), std))

    def forward(self, ids):
        self.ids = ids
        return self.weight.data[ids]

    def backward(self, dout):
        B.scatter_add(self.weight.grad, self.ids.reshape(-1), dout.reshape(-1, self.dim))
        return None


# --------------------------------------------------------------------------------------
class LayerNorm(Module):
    """y = g ⊙ (x - μ)/σ + b over the last axis.      docs/04-normalization.md

    With x̂ = (x-μ)·r, r = 1/sqrt(var+eps), and dx̂ = dy ⊙ g:
        dx = r · ( dx̂ - mean(dx̂) - x̂ · mean(dx̂ ⊙ x̂) )
    The two subtracted terms are the gradient flowing back through μ and through σ.
    """

    def __init__(self, dim, eps=1e-5, bias=True):
        self.eps = eps
        self.weight = Param(B.xp.ones(dim, dtype=B.dtype), decay=False)
        self.bias = Param(B.xp.zeros(dim, dtype=B.dtype), decay=False) if bias else None

    def forward(self, x):
        xp = B.xp
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        self.r = 1.0 / xp.sqrt(var + self.eps)
        self.xhat = xc * self.r
        y = self.xhat * self.weight.data
        if self.bias is not None:
            y = y + self.bias.data
        return y

    def backward(self, dy):
        xhat, r = self.xhat, self.r
        self.weight.grad += _sum_to_last(dy * xhat)
        if self.bias is not None:
            self.bias.grad += _sum_to_last(dy)
        dxhat = dy * self.weight.data
        return r * (dxhat
                    - dxhat.mean(axis=-1, keepdims=True)
                    - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))


class RMSNorm(Module):
    """y = g ⊙ x / rms(x),  rms(x) = sqrt(mean(x²) + eps).      docs/04-normalization.md

    LayerNorm without the mean-centering (and without a bias): cheaper, and in practice just
    as good — it's what Llama uses. With x̂ = x·r, r = 1/rms(x), dx̂ = dy ⊙ g:
        dx = r · ( dx̂ - x̂ · mean(dx̂ ⊙ x̂) )
    (LayerNorm's formula minus the mean term, because there is no μ to backprop through.)
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Param(B.xp.ones(dim, dtype=B.dtype), decay=False)

    def forward(self, x):
        self.r = 1.0 / B.xp.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        self.xhat = x * self.r
        return self.xhat * self.weight.data

    def backward(self, dy):
        xhat, r = self.xhat, self.r
        self.weight.grad += _sum_to_last(dy * xhat)
        dxhat = dy * self.weight.data
        return r * (dxhat - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))


def make_norm(kind, dim, bias=True):
    if kind == "layernorm":
        return LayerNorm(dim, bias=bias)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"unknown norm {kind!r}")


# --------------------------------------------------------------------------------------
class ReLU(Module):
    """max(0, x). Anton's choice; kept for comparison."""

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask


_GELU_C = math.sqrt(2.0 / math.pi)


class GELU(Module):
    """GPT-2's tanh-approximated GELU:  0.5·x·(1 + tanh(c·(x + 0.044715·x³))).

    d/dx = 0.5·(1+t) + 0.5·x·(1-t²)·c·(1 + 3·0.044715·x²),   t = tanh(...)
    Unlike ReLU it is smooth and lets a little negative signal through.
    """

    def forward(self, x):
        self.x = x
        self.t = B.xp.tanh(_GELU_C * (x + 0.044715 * x ** 3))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy):
        x, t = self.x, self.t
        dt = _GELU_C * (1.0 + 3 * 0.044715 * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dt)


def sigmoid(a):
    # 0.5·(1+tanh(a/2)) equals 1/(1+e^-a) but never overflows for large |a|.
    return 0.5 * (1.0 + B.xp.tanh(0.5 * a))


# --------------------------------------------------------------------------------------
class Dropout(Module):
    """Training: zero each element with prob p and scale survivors by 1/(1-p) ("inverted"
    dropout), so the expected activation is unchanged and eval needs no rescaling.
    backward multiplies by the same mask. Eval / p=0: identity."""

    def __init__(self, p=0.0):
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (B.rand(x.shape) >= self.p).astype(B.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask


# --------------------------------------------------------------------------------------
class MLP(Module):
    """GPT-2 feed-forward: Linear(C→4C) → GELU (or ReLU) → Linear(4C→C).
    The "computation" half of a block (attention is the "communication" half)."""

    def __init__(self, dim, rng, bias=True, dropout=0.0, act="gelu", proj_std=0.02):
        hidden = 4 * dim
        self.fc = Linear(dim, hidden, rng, bias=bias)
        self.act = GELU() if act == "gelu" else ReLU()
        self.proj = Linear(hidden, dim, rng, bias=bias, std=proj_std)
        self.drop = Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))

    def backward(self, dy):
        return self.fc.backward(self.act.backward(self.proj.backward(self.drop.backward(dy))))


def swiglu_hidden(dim):
    """Hidden width giving SwiGLU the same parameter count as a 4C GELU MLP.

    GELU MLP: 2 matrices of C×4C = 8C².  SwiGLU: 3 matrices of C×H = 3CH.  3CH = 8C²
    ⇒ H = 8C/3, rounded to a multiple of 8 so the matmuls stay tidy.
    """
    return max(8, int(round(8 * dim / 3 / 8)) * 8)


class SwiGLU(Module):
    """Llama's gated feed-forward:  out = W2( SiLU(x W1) ⊙ (x W3) ).    docs/06-mlp.md

    One branch (W3) carries content; the other (W1, through SiLU) decides how much of each
    hidden feature to let through — a learned, input-dependent gate.

    With a = xW1, b = xW3, s = σ(a), h = a·s·b:
        db = dh ⊙ a·s
        da = dh ⊙ b ⊙ s·(1 + a·(1-s))           (derivative of SiLU(a) = a·σ(a))
        dx = da W1ᵀ + db W3ᵀ                     (x feeds both branches → grads add)
    """

    def __init__(self, dim, rng, bias=False, dropout=0.0, proj_std=0.02):
        hidden = swiglu_hidden(dim)
        self.w1 = Linear(dim, hidden, rng, bias=bias)
        self.w3 = Linear(dim, hidden, rng, bias=bias)
        self.w2 = Linear(hidden, dim, rng, bias=bias, std=proj_std)
        self.drop = Dropout(dropout)

    def forward(self, x):
        a = self.w1(x)
        b = self.w3(x)
        s = sigmoid(a)
        self.a, self.b, self.s = a, b, s
        return self.drop(self.w2(a * s * b))

    def backward(self, dy):
        a, b, s = self.a, self.b, self.s
        dh = self.w2.backward(self.drop.backward(dy))
        db = dh * (a * s)
        da = dh * b * s * (1.0 + a * (1.0 - s))
        return self.w1.backward(da) + self.w3.backward(db)
