"""Optimisers, learning-rate schedule and gradient clipping.     docs/08-optimizers.md

Anton's first training run failed with plain SGD and worked with AdamW. Both are here (plus
SGD with momentum, the halfway house) so the comparison can be re-run, not just believed.
"""
import math

from . import backend as B


class SGD:
    """θ ← θ - lr·g      (momentum > 0:  v ← μ·v + g;  θ ← θ - lr·v)"""

    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        self.params, self.lr, self.momentum, self.wd = list(params), lr, momentum, weight_decay
        self.v = [B.xp.zeros_like(p.data) for p in self.params] if momentum else None

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad
            if self.wd and p.decay:
                g = g + self.wd * p.data
            if self.v is not None:
                self.v[i] *= self.momentum
                self.v[i] += g
                g = self.v[i]
            p.data -= self.lr * g

    def state_dict(self):
        return {f"v.{i}": B.to_numpy(v) for i, v in enumerate(self.v or [])}

    def load_state_dict(self, state):
        if self.v is not None:
            for i in range(len(self.v)):
                self.v[i][...] = B.asarray(state[f"v.{i}"])


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

        m ← β1·m + (1-β1)·g            momentum: running mean of gradients  (direction)
        v ← β2·v + (1-β2)·g²           running mean of squared grads        (scale)
        m̂ = m/(1-β1ᵗ), v̂ = v/(1-β2ᵗ)   bias correction (m, v start at 0)
        θ ← θ - lr·( m̂/(√v̂ + ε) + λ·θ )

    Dividing by √v̂ gives every parameter its own step size: parameters with large, noisy
    gradients take small steps, rarely-updated ones (e.g. embeddings of rare tokens) take
    big ones. "Decoupled" means λ·θ is applied directly instead of being added to g — in
    plain Adam the decay would also get divided by √v̂ and stop acting like decay.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params, self.lr = list(params), lr
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.m = [B.xp.zeros_like(p.data) for p in self.params]
        self.v = [B.xp.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        xp = B.xp
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            if self.wd and p.decay:
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * (m / c1) / (xp.sqrt(v / c2) + self.eps)

    def state_dict(self):
        s = {"t": self.t}
        for i, (m, v) in enumerate(zip(self.m, self.v)):
            s[f"m.{i}"], s[f"v.{i}"] = B.to_numpy(m), B.to_numpy(v)
        return s

    def load_state_dict(self, state):
        self.t = int(state["t"])
        for i in range(len(self.params)):
            self.m[i][...] = B.asarray(state[f"m.{i}"])
            self.v[i][...] = B.asarray(state[f"v.{i}"])


def lr_at(step, max_lr, min_lr, warmup, total):
    """Linear warmup to max_lr, then cosine decay to min_lr.

    Warmup: at step 0 Adam's v̂ is estimated from almost nothing, so early updates are
    erratic; ramping lr up lets the statistics settle before taking big steps.
    Cosine: big steps while far from a minimum, small careful ones at the end.
    """
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm(params, max_norm):
    """Rescale all grads together so their global L2 norm is at most max_norm.

    One unlucky batch can produce a huge gradient that throws the weights far off; clipping
    caps the step length while keeping its direction. Returns the pre-clip norm (worth
    logging: spikes in it are the first sign of instability).
    """
    xp = B.xp
    total = float(xp.sqrt(sum(float((p.grad * p.grad).sum()) for p in params)))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad *= scale
    return total
