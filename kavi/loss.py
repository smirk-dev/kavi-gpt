"""Softmax + cross-entropy, fused.                       docs/07-loss.md

loss = -(1/N) Σ_n log softmax(z_n)[y_n]

Fusing the two gives a famously clean gradient:
    dL/dz_n = (softmax(z_n) - onehot(y_n)) / N
"predicted probabilities minus the truth". Computing softmax and log separately would be
both slower and numerically worse (log of a probability that underflowed to 0 = -inf).
"""
from . import backend as B


class CrossEntropy:
    def forward(self, logits, targets):
        xp = B.xp
        V = logits.shape[-1]
        z = logits.reshape(-1, V)
        y = targets.reshape(-1)
        z = z - z.max(axis=1, keepdims=True)             # stability; cancels out exactly
        lse = xp.log(xp.exp(z).sum(axis=1, keepdims=True))
        logp = z - lse                                    # log-softmax
        self.probs = xp.exp(logp)
        self.y, self.shape = y, logits.shape
        rows = xp.arange(y.shape[0])
        return -logp[rows, y].mean()

    def backward(self):
        xp = B.xp
        n = self.y.shape[0]
        d = self.probs.copy()
        d[xp.arange(n), self.y] -= 1.0
        d /= n
        self.probs = None                                 # free the (N,V) buffer early
        return d.reshape(self.shape)
