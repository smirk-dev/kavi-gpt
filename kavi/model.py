"""The GPT: stack the bricks into a castle.                docs/00-overview.md

Two presets share this one class — the difference is three config knobs:

    gpt2 : pos="learned", norm="layernorm", mlp="gelu"     (GPT-2 / Anton-style baseline)
    kavi : pos="rope",    norm="rmsnorm",   mlp="swiglu"   (Llama-style modern block)
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

from . import backend as B
from .attention import CausalSelfAttention, softmax
from .layers import MLP, Dropout, Embedding, SwiGLU, make_norm
from .loss import CrossEntropy
from .module import Module


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128       # max context length T
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128           # model width C
    dropout: float = 0.0
    bias: bool = False          # biases in Linear/LayerNorm (GPT-2 had them; modern models drop them)
    pos: str = "rope"           # "learned" | "rope" | "none" (no position info: an ablation)
    norm: str = "rmsnorm"       # "layernorm" | "rmsnorm"
    mlp: str = "swiglu"         # "gelu" | "relu" | "swiglu"
    rope_base: float = 10000.0
    attn: str = "naive"         # "naive" | "flash" (tiled online-softmax, docs/12)

    @classmethod
    def preset(cls, name, **overrides):
        presets = {
            "gpt2": dict(pos="learned", norm="layernorm", mlp="gelu"),
            "kavi": dict(pos="rope", norm="rmsnorm", mlp="swiglu"),
        }
        return cls(**{**presets[name], **overrides})

    def to_dict(self):
        return asdict(self)


class Block(Module):
    """Pre-norm transformer block:
        x = x + Attn(Norm(x))      communication: tokens exchange information
        x = x + MLP(Norm(x))       computation:   each token thinks on its own

    The residual "x +" gives gradients a straight highway from the loss to every layer
    (d(x + f(x))/dx = I + f'(x)), which is what lets many blocks stack without the signal
    dying. Pre-norm (norm *inside* the branch) keeps that highway clean.
    """

    def __init__(self, cfg: GPTConfig, rng):
        C = cfg.n_embd
        # GPT-2 trick: shrink the init of the layers that write *into* the residual stream
        # by 1/sqrt(2·n_layer), so the stream's variance doesn't grow with depth.
        proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.norm1 = make_norm(cfg.norm, C, bias=cfg.bias)
        self.attn = CausalSelfAttention(C, cfg.n_head, cfg.block_size, rng, bias=cfg.bias,
                                        dropout=cfg.dropout, rope=(cfg.pos == "rope"),
                                        rope_base=cfg.rope_base, proj_std=proj_std,
                                        flash=(cfg.attn == "flash"))
        self.norm2 = make_norm(cfg.norm, C, bias=cfg.bias)
        if cfg.mlp == "swiglu":
            self.mlp = SwiGLU(C, rng, bias=cfg.bias, dropout=cfg.dropout, proj_std=proj_std)
        else:
            self.mlp = MLP(C, rng, bias=cfg.bias, dropout=cfg.dropout, act=cfg.mlp,
                           proj_std=proj_std)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def backward(self, dy):
        # the residual copies the incoming gradient to both the skip path and the branch
        dx = dy + self.norm2.backward(self.mlp.backward(dy))
        return dx + self.norm1.backward(self.attn.backward(dx))

    def forward_cached(self, x):
        x = x + self.attn.forward_cached(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(Module):
    def __init__(self, cfg: GPTConfig, seed: int = 0):
        self.cfg = cfg
        for knob, ok in (("pos", ("learned", "rope", "none")), ("norm", ("layernorm", "rmsnorm")),
                         ("mlp", ("gelu", "relu", "swiglu")), ("attn", ("naive", "flash"))):
            if getattr(cfg, knob) not in ok:   # a typo like pos="rotary" must not silently
                raise ValueError(f"{knob}={getattr(cfg, knob)!r}; expected one of {ok}")  # drop RoPE
        rng = np.random.default_rng(seed)
        self.wte = Embedding(cfg.vocab_size, cfg.n_embd, rng)
        self.wpe = Embedding(cfg.block_size, cfg.n_embd, rng) if cfg.pos == "learned" else None
        self.drop = Dropout(cfg.dropout)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layer)]
        self.norm_f = make_norm(cfg.norm, cfg.n_embd, bias=cfg.bias)
        self.loss_fn = CrossEntropy()

    # -- training path -----------------------------------------------------------------
    def forward(self, ids, targets=None):
        """ids (B,T) int -> logits (B,T,V); with targets also returns the mean loss."""
        T = ids.shape[1]
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        x = self.wte(ids)
        if self.wpe is not None:
            x = x + self.wpe(B.xp.arange(T)[None, :])          # (1,T,C) broadcasts over B
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        h = self.norm_f(x)
        self.h = h
        # weight tying: the output projection *is* the embedding table, transposed.
        # Same idea in both directions (token <-> vector), half the parameters, better
        # results on small data.
        logits = h @ self.wte.weight.data.T
        if targets is None:
            return logits, None
        return logits, self.loss_fn.forward(logits, targets)

    def backward(self):
        """Backprop from the loss computed by the last forward() into every Param.grad."""
        C, V = self.cfg.n_embd, self.cfg.vocab_size
        dlogits = self.loss_fn.backward()
        W = self.wte.weight
        W.grad += dlogits.reshape(-1, V).T @ self.h.reshape(-1, C)   # head's use of wte
        dx = self.norm_f.backward(dlogits @ W.data)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        dx = self.drop.backward(dx)
        if self.wpe is not None:
            self.wpe.backward(dx.sum(axis=0, keepdims=True))         # undo the broadcast
        self.wte.backward(dx)                                        # input's use of wte

    # -- inference ---------------------------------------------------------------------
    def _forward_cached(self, ids, start):
        x = self.wte.weight.data[ids]
        if self.wpe is not None:
            x = x + self.wpe.weight.data[start:start + ids.shape[1]][None]
        for block in self.blocks:
            x = block.forward_cached(x)
        return self.norm_f(x)[:, -1] @ self.wte.weight.data.T      # logits of last position

    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 use_cache=True, rng=None):
        """Autoregressive sampling: predict, append, repeat. ids is a (B,T0) int array."""
        rng = rng or np.random.default_rng()
        was_training = self.training
        self.eval()
        xp, T_max = B.xp, self.cfg.block_size
        ids = xp.asarray(ids)
        pos = 0
        for i in range(max_new_tokens):
            if use_cache:
                if i == 0 or pos >= T_max:
                    # (re)prefill: keep the last half-context so we don't refill every step
                    ctx = ids[:, -(T_max // 2 if i else T_max):]
                    for blk in self.blocks:
                        blk.attn.reset_cache(ids.shape[0])
                    logits = self._forward_cached(ctx, 0)
                    pos = ctx.shape[1]
                else:
                    logits = self._forward_cached(ids[:, -1:], pos)
                    pos += 1
            else:
                logits, _ = self.forward(ids[:, -T_max:])
                logits = logits[:, -1]
            nxt = sample(B.to_numpy(logits).astype(np.float64), temperature, top_k, top_p, rng)
            ids = xp.concatenate([ids, xp.asarray(nxt)[:, None]], axis=1)
        self.train(was_training)
        return ids


def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Pick the next token per row from (B,V) logits.   docs/10-sampling.md

    temperature  <1 sharpens the distribution (safer, more repetitive), >1 flattens it.
    top_k        keep only the k most likely tokens.
    top_p        "nucleus": keep the smallest set whose probabilities sum to >= p.
    temperature == 0 means greedy argmax.
    """
    rng = rng or np.random.default_rng()
    if temperature == 0:
        return logits.argmax(axis=-1)
    z = logits / temperature
    if top_k is not None:
        kth = np.sort(z, axis=-1)[:, -min(top_k, z.shape[-1])][:, None]
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    if top_p is not None:
        order = np.argsort(-p, axis=-1)
        sorted_p = np.take_along_axis(p, order, axis=-1)
        # drop a token if the mass *before* it already reaches top_p (always keeps the top 1)
        drop = (np.cumsum(sorted_p, axis=-1) - sorted_p) >= top_p
        sorted_p[drop] = 0.0
        p = np.zeros_like(p)
        np.put_along_axis(p, order, sorted_p, axis=-1)
        p /= p.sum(axis=-1, keepdims=True)
    return np.array([rng.choice(p.shape[-1], p=row) for row in p])


__all__ = ["GPT", "GPTConfig", "Block", "sample", "softmax"]
