"""Second, independent oracle: re-implement the forward pass with PyTorch ops, let torch's
autograd produce the gradients, and demand they match our hand-written backward passes.

Finite differences prove our gradients match *our* forward pass; this additionally proves
our forward pass matches the standard definitions (torch's layer_norm, gelu, silu, ...).
Skipped when torch is not installed — it is a test-only dependency.
"""
import math

import numpy as np
import pytest

from kavi import backend as B
from kavi.model import GPT, GPTConfig
from kavi.optim import AdamW

torch = pytest.importorskip("torch")
F = torch.nn.functional


def torch_loss(state, cfg, ids, tgt):
    P = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in state.items()}
    Bsz, T = ids.shape
    C, H = cfg.n_embd, cfg.n_head
    D = C // H
    ids_t, tgt_t = torch.tensor(ids), torch.tensor(tgt)

    def lin(x, name):
        y = x @ P[name + ".weight"]
        return y + P[name + ".bias"] if name + ".bias" in P else y

    def norm(x, name):
        if cfg.norm == "layernorm":
            return F.layer_norm(x, (C,), P[name + ".weight"], P.get(name + ".bias"), eps=1e-5)
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * P[name + ".weight"]

    def rope(t):
        inv = cfg.rope_base ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
        ang = torch.outer(torch.arange(T, dtype=torch.float64), inv)
        ang = torch.cat([ang, ang], -1)
        t1, t2 = t[..., :D // 2], t[..., D // 2:]
        return t * torch.cos(ang) + torch.cat([-t2, t1], -1) * torch.sin(ang)

    x = P["wte.weight"][ids_t]
    if cfg.pos == "learned":
        x = x + P["wpe.weight"][:T]
    for i in range(cfg.n_layer):
        pre = f"blocks.{i}."
        q, k, v = lin(norm(x, pre + "norm1"), pre + "attn.qkv").split(C, dim=-1)
        q, k, v = (t.view(Bsz, T, H, D).transpose(1, 2) for t in (q, k, v))
        if cfg.pos == "rope":
            q, k = rope(q), rope(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(Bsz, T, C)
        x = x + lin(y, pre + "attn.proj")
        h = norm(x, pre + "norm2")
        if cfg.mlp == "swiglu":
            m = lin(F.silu(lin(h, pre + "mlp.w1")) * lin(h, pre + "mlp.w3"), pre + "mlp.w2")
        else:
            act = F.gelu(lin(h, pre + "mlp.fc"), approximate="tanh") if cfg.mlp == "gelu" \
                else F.relu(lin(h, pre + "mlp.fc"))
            m = lin(act, pre + "mlp.proj")
        x = x + m
    logits = norm(x, "norm_f") @ P["wte.weight"].T
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt_t.reshape(-1))
    loss.backward()
    return float(loss.detach()), {k: p.grad.numpy() for k, p in P.items()}


@pytest.mark.parametrize("preset,bias", [("gpt2", True), ("gpt2", False),
                                         ("kavi", False), ("kavi", True)])
def test_model_matches_torch(preset, bias):
    cfg = GPTConfig.preset(preset, vocab_size=13, n_embd=16, n_head=4, block_size=10,
                           n_layer=2, bias=bias)
    model = GPT(cfg, seed=3)
    rng = np.random.default_rng(0)
    for _, p in model.named_params():              # non-trivial gains/biases
        p.data += B.asarray(rng.normal(scale=0.1, size=p.data.shape))
    ids = rng.integers(0, 13, size=(3, 9))
    tgt = rng.integers(0, 13, size=(3, 9))

    model.zero_grad()
    _, loss = model.forward(ids, tgt)
    model.backward()
    t_loss, t_grads = torch_loss(model.state_dict(), cfg, ids, tgt)

    assert abs(float(loss) - t_loss) < 1e-10
    for name, p in model.named_params():
        np.testing.assert_allclose(p.grad, t_grads[name], rtol=1e-7, atol=1e-11, err_msg=name)


def test_adamw_matches_torch():
    rng = np.random.default_rng(0)
    w0, b0 = rng.normal(size=(4, 3)), rng.normal(size=3)
    from kavi.module import Param
    pw, pb = Param(B.asarray(w0.copy())), Param(B.asarray(b0.copy()), decay=False)
    opt = AdamW([pw, pb], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    tw = torch.tensor(w0.copy(), requires_grad=True)
    tb = torch.tensor(b0.copy(), requires_grad=True)
    topt = torch.optim.AdamW([{"params": [tw], "weight_decay": 0.1},
                              {"params": [tb], "weight_decay": 0.0}],
                             lr=0.01, betas=(0.9, 0.95), eps=1e-8)
    for step in range(5):
        gw, gb = rng.normal(size=(4, 3)), rng.normal(size=3)
        pw.grad[...], pb.grad[...] = gw, gb
        opt.step()
        tw.grad, tb.grad = torch.tensor(gw), torch.tensor(gb)
        topt.step()
    np.testing.assert_allclose(pw.data, tw.detach().numpy(), rtol=1e-10)
    np.testing.assert_allclose(pb.data, tb.detach().numpy(), rtol=1e-10)
