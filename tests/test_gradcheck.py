"""Phase 1/2 gate: every brick and the whole model pass finite-difference gradcheck."""
import numpy as np
import pytest

from kavi import backend as B
from kavi.attention import CausalSelfAttention
from kavi.gradcheck import check, check_module
from kavi.layers import GELU, MLP, Embedding, LayerNorm, Linear, ReLU, RMSNorm, SwiGLU
from kavi.model import GPT, Block, GPTConfig

TOL = 1e-6          # single bricks
TOL_DEEP = 1e-5     # composites: errors compound and the h trade-off (truncation O(h²)
                    # vs float rounding O(eps/h)) leaves ~1e-6 noise. A real bug is >1e-3.


def x_of(*shape, seed=0, scale=1.0):
    return B.asarray(np.random.default_rng(seed).normal(scale=scale, size=shape))


def assert_ok(errs, tol=TOL):
    bad = {k: v for k, v in errs.items() if v > tol}
    assert not bad, f"gradcheck failed: {bad}  (all: {errs})"


def randomise(module, seed=0, scale=0.3):
    """Fresh modules have gains=1 and biases=0, which can hide bugs (a missing ⊙g term is
    invisible when g≡1). Perturb every param so each term in the gradient matters."""
    rng = np.random.default_rng(seed)
    for _, p in module.named_params():
        p.data += B.asarray(rng.normal(scale=scale, size=p.data.shape))


@pytest.mark.parametrize("bias", [True, False])
def test_linear(bias):
    m = Linear(6, 5, np.random.default_rng(0), bias=bias, std=0.5)
    randomise(m)
    assert_ok(check_module(m, x_of(2, 3, 6)))


def test_embedding():
    m = Embedding(10, 4, np.random.default_rng(0))
    ids = np.array([[1, 3, 3, 7], [0, 3, 9, 1]])        # repeats on purpose (scatter-add)
    assert_ok(check_module(m, ids, check_input=False, n_samples=40))


def test_layernorm():
    m = LayerNorm(8)
    randomise(m)
    assert_ok(check_module(m, x_of(2, 3, 8)))


def test_rmsnorm():
    m = RMSNorm(8)
    randomise(m)
    assert_ok(check_module(m, x_of(2, 3, 8)))


@pytest.mark.parametrize("act", [GELU, ReLU])
def test_activations(act):
    x = x_of(3, 7)
    if act is ReLU:          # finite differences are undefined exactly at the kink
        x = B.xp.where(abs(x) < 1e-3, 0.5, x)
    assert_ok(check_module(act(), x))


@pytest.mark.parametrize("act", ["gelu", "relu"])
def test_mlp(act):
    m = MLP(6, np.random.default_rng(0), bias=True, act=act, proj_std=0.3)
    randomise(m, scale=0.2)
    assert_ok(check_module(m, x_of(2, 3, 6)), tol=1e-5 if act == "relu" else TOL)


def test_swiglu():
    m = SwiGLU(6, np.random.default_rng(0))
    randomise(m, scale=0.3)
    assert_ok(check_module(m, x_of(2, 3, 6)))


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_attention(rope, bias):
    m = CausalSelfAttention(8, 2, 16, np.random.default_rng(0), bias=bias, rope=rope,
                            proj_std=0.3)
    randomise(m, scale=0.3)
    assert_ok(check_module(m, x_of(2, 5, 8), n_samples=60))


@pytest.mark.parametrize("preset", ["gpt2", "kavi"])
def test_block(preset):
    cfg = GPTConfig.preset(preset, n_embd=8, n_head=2, block_size=16, n_layer=2, bias=True)
    m = Block(cfg, np.random.default_rng(0))
    randomise(m, scale=0.2)
    assert_ok(check_module(m, x_of(2, 5, 8), n_samples=60), tol=TOL_DEEP)


@pytest.mark.parametrize("preset", ["gpt2", "kavi"])
def test_whole_model(preset):
    """End to end: embedding → blocks → tied head → cross-entropy, against the true loss."""
    cfg = GPTConfig.preset(preset, vocab_size=11, n_embd=8, n_head=2, block_size=8,
                           n_layer=2, bias=True)
    model = GPT(cfg, seed=0)
    randomise(model, scale=0.1)
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 11, size=(2, 6))
    tgt = rng.integers(0, 11, size=(2, 6))

    model.zero_grad()
    model.forward(ids, tgt)
    model.backward()

    def objective():
        return float(model.forward(ids, tgt)[1])

    named = model.named_params()
    errs = check(objective, {n: p.grad.copy() for n, p in named},
                 {n: p.data for n, p in named}, n_samples=25)
    assert_ok(errs, tol=TOL_DEEP)
