"""Finite-difference gradient checking — how we *prove* every backward pass.

For a scalar objective L(θ), the centred difference
    (L(θ + h·e_i) - L(θ - h·e_i)) / 2h
approximates ∂L/∂θ_i with error O(h²). We compare it to the analytic gradient our
backward() produced, using the relative error
    |a - n| / (|a| + |n|)          (in [0, 1]; 1 means opposite signs or one side zero)
except when both are tiny (|a| + |n| < 1e-7): then the gradient is zero up to rounding
(e.g. by symmetry) and we compare absolute error instead.
In float64 with h ≈ 1e-4 a correct gradient typically agrees to ~1e-7..1e-10; a bug shows
up as errors of 1e-2 or worse. Always run in float64 (``B.set_precision("float64")``):
float32 has only ~7 significant digits, so the difference quotient drowns in rounding.
"""
import numpy as np

from . import backend as B


def _numeric(f, arr, idx, h):
    old = float(arr[idx])
    arr[idx] = old + h
    plus = f()
    arr[idx] = old - h
    minus = f()
    arr[idx] = old
    return (plus - minus) / (2 * h)


def _rel_err(a, n, zero=1e-7, atol=1e-8):
    """Relative error, except when the true gradient is (near) zero.

    Some gradients are *exactly* zero by symmetry — e.g. the key bias in attention without
    RoPE: adding b to every key adds the constant q·b to a query's scores, softmax ignores
    constants, so ∂L/∂b_k ≡ 0 (docs/05-attention.md). The numerical estimate is then pure
    rounding noise (~1e-10) and the relative error a meaningless ~1. For such elements we
    check the absolute error instead and report 0 if it is within atol.
    """
    if abs(a) + abs(n) < zero:
        return 0.0 if abs(a - n) < atol else 1.0
    return abs(a - n) / (abs(a) + abs(n))


def _sample_indices(shape, n, rng):
    size = int(np.prod(shape))
    flat = rng.choice(size, size=min(n, size), replace=False)
    return [np.unravel_index(i, shape) for i in flat]


def check(objective, analytic, tensors, n_samples=40, h=1e-4, seed=0):
    """Generic checker.

    objective()        -> float  (recomputes the scalar loss from scratch)
    analytic           dict name -> analytic gradient array (already computed)
    tensors            dict name -> the array that is perturbed in place
    Returns dict name -> max relative error over the sampled elements.
    """
    rng = np.random.default_rng(seed)
    worst = {}
    for name, arr in tensors.items():
        grad = B.to_numpy(analytic[name])
        errs = [_rel_err(float(grad[i]), _numeric(objective, arr, i, h))
                for i in _sample_indices(arr.shape, n_samples, rng)]
        worst[name] = max(errs)
    return worst


def check_module(module, x, n_samples=40, h=1e-4, seed=0, check_input=True):
    """Gradcheck a Module on input x with objective L = Σ module(x) ⊙ G, G random.

    Using a random G (instead of e.g. L = Σ out) exercises every output element with a
    different weight, so a bug that mixes up output positions can't cancel out.
    """
    rng = np.random.default_rng(seed + 1)
    out = module.forward(x)
    G = B.asarray(rng.normal(size=out.shape))
    module.zero_grad()
    module.forward(x)
    dx = module.backward(G)

    def objective():
        return float((module.forward(x) * G).sum())

    tensors, analytic = {}, {}
    for name, p in module.named_params():
        tensors[name], analytic[name] = p.data, p.grad.copy()
    if check_input:
        tensors["<input>"], analytic["<input>"] = x, dx
    return check(objective, analytic, tensors, n_samples, h, seed)
