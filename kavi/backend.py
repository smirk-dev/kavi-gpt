"""Array backend switch: the whole model is written against ``B.xp``.

``xp`` is NumPy by default. ``set_backend("cupy")`` swaps in CuPy, whose API mirrors NumPy,
so the exact same hand-written forward/backward code runs on a GPU (docs/11-gpu.md).

Modules must always look the backend up at call time (``B.xp.exp(...)``), never bind
``xp`` at import time, or a later ``set_backend`` would not reach them.
"""
import numpy as np

xp = np                 # active array module (numpy or cupy)
dtype = np.float32      # dtype for parameters and activations
name = "numpy"


def set_backend(backend: str = "numpy", precision: str | None = None) -> None:
    """Select the array library ("numpy" | "cupy") and optionally the float precision."""
    global xp, name
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        import cupy  # noqa: F401  (imported lazily: CPU machines need not have it)
        xp = cupy
    else:
        raise ValueError(f"unknown backend {backend!r}")
    name = backend
    if precision is not None:
        set_precision(precision)


def set_precision(precision: str) -> None:
    """"float32" for training, "float64" for gradient checking."""
    global dtype
    dtype = {"float32": np.float32, "float64": np.float64}[precision]


def asarray(a, dt=None):
    """Move/convert ``a`` onto the active backend (defaults to the float dtype)."""
    return xp.asarray(a, dtype=dtype if dt is None else dt)


def to_numpy(a):
    """Bring an array back to host NumPy (no-op for NumPy arrays)."""
    if name == "cupy":
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    return np.asarray(a)


def normal(rng: np.random.Generator, shape, std: float):
    """Initialise weights on the host with a seeded NumPy RNG, then move them.

    Doing init on the host makes a seed give bit-identical weights on CPU and GPU.
    """
    return asarray(rng.normal(0.0, std, size=shape))


def scatter_add(target, idx, values) -> None:
    """In-place ``target[idx] += values`` that *accumulates* repeated indices.

    ``target[idx] += values`` silently drops repeats (last write wins), which would lose
    gradient for every token that appears twice in a batch. ``np.add.at`` does it right.
    """
    if name == "cupy":
        if hasattr(xp.add, "at"):          # CuPy >= 13
            xp.add.at(target, idx, values)
        else:
            import cupyx
            cupyx.scatter_add(target, idx, values)
    else:
        np.add.at(target, idx, values)


_rng = None


def seed(s: int) -> None:
    """Seed the backend RNG used for dropout masks and data sampling."""
    global _rng
    _rng = xp.random.RandomState(s) if name == "cupy" else np.random.default_rng(s)


def rand(shape):
    """Uniform [0, 1) on the active backend (dropout masks)."""
    global _rng
    if _rng is None:
        seed(0)
    if name == "cupy":
        return _rng.random_sample(shape, dtype=dtype)
    return _rng.random(shape, dtype=np.float32).astype(dtype, copy=False)
