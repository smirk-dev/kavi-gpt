"""Prove the CuPy (GPU) backend computes the same thing as the NumPy (CPU) backend.

    python scripts/backend_parity.py

Builds the same model (same seed -> same weights, because init happens on the host) on
both backends in float64, runs one forward/backward on the same batch, and compares the
loss and every gradient. Run first in every Kaggle kernel: if this fails, nothing that
follows can be trusted.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kavi import backend as B  # noqa: E402
from kavi.model import GPT, GPTConfig  # noqa: E402


def run(backend, preset, ids, tgt):
    B.set_backend(backend, precision="float64")
    cfg = GPTConfig.preset(preset, vocab_size=50, n_embd=32, n_head=4, block_size=16, n_layer=2)
    model = GPT(cfg, seed=7)
    x, y = (B.xp.asarray(ids), B.xp.asarray(tgt))
    model.zero_grad()
    _, loss = model.forward(x, y)
    model.backward()
    return float(loss), {n: B.to_numpy(p.grad) for n, p in model.named_params()}


def main():
    rng = np.random.default_rng(0)
    ids, tgt = rng.integers(0, 50, (4, 16)), rng.integers(0, 50, (4, 16))
    worst_all = 0.0
    for preset in ("gpt2", "kavi"):
        l_cpu, g_cpu = run("numpy", preset, ids, tgt)
        l_gpu, g_gpu = run("cupy", preset, ids, tgt)
        worst = max(float(np.max(np.abs(g_cpu[k] - g_gpu[k])) /
                          (float(np.max(np.abs(g_cpu[k]))) + 1e-30)) for k in g_cpu)
        worst_all = max(worst_all, worst)
        print(f"{preset}: loss cpu {l_cpu:.12f} gpu {l_gpu:.12f} | worst grad rel diff {worst:.2e}")
        assert abs(l_cpu - l_gpu) < 1e-9 and worst < 1e-8, "CPU/GPU mismatch"
    print("backend parity OK")


if __name__ == "__main__":
    main()
