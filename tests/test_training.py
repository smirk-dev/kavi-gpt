"""Phase 3 gate: the full training stack can memorise one batch.

If forward, backward, the optimiser and the schedule are all wired correctly, a model with
far more parameters than the batch has tokens must drive the loss on that batch to ~0. A
bug anywhere (wrong sign, a missing gradient, a stale cache) shows up as a loss that stalls.
"""
import numpy as np
import pytest

from kavi import backend as B
from kavi.model import GPT, GPTConfig
from kavi.optim import SGD, AdamW, clip_grad_norm


def train_one_batch(preset, opt_name, steps):
    B.set_precision("float32")
    cfg = GPTConfig.preset(preset, vocab_size=20, n_embd=32, n_head=4, block_size=16, n_layer=2)
    model = GPT(cfg, seed=0)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 20, size=(4, 17))
    x, y = data[:, :-1], data[:, 1:]
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.0) if opt_name == "adamw" \
        else SGD(model.params(), lr=0.5)
    losses = []
    for _ in range(steps):
        model.zero_grad()
        losses.append(float(model.forward(x, y)[1]))
        model.backward()
        clip_grad_norm(model.params(), 1.0)
        opt.step()
    return losses


@pytest.mark.parametrize("preset", ["gpt2", "kavi"])
def test_adamw_overfits_one_batch(preset):
    losses = train_one_batch(preset, "adamw", 300)
    assert losses[0] > 2.5                       # ≈ ln(20) = 3.0 at init
    assert losses[-1] < 0.05, losses[-10:]


def test_sgd_is_much_slower_than_adamw():
    """The Anton episode-2 lesson, as a test: same steps, SGD is left far behind."""
    sgd = train_one_batch("kavi", "sgd", 150)[-1]
    adam = train_one_batch("kavi", "adamw", 150)[-1]
    assert adam < 0.5 * sgd, (adam, sgd)
