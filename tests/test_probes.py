"""The induction probe runs inside the training loop, so it must leave the model exactly as it
found it (train mode, flash flags) and must index the right attention cells."""
import numpy as np

from kavi import backend as B
from kavi.model import GPT, GPTConfig
from kavi.probes import induction, induction_batch


def test_probe_restores_mode_and_flash_and_indexes_correctly():
    cfg = GPTConfig.preset("kavi", vocab_size=31, n_embd=16, n_head=2, block_size=24, n_layer=2)
    model = GPT(cfg, seed=0)
    model.blocks[1].attn.flash = True
    model.train()
    val = np.random.default_rng(1).integers(0, 31, size=500)
    P = 10
    ids = induction_batch(val, P, 3, np.random.default_rng(2))
    assert ids.shape == (3, 2 * P) and (ids[:, :P] == ids[:, P:]).all()

    r = induction(model, ids)
    assert model.training and model.blocks[0].training
    assert [b.attn.flash for b in model.blocks] == [False, True]
    assert r["ind"].shape == r["prev"].shape == (2, 2)

    # brute force over the cached probabilities of the last block (the probe's forward left them)
    A = B.to_numpy(model.blocks[1].attn.probs)                 # (n, H, 2P-1, 2P-1)
    for h in range(2):
        ind = np.mean([A[b, h, i, i - P + 1] for b in range(3) for i in range(P, 2 * P - 1)])
        prev = np.mean([A[b, h, i, i - 1] for b in range(3) for i in range(P, 2 * P - 1)])
        assert np.isclose(r["ind"][1, h], ind) and np.isclose(r["prev"][1, h], prev)
    assert r["best_ind"] == r["ind"].max()
