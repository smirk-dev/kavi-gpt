"""Behavioural probes that run on a live model: shared by scripts/induction.py and train.py.

induction(model, ...) feeds a random sequence of P tokens followed by the *same* P tokens.
The first copy is unpredictable; a model with induction heads predicts the second copy by
looking up "what followed this token last time". Per head we measure the mean attention
from each second-copy position i to i-P+1 (the token after the earlier occurrence; the
induction score) and to i-1 (the previous-token helper). See docs/13-interpretability.md.
"""
import numpy as np

from . import backend as B


def induction_batch(val_tokens, period, n, rng, source="unigram", vocab_size=None):
    """(n, 2P) int array: n random halves, each repeated once."""
    if source == "uniform":
        half = rng.integers(0, vocab_size, size=(n, period))
    elif source == "unigram":                 # val-split token frequencies, no grammar
        half = val_tokens[rng.integers(0, len(val_tokens), size=(n, period))]
    elif source == "text":                    # a real passage, repeated
        starts = rng.integers(0, len(val_tokens) - period, size=n)
        half = np.stack([val_tokens[s:s + period] for s in starts])
    else:
        raise ValueError(f"source={source!r}; expected uniform | unigram | text")
    return np.concatenate([half, half], axis=1).astype(np.int64)


def induction(model, ids):
    """Run the probe on ids from induction_batch. Returns a dict with per-head score arrays
    (L, H) and the headline numbers. Leaves the model in its previous train/eval mode and
    restores each block's flash flag (the probe needs the explicit attention matrix)."""
    P = ids.shape[1] // 2
    assert 2 * P <= model.cfg.block_size
    was_training = model.training
    flash = [blk.attn.flash for blk in model.blocks]
    for blk in model.blocks:
        blk.attn.flash = False
    model.eval()
    try:
        logits, _ = model.forward(B.xp.asarray(ids[:, :-1]))   # caches attention, all 2P-1 positions
        logits = B.to_numpy(logits).astype(np.float64)
        logp = logits - logits.max(-1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
        nll = -np.take_along_axis(logp, ids[:, 1:, None], -1)[..., 0]      # (n, 2P-1)
        rows = np.arange(P, 2 * P - 1)                                    # second-copy queries
        ind, prev = [], []
        for blk in model.blocks:
            A = B.to_numpy(blk.attn.probs)                                # (n, H, T, T)
            ind.append(A[:, :, rows, rows - P + 1].mean(axis=(0, 2)))
            prev.append(A[:, :, rows, rows - 1].mean(axis=(0, 2)))
    finally:
        for blk, f in zip(model.blocks, flash):
            blk.attn.flash = f
        if was_training:
            model.train()
    ind, prev = np.array(ind), np.array(prev)
    li, h = np.unravel_index(ind.argmax(), ind.shape)
    return {"loss_first": float(nll[:, :P - 1].mean()), "loss_second": float(nll[:, P:].mean()),
            "ind": ind, "prev": prev, "best_ind": float(ind[li, h]), "best_head": (int(li), int(h))}
