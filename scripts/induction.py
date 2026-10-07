"""Find induction heads: the copy mechanism behind in-context learning.   docs/13-interpretability.md

    python scripts/induction.py ckpt/kavi-s2.npz

Feed the model a random token sequence of length P, then the *same* sequence again. In the
first copy nothing is predictable (the tokens are random), so the loss is high. In the second
copy, a model that can "look up what came after this token last time" predicts perfectly.

An induction head is the attention head that does the lookup. At position i in the second
copy (current token = x[i-P]), it attends to position i-P+1: the token that *followed* the
earlier occurrence of the current token. So per head we measure:
  * induction score  = mean attention from i to i-P+1   (over second-copy positions)
  * prev-token score = mean attention from i to i-1     (the helper head that makes it work)
A head with uniform attention over ~i keys would score ~1/i; a real induction head scores
tens of percent. Averaged over several random sequences.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kavi import backend as B  # noqa: E402
from kavi.checkpoint import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--period", type=int, default=50)
    ap.add_argument("--n", type=int, default=8, help="random sequences to average over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["uniform", "unigram", "text"], default="unigram",
                    help="uniform: any vocab id; unigram: ids drawn at random from the val split "
                         "(realistic frequencies, no grammar); text: a real val passage")
    args = ap.parse_args()

    B.set_backend("numpy", precision="float32")
    model, cfg = load_model(args.checkpoint)
    for blk in model.blocks:
        blk.attn.flash = False                      # need the explicit probability matrix
    model.eval()
    P, V = args.period, model.cfg.vocab_size
    assert 2 * P <= model.cfg.block_size
    rng = np.random.default_rng(args.seed)
    if args.source == "uniform":
        half = rng.integers(0, V, size=(args.n, P))
    else:
        from kavi.data import TokenData
        val = TokenData(ROOT / cfg["data"]).val
        if args.source == "unigram":
            half = val[rng.integers(0, len(val), size=(args.n, P))]
        else:
            starts = rng.integers(0, len(val) - P, size=args.n)
            half = np.stack([val[s:s + P] for s in starts])
    ids = np.concatenate([half, half], axis=1)       # (n, 2P)

    _, loss_first = model.forward(ids[:, :P], ids[:, 1:P + 1])
    logits, _ = model.forward(ids[:, :-1])           # caches attention for all 2P-1 positions
    logits = B.to_numpy(logits).astype(np.float64)
    logp = logits - logits.max(-1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
    tgt = ids[:, 1:]
    nll = -np.take_along_axis(logp, tgt[..., None], -1)[..., 0]     # (n, 2P-1)
    print(f"{args.source} tokens, period {P}, {args.n} sequences, vocab {V} (uniform guess = {np.log(V):.2f})")
    print(f"  loss on first copy  : {nll[:, :P - 1].mean():.3f} nats")
    print(f"  loss on second copy : {nll[:, P:].mean():.3f} nats   <- in-context copying")

    rows = np.arange(P, 2 * P - 1)                   # query positions in the second copy
    print("\nlayer head | induction (i -> i-P+1) | prev-token (i -> i-1)")
    best = []
    for li, blk in enumerate(model.blocks):
        A = B.to_numpy(blk.attn.probs)               # (n, H, T, T)
        ind = A[:, :, rows, rows - P + 1].mean(axis=(0, 2))
        prev = A[:, :, rows, rows - 1].mean(axis=(0, 2))
        for h in range(A.shape[1]):
            flag = "  <- induction head" if ind[h] > 0.3 else ("  <- previous-token head" if prev[h] > 0.3 else "")
            print(f"  {li:3d}  {h:3d} | {ind[h]:6.3f} | {prev[h]:6.3f}{flag}")
            best.append((ind[h], li, h))
    s, li, h = max(best)
    print(f"\nstrongest induction head: layer {li} head {h} ({s:.1%} of its attention on the "
          f"target; uniform would be ~{1 / (1.5 * P):.1%})")


if __name__ == "__main__":
    main()
