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
from kavi.data import TokenData  # noqa: E402
from kavi.probes import induction, induction_batch  # noqa: E402


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
    P, V = args.period, model.cfg.vocab_size
    val = None if args.source == "uniform" else TokenData(ROOT / cfg["data"]).val
    ids = induction_batch(val, P, args.n, np.random.default_rng(args.seed), args.source, V)
    r = induction(model, ids)
    print(f"{args.source} tokens, period {P}, {args.n} sequences, vocab {V} (uniform guess = {np.log(V):.2f})")
    print(f"  loss on first copy  : {r['loss_first']:.3f} nats")
    print(f"  loss on second copy : {r['loss_second']:.3f} nats   <- in-context copying")

    print("\nlayer head | induction (i -> i-P+1) | prev-token (i -> i-1)")
    for li in range(r["ind"].shape[0]):
        for h in range(r["ind"].shape[1]):
            ind, prev = r["ind"][li, h], r["prev"][li, h]
            flag = "  <- induction head" if ind > 0.3 else ("  <- previous-token head" if prev > 0.3 else "")
            print(f"  {li:3d}  {h:3d} | {ind:6.3f} | {prev:6.3f}{flag}")
    li, h = r["best_head"]
    print(f"\nstrongest induction head: layer {li} head {h} ({r['best_ind']:.1%} of its attention on the "
          f"target; uniform would be ~{1 / (1.5 * P):.1%})")


if __name__ == "__main__":
    main()
