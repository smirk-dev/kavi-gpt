"""See what every attention head looks at.          docs/13-interpretability.md

    python scripts/attention_viz.py runs/kavi-s1/best.npz --text "ROMEO: But soft, what light" \
        --out docs/img/attention.png

Runs one forward pass, grabs each layer's attention probabilities (the (T,T) matrix each
head produced, cached by CausalSelfAttention.forward), and draws a grid: rows = layers,
columns = heads. Row i of a heatmap shows where token i looked. Common patterns to spot:
  * "previous token" heads — a bright line just below the diagonal
  * "first token" / sink heads — a bright first column (a place to dump attention)
  * heads that jump back to the same word earlier in the text (induction-like copying)
Also prints, per head, its average attention distance and entropy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kavi import backend as B  # noqa: E402
from kavi.checkpoint import load_model  # noqa: E402
from kavi.data import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--text", default="ROMEO:\nBut soft, what light through yonder window breaks?")
    ap.add_argument("--out", default="docs/img/attention.png")
    args = ap.parse_args()

    B.set_backend("numpy", precision="float32")
    model, cfg = load_model(args.checkpoint)
    if model.cfg.attn == "flash":
        model.cfg.attn = "naive"
        for blk in model.blocks:
            blk.attn.flash = False          # need the explicit probability matrix
    tok = load_tokenizer(ROOT / cfg["data"] / "tokenizer.json")
    ids = tok.encode(args.text)[: model.cfg.block_size]
    labels = [tok.decode([i]).replace("\n", "\\n") for i in ids]
    model.eval()
    model.forward(np.array([ids]))
    maps = [B.to_numpy(blk.attn.probs[0]) for blk in model.blocks]   # each (H, T, T)

    L, H, T = len(maps), maps[0].shape[0], len(ids)
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    print(f"{L} layers x {H} heads, {T} tokens")
    print("layer head | mean look-back distance | entropy (nats; low = focused)")
    for li, m in enumerate(maps):
        for h in range(H):
            p = m[h]
            look = float((p * dist).sum(-1).mean())
            ent = float(-(p * np.log(p + 1e-12)).sum(-1).mean())
            print(f"  {li:3d}  {h:3d} | {look:6.2f} | {ent:5.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(L, H, figsize=(2.2 * H, 2.2 * L), squeeze=False)
    for li in range(L):
        for h in range(H):
            ax = axes[li][h]
            ax.imshow(maps[li][h], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"layer {li}")
            if li == 0:
                ax.set_title(f"head {h}", fontsize=9)
    fig.suptitle(f"attention patterns — {Path(args.checkpoint).parent.name}\n"
                 f"text: {' '.join(labels)[:120]}", fontsize=9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
