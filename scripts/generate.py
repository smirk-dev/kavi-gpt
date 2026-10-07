"""Sample text from a trained checkpoint.

    python scripts/generate.py runs/kavi-s1/best.npz --prompt "ROMEO:" --tokens 300
    python scripts/generate.py runs/kavi-s1/best.npz --temperature 0.7 --top-p 0.9 --n 3
    python scripts/generate.py runs/kavi-s1/best.npz --bench      # KV cache vs no cache

The tokenizer is found through the data directory recorded in the checkpoint's config.
"""
import argparse
import sys
import time
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
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=1, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time cached vs uncached decoding")
    args = ap.parse_args()

    B.set_backend("numpy", precision="float32")
    model, cfg = load_model(args.checkpoint)
    tok = load_tokenizer(ROOT / cfg["data"] / "tokenizer.json")
    prompt = np.array([tok.encode(args.prompt)])

    if args.bench:
        for cache in (False, True):
            t = time.time()
            model.generate(prompt, args.tokens, temperature=0, use_cache=cache)
            dt = time.time() - t
            print(f"{'KV cache' if cache else 'no cache'}: {args.tokens / dt:7.1f} tokens/s")
        return

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        out = model.generate(prompt, args.tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, use_cache=not args.no_cache,
                             rng=rng)
        if args.n > 1:
            print(f"----- sample {i + 1} -----")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
