"""Tokenise the train/val split.

    python scripts/tokenize_data.py char
    python scripts/tokenize_data.py bpe --vocab 4096

Writes data/<name>/{train.bin,val.bin,meta.json,tokenizer.json}. The tokenizer is trained
on the *training* split only — fitting it on val would leak validation text into the vocab.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kavi.data import write_token_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["char", "bpe"])
    ap.add_argument("--vocab", type=int, default=4096, help="BPE vocabulary size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    val = (ROOT / "data" / "val.txt").read_text(encoding="utf-8")
    t0 = time.time()
    if args.kind == "char":
        from kavi.tokenizers.char import CharTokenizer
        # chars from both splits: a char missing from train would make val un-encodable,
        # and a character inventory leaks essentially nothing about the val text.
        tok = CharTokenizer.train(train + val)
        out = args.out or "data/char"
    else:
        from kavi.tokenizers.bpe import BPETokenizer
        tok = BPETokenizer.train(train, args.vocab, verbose=True)
        out = args.out or f"data/bpe{args.vocab}"
    print(f"trained {args.kind} tokenizer: vocab {tok.vocab_size} in {time.time() - t0:.1f}s")
    meta = write_token_data(ROOT / out, tok, train, val)
    print(f"train {meta['train_tokens']:,} tokens, val {meta['val_tokens']:,} tokens, "
          f"{meta['val_bytes'] / meta['val_tokens']:.2f} bytes/token  -> {out}")


if __name__ == "__main__":
    main()
