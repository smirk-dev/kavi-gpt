"""Token datasets, batching and the bits-per-byte metric.          docs/09-training.md"""
import json
import math
from pathlib import Path

import numpy as np

from . import backend as B


def load_tokenizer(path):
    path = Path(path)
    kind = json.loads(path.read_text(encoding="utf-8"))["kind"]
    if kind == "char":
        from .tokenizers.char import CharTokenizer
        return CharTokenizer.load(path)
    if kind == "bpe":
        from .tokenizers.bpe import BPETokenizer
        return BPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


class TokenData:
    """A tokenised corpus directory: train.bin, val.bin (uint16), meta.json, tokenizer.json."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.train = np.fromfile(self.dir / "train.bin", dtype=np.uint16).astype(np.int64)
        self.val = np.fromfile(self.dir / "val.bin", dtype=np.uint16).astype(np.int64)
        self.vocab_size = self.meta["vocab_size"]

    def tokenizer(self):
        return load_tokenizer(self.dir / "tokenizer.json")

    def bytes_per_token(self, split="val"):
        return self.meta[f"{split}_bytes"] / self.meta[f"{split}_tokens"]

    def bpb(self, loss_nats, split="val"):
        """Convert mean cross-entropy (nats per token) into bits per byte of UTF-8 text.

        Loss per token is NOT comparable across tokenizers: a BPE token covers ~3-4 chars,
        so predicting it is a harder (and bigger) job than predicting one character.
        Dividing the total information by the number of *bytes* it describes fixes that:
            bpb = (loss / ln 2) / bytes_per_token
        Lower is better; it is literally how many bits per byte a compressor driven by this
        model would need.
        """
        return loss_nats / math.log(2) / self.bytes_per_token(split)

    def batch(self, split, batch_size, block_size, rng):
        """Random windows: x = data[i : i+T], y = data[i+1 : i+T+1] (the next tokens)."""
        data = self.train if split == "train" else self.val
        ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[i:i + block_size] for i in ix])
        y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
        if B.name == "cupy":
            return B.xp.asarray(x), B.xp.asarray(y)
        return x, y


def write_token_data(directory, tokenizer, train_text, val_text):
    """Encode the splits and store them with the metadata bpb needs."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tokenizer.save(d / "tokenizer.json")
    meta = {"vocab_size": tokenizer.vocab_size, "tokenizer": tokenizer.kind}
    for split, text in (("train", train_text), ("val", val_text)):
        ids = np.array(tokenizer.encode(text), dtype=np.int64)
        assert ids.max() < 2 ** 16, "uint16 storage needs vocab < 65536"
        ids.astype(np.uint16).tofile(d / f"{split}.bin")
        meta[f"{split}_tokens"] = int(len(ids))
        meta[f"{split}_bytes"] = len(text.encode("utf-8"))
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
