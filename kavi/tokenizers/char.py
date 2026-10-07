"""Character-level tokenizer: one token per distinct character.     docs/01-tokenization.md

The simplest possible tokenizer, and the baseline BPE has to beat. Tiny vocabulary
(~100 symbols) but long sequences: the model must spell every word letter by letter.
"""
import json


class CharTokenizer:
    kind = "char"

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def train(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": self.kind, "chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])
