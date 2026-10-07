"""Phase 4 gate: BPE is lossless and its incremental trainer matches the naive algorithm."""
from collections import Counter

from kavi.tokenizers.bpe import SPLIT_PATTERN, BPETokenizer, _merge

TEXT = ("To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune, ünïcödé & 1234 under_score\t \n") * 3


def test_split_is_lossless():
    assert "".join(SPLIT_PATTERN.findall(TEXT)) == TEXT


def test_round_trip():
    tok = BPETokenizer.train(TEXT, 300)
    assert tok.vocab_size == 300
    assert tok.decode(tok.encode(TEXT)) == TEXT
    weird = "never seen: Ωmega 🙂 \x00 tabs\t\tand\r\nCRLF"
    assert tok.decode(tok.encode(weird)) == weird      # byte fallback: nothing is unknown


def naive_train(text, vocab_size):
    """Recount everything after every merge — slow but obviously correct."""
    words = Counter(tuple(c.encode()) for c in SPLIT_PATTERN.findall(text))
    words = {k: v for k, v in words.items()}
    merges = []
    while len(merges) < vocab_size - 256:
        counts = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                counts[p] += f
        if not counts:
            break
        # tie-break must mirror the heap: highest count, then the smallest pair
        top = max(counts.values())
        pair = min(p for p, c in counts.items() if c == top)
        merges.append(pair)
        new_id = 255 + len(merges)
        words = {tuple(_merge(list(w), pair, new_id)): f for w, f in words.items()}
    return merges


def test_incremental_trainer_matches_naive():
    assert BPETokenizer.train(TEXT, 330).merges == naive_train(TEXT, 330)


def test_compresses():
    tok = BPETokenizer.train(TEXT, 330)
    assert len(tok.encode(TEXT)) < 0.5 * len(TEXT.encode())
