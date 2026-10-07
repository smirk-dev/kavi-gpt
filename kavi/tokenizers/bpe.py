"""Byte-level Byte-Pair Encoding, trained from scratch.          docs/01-tokenization.md

Start from the 256 possible bytes (so *any* text is encodable — no unknown tokens), then
repeatedly merge the most frequent adjacent pair into a new token:

    1. count every adjacent pair of tokens in the corpus
    2. the most frequent pair (a, b) becomes new token id 256 + k
    3. replace every occurrence of (a, b) with it; repeat until the vocab is full

Two engineering details make this fast and well-behaved:

* **Pre-tokenisation.** Text is first split into chunks (words with their leading space,
  numbers, punctuation runs, whitespace) and merges never cross a chunk boundary. Otherwise
  BPE would learn junk tokens like "e. T" that glue the end of one word to the next. It
  also means we only process each *distinct* chunk once, weighted by its count.

* **Incremental pair counts.** The naive algorithm recounts every pair in the corpus after
  each merge: O(merges × corpus). We keep a running pair -> count table plus an index
  pair -> {chunks containing it}; a merge only touches the chunks that contain the merged
  pair, and a lazy max-heap finds the next best pair in O(log n).
"""
import heapq
import json
import re
from collections import Counter, defaultdict

# GPT-2-style split using only the stdlib `re` (no `regex` module needed):
#   contractions | optional space + letters | optional space + digits |
#   optional space + other symbols (incl. "_") | trailing whitespace | whitespace
# [^\W\d_] is "any Unicode letter". Every character falls into exactly one alternative,
# so the split is lossless — tested by an exact encode/decode round trip.
SPLIT_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def _merge(word, pair, new_id):
    out, i, a, b = [], 0, pair[0], pair[1]
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class BPETokenizer:
    kind = "bpe"

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]               # in learned order
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    # -- training ----------------------------------------------------------------------
    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_count = defaultdict(int)
        where = defaultdict(set)                               # pair -> chunk indices
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_count[pair] += f
                where[pair].add(wi)
        # heapq is a min-heap: store -count. Ties break on the pair itself -> deterministic.
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)

        merges = []
        while len(merges) < vocab_size - 256 and heap:
            neg, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg or -neg <= 0:
                continue                                       # stale heap entry: skip
            new_id = 256 + len(merges)
            merges.append(pair)
            changed = set()
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                if len(w) < 2:
                    continue
                # remove this chunk's old pairs, merge, add its new pairs
                for p in zip(w, w[1:]):
                    pair_count[p] -= f
                    changed.add(p)
                w = _merge(w, pair, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    pair_count[p] += f
                    where[p].add(wi)
                    changed.add(p)
            del pair_count[pair], where[pair]
            for p in changed:
                if p in pair_count and pair_count[p] > 0:
                    heapq.heappush(heap, (-pair_count[p], p))
            if verbose and len(merges) % 500 == 0:
                tok = cls(merges)
                print(f"  merge {len(merges):5d}: {tok.vocab[new_id]!r} (count {-neg})")
        return cls(merges)

    # -- encode / decode ---------------------------------------------------------------
    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        w = list(chunk.encode("utf-8"))
        while len(w) >= 2:
            # apply the earliest-learned merge present — exactly the order training used
            pair = min(zip(w, w[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            w = _merge(w, pair, 256 + rank)
        self._cache[chunk] = w
        return w

    def encode(self, text):
        out = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        # errors="replace": a sampled sequence can end mid-way through a multi-byte char
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i):
        return self.vocab[i].decode("utf-8", errors="replace")

    # -- persistence -------------------------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": self.kind, "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])
