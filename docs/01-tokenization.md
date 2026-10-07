# 01 · Tokenization: from text to integers

> **What you'll learn.** Why a model needs text cut into integer tokens, how Kavi's two
> tokenizers work (one id per character, and byte-level BPE trained from scratch), and the
> engineering that makes BPE training take seconds instead of hours. You'll also learn the
> metric that lets models with different tokenizers be compared fairly: **bits per byte**.

---

## 1. Why tokenize at all?

A neural network does arithmetic on numbers, so text has to become numbers first. The plan
is to cut the text into pieces from a fixed inventory (the **vocabulary**), give each piece
an integer id, and let the embedding table turn each id into a vector
([02-embeddings](02-embeddings.md)).

How big the pieces are is a real design decision:

* **Characters.** Tiny vocabulary, nothing is ever out of vocabulary, but sequences are long
  and the model has to learn to spell every word one letter at a time.
* **Words.** Short sequences, but the vocabulary is huge and open: a new name or a typo has
  no id.
* **Subwords (BPE).** The middle ground. Frequent words become single tokens (`' the'`,
  `' husband'`), rare words are built from a few pieces, and with a byte-level base nothing
  is ever unknown.

Kavi implements the first and the third, so the comparison can actually be run.

---

## 2. The corpus

`scripts/prepare_data.py` downloads Project Gutenberg eBook #100, *The Complete Works of
William Shakespeare*, strips Gutenberg's header and licence, normalises line endings, and
splits the text ([09-training](09-training.md#1-the-data-split) explains the split). The
result:

| | |
|---|---|
| complete works | 5,359,343 characters, 196,015 lines, **100 distinct characters** |
| train split | 4,823,350 characters (4,879,825 UTF-8 bytes) |
| val split | 535,993 characters (542,795 UTF-8 bytes) |

There are more bytes than characters because the edition uses typographic punctuation:
`’` (21,463 uses in train alone), `‘`, `“ ”`, `—`, `…`, and a few accented letters (`é`, `æ`,
`œ`, …). Each of these takes 2 or 3 bytes in UTF-8. That detail matters later, for
bits-per-byte and for the split regex.

---

## 3. The character tokenizer (`kavi/tokenizers/char.py`)

`CharTokenizer` sorts the set of distinct characters and numbers them `0..V-1`. `encode` is
a dictionary lookup per character and `decode` joins the characters back. That's all of it.

```
"ROMEO" → [stoi['R'], stoi['O'], stoi['M'], stoi['E'], stoi['O']]
```

`scripts/tokenize_data.py char` trains it on **train + val** text. That looks like a leak, but
it's a deliberate choice: a character that appears only in val would make val impossible to
encode (`CharTokenizer.encode` raises `KeyError` on unseen characters, with no fallback),
and the list of which characters exist reveals essentially nothing about the val text.
Result: vocab 100, 4,823,350 train tokens, 535,993 val tokens, 1.013 bytes/token.

---

## 4. Byte-level BPE (`kavi/tokenizers/bpe.py`)

### 4.1 The algorithm

Byte-Pair Encoding builds a vocabulary by repeatedly gluing together the most common
neighbouring pair:

1. Start with the 256 possible byte values as tokens 0–255. Any text, in any language,
   with emoji or control characters, is a sequence of bytes, so **nothing is ever unknown**.
2. Count every adjacent pair of tokens in the corpus.
3. The most frequent pair $(a, b)$ becomes a new token with id $256 + k$, where $k$ is the
   number of merges so far.
4. Replace every occurrence of $(a, b)$ with the new token, and repeat until the vocabulary
   reaches the target size.

For `vocab_size=4096` that's $4096 - 256 = 3840$ merges. The output of training is the
**ordered list of merges**, nothing else: `BPETokenizer.save` writes `{"kind": "bpe",
"merges": [...]}`, and the vocabulary (`self.vocab`, the byte string for each id) is rebuilt
by replaying the merges.

The early merges are the most common pairs in English: the first twenty learned include
`' t'`, `'he'`, `' a'`, `'ou'`, `' the'`, `'nd'`, and `b'\xe2\x80'` (the first two bytes shared
by `’`, `‘`, `“`, `”`, `—`, `…`). With `verbose=True` the trainer prints its progress every 500
merges. The real run printed:

| merge # | new token | pair count when merged |
|---|---|---|
| 500 | `b'li'` | 863 |
| 1000 | `b' husband'` | 351 |
| 1500 | `b'ery'` | 209 |
| 2000 | `b'rows'` | 144 |
| 2500 | `b'OLDIER'` | 108 |
| 3000 | `b'amb'` | 84 |
| 3500 | `b' persu'` | 69 |

`OLDIER` is a lovely Shakespeare-specific token: speaker names are written in capitals
(`SOLDIER.`, `FIRST SOLDIER.`), so capitalised fragments get frequent enough to earn their
own ids. By the end, pairs are being merged at counts below 70. The vocabulary is starting
to memorise word pieces that are rare in a 4.9 MB corpus.

### 4.2 Why bytes, not characters?

A character-level BPE base would need every character that could ever appear in its base
vocabulary. Unicode has around 150,000. With bytes, the base is exactly 256 symbols and any
string encodes. The cost: a character outside the merges is spelled as 2–4 raw bytes, and a
single token may hold half a character. You'll see this below with `café`, which encodes as
`'ca' 'f' b'\xc3' b'\xa9'`, because the two bytes of `é` were never merged.

### 4.3 Pre-tokenisation: merges must not cross chunk boundaries

Running BPE on the raw byte stream would let it learn tokens like `"e. T"` that glue the
end of one sentence to the start of the next. Such tokens waste vocabulary on accidents of
adjacency and break the reuse of word pieces. So the text is first **split into chunks**,
and merges are only ever counted and applied *inside* a chunk.

This also gives a big speed-up. Training only needs each **distinct** chunk once, weighted
by how often it occurs. The train split has 1,297,859 chunks but only **35,463 distinct**
ones (266,172 bytes in total), so BPE works on about 5% of the raw data.

### 4.4 The split regex, piece by piece

GPT-2's split pattern uses the third-party `regex` module for `\p{L}` (any letter). Kavi
rewrites it with only the standard library's `re`:

```python
SPLIT_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
```

At each position, `re.findall` tries the alternatives left to right and takes the first
that matches:

| # | Alternative | Matches | Example |
|---|---|---|---|
| 1 | `'(?:s\|t\|re\|ve\|m\|ll\|d)` | an ASCII-apostrophe contraction | `'s`, `'ll`, `'t` |
| 2 | ` ?[^\W\d_]+` | optional single space + a run of letters | `' husband'`, `'ROMEO'` |
| 3 | ` ?\d+` | optional space + a run of digits | `' 1599'` |
| 4 | ` ?(?:[^\s\w]\|_)+` | optional space + a run of symbols (and `_`) | `' --'`, `'!'`, `'’'` |
| 5 | `\s+(?!\S)` | whitespace not followed by a non-space | trailing whitespace |
| 6 | `\s+` | any other whitespace run | a lone `'\n'` |

**Why `[^\W\d_]` means "any letter".** `\w` is "word character": letters, digits and `_`.
`\W` is its complement. The negated class `[^\W\d_]` keeps everything that is *not* a
non-word character, *not* a digit and *not* an underscore. That is word characters minus
digits minus underscore, which leaves the letters (in every script, since `re` is
Unicode-aware for `str` patterns). Strictly it also keeps the few numeric symbols that
Python counts as alphanumeric but not decimal, such as `²` or `½`: `"x² ½"` splits as
`['x²', ' ½']`. That is harmless.

**Why alternative 5 exists.** Take `"\n    THE"`. Without it, `\s+` would grab all five
whitespace characters and `THE` would start a chunk without its leading space. The negative
lookahead `(?!\S)` makes the whitespace run stop one short, leaving the last space for
alternative 2. The real encoder gives `'\n   '`, `' T'`, `'HE'`: the word keeps its space,
just as `' the'` does in running prose.

**Why the split is lossless.** Every single character matches at least one alternative:
whitespace matches 6, a letter matches 2, a decimal digit matches 3, `_` and every
remaining non-word non-space character match 4. So at every position `findall` finds a
non-empty match and never skips a character, which means `"".join(chunks) == text` exactly.
`tests/test_bpe.py::test_split_is_lossless` checks this on text containing accents, tabs,
digits and an underscore.

**A Shakespeare quirk.** This edition uses the typographic apostrophe `’` (23,736 times in
train) and the ASCII `'` exactly once. So alternative 1 almost never fires, and `What’s`
splits as `'What'`, `'’'`, `'s'`. Nothing breaks (BPE simply learns `’` and `s` as separate
frequent tokens), but GPT-2's contraction rule is decorative on this corpus.

### 4.5 Training fast: incremental pair counts + a lazy max-heap

**The naive algorithm** recounts every pair after every merge: for each of the $M$ merges,
scan all chunks. That costs $O(M \times \text{corpus})$. Even on the deduplicated chunks
that's $3840 \times 266{,}172 \approx 10^9$ Python-level steps. `naive_train` in
`tests/test_bpe.py` is exactly this, and it's only usable on tiny text.

**The observation:** a merge of $(a, b)$ only changes the chunks that contain $(a, b)$. Every
other pair count stays the same. `BPETokenizer.train` keeps three structures:

* `pair_count[pair]`: the current weighted count of each pair (each occurrence counts once
  per copy of its chunk in the corpus, i.e. `+= freq`);
* `where[pair]`: the set of chunk indices that contain that pair;
* `heap`: a max-heap of `(-count, pair)` (Python's `heapq` is a min-heap, hence the minus).

One merge step:

1. **Pop** the top of the heap. If its stored count no longer matches `pair_count`, it's a
   **stale** entry left over from before some earlier merge changed that pair's count. Skip
   it. This is the "lazy" part: instead of updating entries inside the heap, which `heapq`
   can't do, we push fresh entries and discard old ones when they surface.
2. Record the merge, `new_id = 256 + len(merges)`.
3. For each chunk in `where[pair]`: subtract all of its pairs' counts, rewrite it with
   `_merge`, then add back the pairs of the rewritten chunk (and register the chunk in
   `where` for them). Remember every pair whose count was touched.
4. Push a fresh `(-count, pair)` for every touched pair that still has a positive count.

Early merges like `' t'` touch thousands of chunks, but later ones touch only a handful. Each
heap operation costs $O(\log n)$. The whole 3840-merge run took **9.2 s** on the laptop
(3.2 s on Kaggle's CPU).

Two correctness details:

* **Deterministic ties.** Heap entries are tuples, so equal counts are broken by the
  smaller pair. `naive_train` uses the same rule ("highest count, then the smallest pair"),
  which lets `tests/test_bpe.py::test_incremental_trainer_matches_naive` demand that the two
  produce **identical merge lists**.
* **`where` is a superset.** A chunk stays in `where[p]` even after a merge removes `p` from
  it. Processing such a chunk is a no-op (subtract its pairs, merge nothing, add them back),
  so this costs a little time but keeps the code simple and still correct.

### 4.6 Encoding: apply merges by rank

To encode new text, `encode` splits it with the same regex, and `_encode_chunk` turns each
chunk into bytes and repeatedly applies **the earliest-learned merge present** in the chunk:

```python
pair = min(zip(w, w[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
```

`ranks[pair]` is the merge's position in the learned list. When no adjacent pair has a rank,
the chunk is finished. Applying merges in training order reproduces exactly the
segmentation the trainer would have produced, so the model sees tokens with the same
statistics it was trained on. Results are cached per distinct chunk (`self._cache`), so
encoding the 4.8 M-character train split takes about 1.2 s.

```
"ROMEO: But soft!" → [2061, 58, 885, 2527, 33] = 'ROMEO' ':' ' But' ' soft' '!'
```

### 4.7 Decoding, and `errors="replace"`

`decode` concatenates each id's byte string and decodes the result as UTF-8. For any
sequence produced by `encode` this is exact. But a **sampled** sequence can be invalid
UTF-8: the model can emit a lone continuation byte, or a sample can stop halfway through
`—`. `errors="replace"` turns such bytes into `�` (U+FFFD) instead of raising, so generation
never crashes on a bad byte. Try `tok.decode([195])`: you get `'�'`.

### 4.8 No leakage

`scripts/tokenize_data.py bpe` trains on `data/train.txt` **only**. Merges learned from val
would encode val text more efficiently, giving a lower loss and a better bpb for free, and
leak information about the held-out text into the model's input format. The resulting
`data/bpe4096/meta.json`:

| | train | val |
|---|---|---|
| tokens | 1,578,753 | 179,413 |
| bytes | 4,879,825 | 542,795 |
| bytes / token | 3.09 | **3.03** |

Val compresses slightly worse than train, which is a small, honest signature of
"merges were fitted to train". A full-corpus check confirms `decode(encode(text)) == text`
for both splits, and that the stored `.bin` files equal a fresh encode.

---

## 5. The vocabulary-size trade-off

Moving from $V = 100$ (chars) to $V = 4096$ (BPE) changes three things at once:

1. **Sequence length.** At 3.03 bytes/token a 256-token context covers about 775 bytes of
   text instead of 256. Attention costs $O(T^2)$, so seeing more text per position is a big
   win.
2. **Embedding parameters.** The table is $V \times C$: $4096 \times 256 = 1{,}048{,}576$
   parameters, about 18% of the 5.76 M-parameter BPE model (counted once thanks to tying).
   The output softmax also costs $O(B T V)$ per step.
3. **Data per token.** The train split is fixed. As $V$ grows there are fewer tokens in
   total (1.58 M instead of 4.82 M), and they're spread over more types. In the BPE train
   data 269 ids never occur (mostly raw bytes that no ASCII text uses), 418 occur fewer than
   10 times, and the median id occurs only 101 times. A rarely seen token gets a poorly
   trained embedding.

So bigger isn't automatically better on a small corpus. 4096 is a guess at the sweet spot
for a ~5 MB corpus. Anton used 10k on a much smaller corpus, which makes problem 3 worse.

---

## 6. Loss per token is not comparable

The model's loss is the average cross-entropy **per token**, in nats. A BPE token carries
about 3 bytes of text and a char token about 1, so predicting a BPE token is a bigger task
and its loss is naturally larger. A char model at 1.6 nats/token and a BPE model at
4.5 nats/token might be equally good. We need a unit that doesn't depend on how the text
was cut.

## 7. Bits per byte: the fair metric

The idea is to measure the total surprise over a piece of text and divide it by the
**number of bytes** of that text, not the number of tokens.

Let a text of $N_{\text{bytes}}$ bytes be encoded as $N_{\text{tok}}$ tokens, and let the
mean loss be $\ell$ nats per token. Then:

* The total surprise is $N_{\text{tok}} \cdot \ell$ nats.
* In bits that is $N_{\text{tok}} \cdot \ell / \ln 2$ (since $1 \text{ nat} = 1/\ln 2$ bits).
* Divided over the bytes it describes:

$$
\text{bpb} \;=\; \frac{N_{\text{tok}}\,\ell}{\ln 2 \cdot N_{\text{bytes}}}
\;=\; \frac{\ell}{\ln 2 \cdot (N_{\text{bytes}}/N_{\text{tok}})}
\;=\; \frac{\ell / \ln 2}{\text{bytes per token}} .
$$

This is `TokenData.bpb` in `kavi/data.py`, where `bytes_per_token` comes from `meta.json`
(val split by default). The quantity has a concrete meaning: an arithmetic coder driven by
the model's predictions would need about this many bits to store each byte. Plain UTF-8
uses 8, and a good compressor like `xz` reaches around 2–2.5 on English.

**Worked example (real numbers).** The CPU char run's step-0 val loss was 4.5829 nats and
char val bytes/token is $542{,}795 / 535{,}993 = 1.0127$:

$$
\text{bpb} = \frac{4.5829 / 0.6931}{1.0127} = 6.529,
$$

which is exactly the `val_bpb` in `runs/char_kavi/log.jsonl`. The uniform guess gives a
useful ceiling for each tokenizer: $\ln 100/\ln 2/1.0127 = 6.56$ bpb for char and
$\ln 4096/\ln 2/3.0254 = 3.97$ bpb for BPE-4096. Note that even an untrained BPE model
starts at a better bpb, because the tokenizer has already done some of the compression.

---

## 8. How we know it's right

`tests/test_bpe.py`:

* `test_split_is_lossless`: `"".join(findall(text)) == text`.
* `test_round_trip`: train a 300-token vocab, round-trip the training text, then round-trip
  unseen text with `Ω`, an emoji, `\x00`, tabs and `\r\n`. This is byte fallback in action.
* `test_incremental_trainer_matches_naive`: the fast trainer and the obviously correct
  slow one produce the same 74 merges.
* `test_compresses`: BPE output is less than half as many tokens as bytes.

## 9. Pitfalls

* **Training the tokenizer on val** gives a quietly optimistic bpb. Train it on train only.
* **Comparing nats/token across tokenizers.** Always compare bpb.
* **`bytes_per_token` is a split average.** `bpb()` uses the whole val split's ratio, while
  an eval uses a sample of windows. The conversion is exact for the full split and a very
  close approximation for the sample.
* **The char tokenizer has no fallback.** Encoding a prompt containing a character not in
  its 100 (say `é` typed with a combining accent, or an emoji) raises `KeyError`. BPE never
  fails.
* **uint16 storage.** `write_token_data` stores ids as `uint16` and asserts $V < 65536$.

## 10. Try it yourself

1. **Watch BPE learn.** Run `BPETokenizer.train(open("data/train.txt", encoding="utf-8").read(), 1024, verbose=True)`.
   *Expected:* under a few seconds. The merge-500 milestone prints `b'li'` (count 863), the
   same as in the 4096 run, because training is deterministic and a smaller vocab is just a
   prefix of a larger one.
2. **Find the multi-byte splits.** Encode `"café — naïve"` with the trained 4096 tokenizer
   and print `[tok.vocab[i] for i in ids]`. *Expected:* `é` and `ï` come out as two raw
   bytes each, while `—` is a single token (`b'\xe2\x80\x94'`), because the em dash is
   frequent in Shakespeare.
3. **Time the naive trainer.** Run `naive_train` from `tests/test_bpe.py` on the first
   200,000 characters of train with `vocab_size=512`, and compare with
   `BPETokenizer.train` on the same text. *Expected:* identical merges, with the naive one
   taking many times longer, and the gap growing with text size.
4. **bpb by hand.** Take any `val_loss` from a BPE run's `log.jsonl` and recompute
   `val_bpb` with $\ell / \ln 2 / 3.0254$. *Expected:* agreement to the printed digits (for
   example, the Kaggle `kavi` probe's 4.5208 nats gives 2.156 bpb).
