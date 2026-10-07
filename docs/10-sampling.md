# 10 · Sampling and the KV cache

> **What you'll learn.** A trained model gives a probability distribution over the next token,
> and turning that into text is a separate set of design decisions. This chapter derives
> what temperature does to a distribution, shows exactly which tokens top-k and top-p (nucleus)
> keep in `kavi/model.py::sample`, and then builds the **KV cache**: why naive generation wastes
> almost all its work, what gets cached, how prefill and decode differ, and the one place where
> the cached and uncached paths intentionally give different results.

Prerequisites: [attention](05-attention.md) (especially RoPE's `offset`), [the loss](07-loss.md).

---

## 1. Autoregressive generation

The model was trained to predict token $t+1$ from tokens $\le t$. To generate, we use that
one step at a time:

```
ids = prompt
repeat:
    logits = model(ids)[last position]     # (B, V)
    next   = sample(logits)                 # pick one token per row
    ids    = ids + [next]
```

That's `GPT.generate`. Each new token goes back in as input, which is why the model is called
*autoregressive*. The context is capped at `block_size` tokens: without the cache, the model
only ever sees `ids[:, -block_size:]`.

From the logits $z \in \mathbb{R}^V$, the model's distribution is $p_i = e^{z_i}/\sum_j e^{z_j}$.
`sample` decides how to draw from it.

## 2. Temperature

Divide the logits by $\tau > 0$ before the softmax:

$$
p_i(\tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}} .
$$

**Derivation of the effect.** Compare two tokens:

$$
\frac{p_i(\tau)}{p_j(\tau)} = e^{(z_i - z_j)/\tau} = \left(\frac{p_i(1)}{p_j(1)}\right)^{1/\tau} .
$$

Equivalently $p_i(\tau) \propto p_i^{1/\tau}$: temperature raises every probability to the
power $1/\tau$ and renormalises.

- $\tau < 1$: the exponent $1/\tau > 1$ widens every gap, so likely tokens get likelier
  (**sharper**: safer, more repetitive text).
- $\tau > 1$: gaps shrink toward 1 (**flatter**: more surprising, more errors). As
  $\tau\to\infty$ every ratio goes to 1, i.e. uniform.
- $\tau \to 0$: if $z_a$ is the unique maximum, then for every $j \ne a$,
  $p_j/p_a = e^{-(z_a - z_j)/\tau} \to 0$, so $p \to$ one-hot on $\arg\max z$. That's **greedy
  decoding**.

Example, $p = (0.5, 0.3, 0.15, 0.05)$:

| $\tau$ | 0.5 | 1 | 2 |
|---|---|---|---|
| $p(\tau)$ | (0.685, 0.247, 0.062, 0.007) | (0.5, 0.3, 0.15, 0.05) | (0.379, 0.294, 0.208, 0.120) |

Dividing by 0 is undefined, so `sample` treats `temperature == 0` as a special case and returns
`logits.argmax(axis=-1)` directly.

Real samples from the char model (`runs/char_kavi/best.npz`, 1000 steps, val bpb 2.275),
prompt `ROMEO:`:

```
τ = 0 (greedy)    ROMEO: and I will not the man of my lord.
                  CLOWN.
                  I will not the man of the man of my lord.
τ = 0.8, p = 0.95 ROMEO: a brother,
                  The the Partia's noble fortune, by the dog.
τ = 1.5           ROMEO: TaQuoot?yi
                  THILPHEM Georf't. That disteoned unmatter
```

Greedy decoding falls into a loop ("the man of the man of"): once a phrase is the most likely
continuation of itself, argmax repeats it forever. A high temperature gives invented words
and random capitals. The middle setting (the `scripts/generate.py` default) reads most like
Shakespeare.

## 3. Top-k

Keep only the $k$ largest logits and sample among them:

```python
kth = np.sort(z, axis=-1)[:, -min(top_k, V)][:, None]     # k-th largest value per row
z = np.where(z < kth, -np.inf, z)
```

$e^{-\infty} = 0$, so the dropped tokens get probability exactly 0 after the softmax, and the
kept ones are renormalised. The point is to cut the **long tail**: with 100 characters (or
4096 BPE tokens), each tail token is unlikely, but together they can hold a few percent of the
mass, and an occasional draw from the tail derails the text. Top-k is applied to the
temperature-scaled logits (temperature first, then filtering). With ties at the $k$-th value,
`z < kth` keeps all of them, so slightly more than $k$ tokens can survive.

## 4. Top-p (nucleus)

A fixed $k$ is crude. When the model is confident, 2 tokens may hold 99% of the mass; when
it's unsure, 40 tokens may be plausible. Top-p adapts: keep the **smallest set of most-likely
tokens whose total probability reaches $p$**.

Sort probabilities in decreasing order, $p_{(1)} \ge p_{(2)} \ge \dots$, with cumulative sums
$C_r = \sum_{s\le r} p_{(s)}$. The code's rule is

```python
drop = (np.cumsum(sorted_p, axis=-1) - sorted_p) >= top_p     # C_{r-1} >= p
```

meaning **drop token $r$ if the mass strictly before it, $C_{r-1}$, already reaches $p$**.

**Claim:** this keeps exactly the smallest prefix whose mass is $\ge p$. Let $k^*$ be the
smallest $r$ with $C_r \ge p$.

- For $r \le k^*$: $C_{r-1} \le C_{k^*-1} < p$ (by minimality of $k^*$), so token $r$ is kept.
- For $r > k^*$: $C_{r-1} \ge C_{k^*} \ge p$, so token $r$ is dropped.

The top token always survives because $C_0 = 0 < p$ for any $p > 0$. So even a tiny `top_p`
can't empty the distribution. The kept probabilities are put back in the original order
(`put_along_axis`) and renormalised.

`test_sampling_filters` uses $p = (0.5, 0.3, 0.15, 0.05)$: with `top_p=0.7`, the mass before
token 2 is $0.5 < 0.7$ (kept) and before token 3 is $0.8 \ge 0.7$ (dropped), so the samples are
exactly $\{0, 1\}$. With `top_p=0.4`, only token 0 is kept, since the mass before token 1 is $0.5 \ge 0.4$.

Order of operations in `sample`: **temperature → top-k (on logits) → softmax → top-p (on
probabilities) → draw** with `rng.choice`, once per batch row. Because top-p runs after
temperature, the same `top_p` keeps fewer tokens at low $\tau$ (where the distribution is sharper).

## 5. Why naive generation is slow

To generate token $n$ without a cache, `generate(use_cache=False)` runs the full forward pass
over the whole window of $n$ tokens (up to `block_size`) and throws away every output except
the last. Per layer that costs:

- linear layers (qkv, proj, MLP): $O(n\,C^2)$, for all $n$ positions;
- attention scores and weighted sum: $O(n^2 C)$.

But the keys and values of tokens $1..n-1$ are **the same as last step**: the causal mask
means a token's $k$ and $v$ (and its whole hidden state) depend only on itself and earlier
tokens, and those didn't change. Summing over $N$ generated tokens ($N \le$ `block_size`):

$$
\text{no cache: } \sum_{n=1}^{N} \big(nC^2 + n^2C\big) = O(N^2C^2 + N^3 C), \qquad
\text{cache: } \sum_{n=1}^{N} \big(C^2 + nC\big) = O(NC^2 + N^2 C).
$$

Attention goes from cubic to quadratic overall, and from $O(n^2)$ to $O(n)$ per token. Past
`block_size` the uncached cost per token stops growing, but each token still recomputes the
entire window.

**Measured** with `python scripts/generate.py <char checkpoint> --bench` (greedy, on CPU):
**86.2 tokens/s without cache, 623.4 tokens/s with cache, 7.2× faster.** The speedup is smaller
than the FLOP ratio because at one token per step the cached path is dominated by Python and
NumPy call overhead, not arithmetic.

## 6. The KV cache

### 6.1 What is stored

Each `CausalSelfAttention` owns a cache, created by `reset_cache(batch)`:

```python
shape = (batch, n_head, max_len, head_dim)          # (B, H, T_max, D)
cache = {"k": zeros(shape), "v": zeros(shape), "len": 0}
```

That's one `k` and one `v` buffer per layer, preallocated to `block_size` positions, plus the
number of filled positions. Memory is $2 \cdot B \cdot T_{\max} \cdot C$ numbers per layer. For the
char model ($C = 128$, $T_{\max} = 128$, 4 layers, float32) that's 512 KiB. Keys are stored
**after RoPE**: a key's rotation depends only on its own position, so it never needs recomputing.
Queries aren't cached, since each query is used once, by its own token.

### 6.2 `forward_cached`: prefill and decode with one function

`forward_cached(x)` takes only the **new** tokens `x` of shape `(B, t, C)`:

1. `start = cache["len"]`, the absolute position of the first new token.
2. Compute q, k, v for the $t$ new tokens only, and apply RoPE with `offset=start` (so token $i$ is
   rotated for position `start + i`, [chapter 05 §11.6](05-attention.md)).
3. Write k, v into slots `start : start+t`, and set `len = end = start + t`.
4. Attend: scores `(B, H, t, end)` between the $t$ new queries and all `end` cached keys.
5. Mask with

   ```python
   allowed = arange(end)[None, :] <= (start + arange(t))[:, None]      # (t, end)
   ```

   so new token $i$, which sits at absolute position `start + i`, may see keys `0 .. start+i`.

Two regimes use this:

- **Prefill** (`start = 0`, $t$ = prompt length): `allowed` is exactly the $t\times t$
  lower-triangular causal mask, so it's ordinary masked attention over the prompt, done in one
  batched pass.
- **Decode** ($t = 1$): `allowed` is a single row of all True (the new token may see every
  cached key, itself included). This is where the savings come from.

The mask also handles anything in between, e.g. feeding 5 tokens after 7 are cached.
`test_kv_cache_matches_full_forward` does exactly that: a 5-token prefill followed by single steps.

### 6.3 The model-level loop: `GPT._forward_cached`

```python
x = wte[ids]                                       # new tokens only
if wpe is not None: x = x + wpe[start : start+t]   # learned positions need the offset too
for block in blocks: x = block.forward_cached(x)   # attn uses the cache; norms/MLP are per-token
return norm_f(x)[:, -1] @ wteᵀ                     # logits of the last new token only
```

The norms and MLP act on each token independently, so they need no cache. Only attention
mixes positions. Dropout and flash attention don't appear on this path: generation runs in eval
mode, and decoding one token at a time doesn't need tiling.

### 6.4 When the cache is full: sliding re-prefill

The cache has `block_size` slots. `generate` tracks `pos`, the number of filled slots:

```python
if i == 0 or pos >= T_max:
    ctx = ids[:, -(T_max // 2 if i else T_max):]      # first time: up to T_max; later: last half
    reset every layer's cache; logits = _forward_cached(ctx, 0); pos = len(ctx)
else:
    logits = _forward_cached(ids[:, -1:], pos); pos += 1
```

When the cache fills, we restart it from the **last `block_size // 2` tokens** (64 for our
models) at positions 0..63, then continue one token at a time. Why half rather than
`block_size - 1`? Re-prefilling 127 tokens to gain room for one would mean a full prefill on
every single step after the first 128, which loses the cache's whole advantage. Keeping half
costs one 64-token prefill per 64 generated tokens.

**Consequence: past `block_size`, cached and uncached generation differ, on purpose.** Without
the cache, every token is predicted from the last `block_size` tokens. With the cache, right after
a re-prefill the model sees only 64 tokens of context, growing back to 127 before the next
refill. Different context gives different logits, which eventually gives different text. In a
quick check with `block_size=12` and a 3-token prompt, greedy cached and uncached outputs
matched for the first 13 tokens and diverged at index 13, the first token predicted after
the cache filled. That's why the equivalence tests stay inside `block_size`, and the
long-generation test only checks that it runs.

(With RoPE, restarting the positions at 0 by itself changes nothing, because scores depend only on
relative positions. The difference comes from the shorter context. With learned positions,
the uncached path also always numbers its window from 0, so the same holds.)

## 7. How we know it's right

- **`tests/test_inference.py::test_kv_cache_matches_full_forward`** (both presets, so both RoPE
  offsets and `wpe` offsets): run the full forward on 12 tokens, then a 5-token prefill plus 7
  single-token decodes through the cache. Every cached logit equals the corresponding
  full-forward logit to `rtol=1e-9`. This tests the cache writes, the `allowed` mask, and the
  position offsets in one go.
- **`test_greedy_generation_same_with_and_without_cache`**: 3-token prompt + 9 greedy tokens =
  12 = `block_size`, identical token ids with and without cache.
- **`test_long_generation_runs_past_block_size`**: 40 tokens with `block_size=12`, which forces
  several re-prefills. It checks only the shape (section 6.4).
- **`test_sampling_filters`**: greedy picks token 0; `top_k=2` yields exactly $\{0,1\}$ over 200
  draws; `top_p=0.7` yields $\{0,1\}$; `top_p=0.4` yields $\{0\}$.

## 8. Pitfalls

- **Forgetting the RoPE / `wpe` offset** in decode: every new token is treated as position 0.
  Outputs look plausible but are wrong. The cache-equivalence test catches it.
- **Caching keys before RoPE** and rotating them again on every step: correct but wasteful, and
  easy to rotate with the wrong position.
- **Wrong mask during multi-token prefill**: a decode-style "allow everything" mask lets prompt
  tokens see later prompt tokens. In layer 1 the last position would still come out right (it
  sees everything anyway), but from layer 2 on it reads keys and values computed from states
  that peeked, so both the logits and the cache are wrong.
- **Comparing cached vs uncached beyond `block_size`** and calling the difference a bug
  (section 6.4).
- **Greedy for creative text** loops (section 2). For Shakespeare, use $\tau \approx 0.7$–$0.9$ with top-p 0.9–0.95.
- **`top_p = 1.0` isn't exactly "off".** Float round-off can push the running sum to
  $\ge 1$ just before the last few near-zero tokens, which then get dropped. That's harmless,
  but pass `top_p=None` to disable it for certain.

## 9. Try it yourself

1. **Temperature by formula.** Take `logits = log([0.5, 0.3, 0.15, 0.05])`, apply
   `softmax(logits / τ)` for τ = 0.5 and 2, and compare with $p^{1/\tau}$ renormalised.
   *Expected:* identical, matching the table in section 2.
2. **Nucleus size.** On real logits from the char model after the prompt `"ROMEO:\nBut sof"`,
   count how many tokens top-p 0.9 keeps; then try `"ROMEO:\nBut soft, what "` (ending in a
   space). *Expected* (measured on the 1000-step checkpoint): after `sof` the top token has
   $p = 0.987$ and the nucleus holds **1** token; after the space the top three are about 0.09 each
   and the nucleus holds **17** tokens. Top-p adapts to the model's confidence; a fixed top-k can't.
3. **Feel the cache.** Run `python scripts/generate.py runs/char_kavi/best.npz --bench --tokens 100`
   and again with `--tokens 400`. *Expected:* the uncached rate drops as the window grows toward
   128 and then levels off; the cached rate stays roughly flat. Exact numbers depend on the
   machine and on whether anything else is using the CPU.
4. **See the divergence.** Use the model from `tests/test_inference.py::make("kavi")`
   (`block_size=12`), generate 30 greedy tokens from `[[1,2,3]]` with and without the cache, and
   find the first differing index. *Expected:* identical through index 12; they may diverge
   from index 13 on (we saw 13).
