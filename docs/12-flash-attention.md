# 12 · Flash attention: the algorithm without the CUDA

> **What you'll learn.** Naive attention builds a $T\times T$ matrix per head, and on a GPU
> moving that matrix around costs more than computing it. FlashAttention computes *exactly* the
> same result tile by tile without ever storing the matrix. This chapter derives the
> **online softmax** that makes this possible (with a proof that it equals the ordinary softmax),
> walks through `kavi/flash.py::flash_forward` and `flash_backward` line by line, and proves the
> identity $\Delta_i = \operatorname{rowsum}(dO_i \odot O_i)$ that lets the backward pass work without
> the probability matrix.

Prerequisites: [attention, especially the backward pass in §7](05-attention.md),
[GPU basics](11-gpu.md). Anton planned flash attention for a later PyTorch episode. Here it's
written in plain NumPy, where every step can be read and tested.

---

## 1. Why naive attention is memory-bound

Per head, naive attention (`CausalSelfAttention.forward` with `flash=False`) does

$$
S = QK^\top s \;\to\; P = \operatorname{softmax}(S) \;\to\; O = PV, \qquad s = 1/\sqrt D,
$$

and **keeps $P$** (`self.probs`, plus `self.probs_d` after dropout) for the backward pass. That's
$B\cdot H\cdot T^2$ numbers per layer. For the BPE config (`B=32, H=8, T=256`) in float32:

$$
32 \cdot 8 \cdot 256^2 \cdot 4\text{ bytes} = 67\text{ MB per layer for } P \text{ alone},
$$

which is hundreds of MB over 6 layers, and it grows **quadratically** with context length.
Doubling $T$ quadruples it.

On a GPU the problem is speed as well as size. A GPU has two kinds of memory:

- **HBM** (the "GPU RAM", many GB): large, but slow to read and write compared with the compute units.
- **SRAM** (on-chip, roughly a hundred KB per streaming multiprocessor): tiny, but much faster.

A naive implementation writes $S$ to HBM, reads it back for the softmax, writes $P$, and reads
it again for $PV$. Softmax and masking do only a few FLOPs per element, so the GPU spends most
of its time **waiting on memory** rather than computing. The fix is to bring small tiles of
$Q, K, V$ into SRAM, do *all* the work for a tile there (scores, softmax, weighted sum), and
write only the final output. The obstacle is the softmax: its denominator needs the *whole row*
of scores, and a tile only sees part of the row.

## 2. Online softmax

Consider one query row with scores $s_1, \dots, s_N$ and values $v_1, \dots, v_N$. We want

$$
o = \sum_j \frac{e^{s_j}}{\sum_k e^{s_k}}\, v_j ,
$$

but we see the keys in chunks, and we don't want to keep the scores around.

**State.** After processing some set $\mathcal{S}$ of keys, keep three things:

$$
m_{\mathcal S} = \max_{j\in\mathcal S} s_j, \qquad
\ell_{\mathcal S} = \sum_{j\in\mathcal S} e^{s_j - m_{\mathcal S}}, \qquad
a_{\mathcal S} = \sum_{j\in\mathcal S} e^{s_j - m_{\mathcal S}}\, v_j .
$$

Subtracting the running max keeps every exponent $\le 0$, so `exp` never overflows. This is the
same stability trick as `softmax` in `kavi/attention.py`, applied incrementally.

**Update.** A new chunk $\mathcal J$ arrives. Let $m' = \max(m_{\mathcal S}, \max_{j\in\mathcal J} s_j)$.
Then

$$
\ell_{\mathcal S\cup\mathcal J} = e^{m_{\mathcal S}-m'}\,\ell_{\mathcal S} + \sum_{j\in\mathcal J} e^{s_j-m'},
\qquad
a_{\mathcal S\cup\mathcal J} = e^{m_{\mathcal S}-m'}\,a_{\mathcal S} + \sum_{j\in\mathcal J} e^{s_j-m'}\,v_j .
$$

*Proof.* Each old term was stored relative to the old max. Multiplying by
$\alpha = e^{m_{\mathcal S} - m'}$ moves it to the new max:

$$
e^{m_{\mathcal S}-m'}\cdot e^{s_j - m_{\mathcal S}} = e^{s_j - m'} .
$$

So $\alpha\,\ell_{\mathcal S} = \sum_{j\in\mathcal S} e^{s_j - m'}$, and adding the new chunk's terms
gives $\sum_{j\in\mathcal S\cup\mathcal J} e^{s_j-m'}$, which is the definition of
$\ell_{\mathcal S\cup\mathcal J}$ (its max is $m'$). The same algebra works for $a$. ∎

By induction over the chunks, after all $N$ keys we hold $m, \ell, a$ for the full row.

**The result equals ordinary softmax attention:**

$$
\frac{a}{\ell} = \frac{\sum_j e^{s_j - m} v_j}{\sum_k e^{s_k - m}}
= \frac{e^{-m}\sum_j e^{s_j} v_j}{e^{-m}\sum_k e^{s_k}} = \sum_j \frac{e^{s_j}}{\sum_k e^{s_k}} v_j = o . \;\checkmark
$$

**Start state:** $m = -\infty$, $\ell = 0$, $a = 0$ (the empty set). The first update has
$\alpha = e^{-\infty - m'} = 0$, which correctly multiplies the empty zeros.

**The log-sum-exp.** At the end we also keep

$$
\text{lse} = m + \log \ell = \log\!\big(e^{m}\textstyle\sum_k e^{s_k - m}\big) = \log \sum_k e^{s_k},
$$

the log of the softmax denominator. It's one number per row, and the backward pass needs
nothing else from the forward pass besides $O$.

## 3. The tiled forward pass (`flash_forward`)

`q, k, v` have shape `(B, H, T, D)` (already split into heads and rotated by RoPE). Loop over
**query tiles** of `block` rows (default 32); for each, stream over **key tiles**:

```python
for qs in range(0, T, block):                      # query tile [qs, qe)
    qe = min(qs + block, T)
    qi = q[..., qs:qe, :]
    m   = full(-inf);  l = zeros;  acc = zeros     # per query row: m, ℓ, a
    for ks in range(0, qe if causal else T, block):     # key tile [ks, ke)
        ke = min(ks + block, T)
        s = (qi @ k[..., ks:ke, :]ᵀ) * scale            # (B,H,bq,bk) scores of this tile
        s = where(causal_mask(qs, qe, ks, ke), s, -inf)
        m_new = maximum(m, s.max(-1))                   # m'
        p     = exp(s - m_new)                          # e^{s_j - m'}
        alpha = exp(m - m_new)                          # e^{m - m'}
        l     = alpha * l   + p.sum(-1)                 # ℓ update
        acc   = alpha * acc + p @ v[..., ks:ke, :]      # a update
        m = m_new
    out[..., qs:qe, :] = acc / l
    lse[..., qs:qe, :] = m + log(l)
```

Every line is one equation from section 2, applied to a whole tile of rows in parallel.
`_causal_mask(qs, qe, ks, ke)` is the global causal mask restricted to the tile:
`arange(ks, ke)[None, :] <= arange(qs, qe)[:, None]`, i.e. key position ≤ query position.

**Skipping the future.** In causal mode the key loop stops at `qe`. Any key tile starting at
`ks ≥ qe` holds only keys later than every query in the tile, so every score in it would be
masked. Not computing it is exact. Query and key tiles share the same `block`, so the last
tile visited is the diagonal tile `ks = qs`, and with $n$ tiles per side we visit
$n(n+1)/2$ of the $n^2$ tile pairs (15 of 25 for `T=33, block=8`). That's half the work of the
naive version, which computes the full $T\times T$ matrix and then masks half of it.

**No NaNs.** The danger in online softmax is a row whose running max is still $-\infty$ after a
tile: then `s - m_new` is $-\infty - (-\infty) = $ NaN. That can't happen here. The **first** key
tile (`ks = 0`) contains key 0, which every query may attend to ($0 \le i$), so after the first tile
every row's $m$ is finite. After that, $m$ never decreases, and fully-masked entries in later tiles
give `exp(-inf - finite) = 0`. (In fact no row of any visited tile is fully masked: the diagonal
tile lets query $i$ see key `qs`.)

**Ragged edges.** `min(qs + block, T)` handles a $T$ that isn't a multiple of `block`. The last
tiles are just smaller. The tests use $T \in \{1, 7, 16, 33, 20\}$ with blocks of 4, 8 and 64
to exercise this, including a block larger than $T$.

**What's stored:** `out` `(B,H,T,D)` and `lse` `(B,H,T,1)`. The module keeps `q, k, v, out, lse`
(`CausalSelfAttention.forward`, flash branch), so attention memory is $O(T)$ per head instead of $O(T^2)$.
For the BPE example above, `lse` is 32·8·256·4 bytes = 262 KB per layer, versus 67 MB for $P$.

## 4. The backward pass with recomputation (`flash_backward`)

The naive backward ([chapter 05 §7](05-attention.md)) needs $P$. We didn't store it, so we
**recompute it, tile by tile, from `lse`**:

$$
P_{ij} = \exp(S_{ij} - \text{lse}_i) = \frac{e^{S_{ij}}}{e^{\text{lse}_i}} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}} .
$$

That's the exact softmax probability, with no running max or rescaling needed because the
final denominator is already known. Masked entries have $S = -\infty$, so $P = 0$.

The gradient formulas are the naive ones, split into sums over tiles:

$$
\begin{aligned}
dV_j &= \sum_i P_{ij}\, dO_i && \text{(accumulate over query tiles)}\\
dP_{ij} &= dO_i \cdot V_j\\
dS_{ij} &= s\,P_{ij}\,(dP_{ij} - \Delta_i), \qquad \Delta_i = \textstyle\sum_j dP_{ij}P_{ij}\\
dQ_i &= \sum_j dS_{ij}\, K_j, \qquad dK_j = \sum_i dS_{ij}\, Q_i
\end{aligned}
$$

### 4.1 The $\Delta$ trick

$\Delta_i = \sum_j dP_{ij}P_{ij}$ is a sum over the **whole row**, but each tile sees only part of
it. Fortunately it has a closed form that needs no $P$ at all:

$$
\Delta_i = \sum_j P_{ij}\,(dO_i\cdot V_j) = dO_i \cdot \Big(\sum_j P_{ij} V_j\Big) = dO_i\cdot O_i .
$$

(Substitute $dP_{ij} = dO_i\cdot V_j$, pull the $j$-independent $dO_i$ out of the sum, and
recognise the forward output.) So $\Delta = \operatorname{rowsum}(dO \odot O)$ costs $O(T\cdot D)$ and is
computed once, before the loops:

```python
delta = (dout * out).sum(axis=-1, keepdims=True)        # (B,H,T,1)
```

### 4.2 The loop

```python
for each query tile i, for each key tile j ≤ diagonal:
    s  = (qi @ kjᵀ) * scale;  s = where(mask, s, -inf)
    p  = exp(s - lse_i)                         # recomputed probabilities
    dv[j] += pᵀ @ doi                           # dV_j += Σ_i P_ij dO_i
    dp = doi @ vjᵀ                              # dP_ij = dO_i · V_j
    ds = p * (dp - delta_i) * scale             # dS (scale folded in, as in the naive code)
    dq[i] += ds @ kj
    dk[j] += dsᵀ @ qi
```

`dq`, `dk`, `dv` start at zero and **accumulate** (`+=`): each query row receives contributions
from every key tile, and each key from every query tile. The tile-skipping rule is the same as
in the forward pass, and it's exact here too, because skipped tiles have $P = 0$ and so
contribute 0 to every sum. The module's `_qkv_backward` then undoes RoPE and `qkv` as usual.

**The trade.** The backward pass recomputes $QK^\top$ and `exp`, roughly one extra forward's worth of
score FLOPs. On a GPU that's a good deal: FLOPs are cheap and HBM traffic is expensive, so
recomputing beats storing and reloading.

## 5. Memory: $O(T)$ vs $O(T^2)$

| | naive | flash |
|---|---|---|
| kept for backward | `probs` (and `probs_d`): $B H T^2$ | `out` + `lse`: $BHTD + BHT$ |
| largest temporary | full scores $B H T^2$ | one tile $B H \cdot \text{block}^2$ |
| score FLOPs (forward) | $T^2$ per head | about $T^2/2$ (future tiles skipped) |
| attention dropout | yes | **no** (section 6) |

Both still keep `q, k, v` and the output ($O(TD)$, needed anyway). The $T^2$ term is what goes away.

**Honest note on speed in NumPy.** Kavi's flash attention is for *understanding*, not speed. NumPy
has no SRAM to manage, and the tile loops run in Python. A quick timing of the attention module's
forward+backward (`B=32, C=128, H=4`, with a training run sharing the CPU, so the numbers are rough)
gave 0.077 s flash vs 0.100 s naive at $T=128$, but 1.02 s vs 0.58 s at $T=256$. Skipping masked tiles
helps a little at short lengths, but the loop overhead and the recomputation catch up. The
docstring's claim that flash is "slower than the naive version" holds for longer contexts.
The real payoff, a fused GPU kernel keeping tiles in SRAM, is what makes long contexts
(tens of thousands of tokens) practical. The algorithm is the same as here.

## 6. Why there is no attention dropout in flash mode

Naive attention applies dropout to $P$ (`attn_drop`). Flash mode never materialises $P$, so
there's nothing to apply a mask to, and storing a $T\times T$ dropout mask would defeat the point.
(Production kernels regenerate the dropout mask inside each tile from a random seed. Kavi
doesn't.) So with `attn="flash"`, **attention dropout is silently skipped**. Residual dropout
(`resid_drop`, after `proj`) and the other dropouts still apply. The comment in
`CausalSelfAttention.__init__` says so. Consequence: with `dropout > 0`, naive and flash
*training* are slightly different models. With `dropout = 0`, or in eval mode, they compute the same function.

## 7. Using it

```json
"model": {"attn": "flash", ...}
```

`GPTConfig.attn = "flash"` sets `flash=True` on every block's `CausalSelfAttention`. Training uses
the flash path; generation through the KV cache (`forward_cached`, [chapter 10](10-sampling.md))
always uses the plain formula, because one query against the cache is already $O(T)$.
`scripts/attention_viz.py` switches flash off before its forward pass, because it needs the explicit
probability matrix to draw ([chapter 13](13-interpretability.md)).

## 8. How we know it's right

- **`tests/test_flash.py::test_flash_matches_naive`**, parametrised over
  `(T, block) ∈ {(1,4), (7,4), (16,4), (33,8), (20,64)}`: random `q, k, v, dout` of shape
  `(2, 3, T, 8)` in float64. The forward output equals naive attention to `rtol=atol=1e-12`, and all
  three gradients equal the naive backward (`naive_grads`) to `rtol=1e-10`. The cases cover a single
  token, ragged last tiles, an exact multiple of the block, and a block larger than the sequence.
- **`test_flash_module_gradcheck`** (with and without RoPE): the whole `CausalSelfAttention` with
  `flash=True`, `T=37` (ragged against `block=32`), checked by finite differences, max relative
  error < 1e-6. This checks the flash backward against the flash forward directly, independent of
  the naive code.

Together: flash forward = naive forward (to 1e-12), flash backward = naive backward (to 1e-10),
and the naive backward was itself proven by gradcheck and torch parity in chapter 05.

## 9. Pitfalls

- **Forgetting to rescale** `l` and `acc` by `alpha` when the max grows: the result is a
  wrong-but-plausible weighted average. The equivalence test catches it at once.
- **Starting the key loop at the diagonal** instead of 0: the first tile would no longer
  contain key 0, so a row could keep $m = -\infty$ and produce NaNs. It's also just wrong, since it skips the past.
- **Storing `m` and `l` separately** for the backward pass works, but `lse = m + log l` is one
  number and gives exact probabilities in one `exp`.
- **Computing $\Delta$ per tile** from the partial $P$ is wrong: $\Delta_i$ is a full-row sum.
  Use $dO_i \cdot O_i$.
- **Expecting a NumPy speedup** (section 5).

## 10. Try it yourself

1. **Online softmax on paper.** Scores $(1, 3, 2)$ in chunks $\{1\}$, $\{3, 2\}$. After chunk 1:
   $m=1, \ell=1$. After chunk 2: $m'=3$, $\alpha = e^{-2}$, $\ell = e^{-2} + 1 + e^{-1}$.
   *Expected:* $\ell\, e^{m} = e^1 + e^3 + e^2$, the plain softmax denominator.
2. **Block size changes nothing.** Run `flash_forward(q, k, v, block=b)` for `b ∈ {1, 3, 32, 1000}`
   on the same random input. *Expected:* outputs agree to ~1e-15 (float64). `block=1` is the pure
   one-key-at-a-time online softmax; `block ≥ T` is a single tile, which is naive attention.
3. **Check Δ.** Using `naive` and `naive_grads` from `tests/test_flash.py`, compute
   `(dp * p).sum(-1)` and `(dout * out).sum(-1)`. *Expected:* equal to ~1e-15.
4. **Count the memory.** For `B=32, H=8, D=32`, tabulate the bytes of `probs` vs `lse` for
   $T \in \{256, 1024, 8192\}$ in float32. *Expected:* 67 MB, 1.07 GB, 68.7 GB per layer for `probs`;
   0.26 MB, 1.05 MB, 8.4 MB for `lse`. Quadratic vs linear.
