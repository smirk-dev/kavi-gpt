# 02 — Embeddings: from token ids to vectors

**What you'll learn.** How an integer token id becomes a learnable vector, and why an embedding
table is just a `Linear` layer applied to one-hot vectors. That view hands us the backward pass
for free, and it explains a nasty NumPy trap that silently drops gradient. You'll also see why
attention can't tell word order without position information, how learned position
embeddings fix that, and how *weight tying* reuses the embedding table as the output layer.

Prerequisite: the Linear backward pass and the fan-out rule from
[chapter 03](03-linear-and-backprop.md). Tokens themselves come from
[chapter 01](01-tokenization.md).

---

## 1. Intuition

The tokenizer gives us integers: `ids` of shape $(B,T)$ with values in $\{0,\dots,V-1\}$. An
integer is a bad input for a neural network. Token 57 isn't "bigger" than token 12 in any
useful sense. So each token gets its own row in a table $W\in\mathbb{R}^{V\times C}$, and the
network reads row `ids[b,t]`. The table starts random. Training moves rows around until tokens
that behave alike end up with similar vectors.

## 2. Embedding = Linear on one-hot vectors

Write token $v$ as a one-hot row vector $e_v\in\{0,1\}^V$: all zeros except a 1 at position $v$.
Then
$$ e_v W = \sum_{u} (e_v)_u W_{u,:} = W_{v,:}. $$
Multiplying by a one-hot vector *selects a row*. For a whole batch, flatten the $N=B\cdot T$
ids into a one-hot matrix $E\in\{0,1\}^{N\times V}$ with $E_{nu} = [\,\text{id}_n = u\,]$:
$$ Y = E\,W \in \mathbb{R}^{N\times C}, \qquad Y_{n,:} = W_{\text{id}_n,:}. $$
That's a bias-free `Linear` layer whose input happens to be one-hot. Computing it by building
$E$ would waste $N\cdot V$ memory and $N\cdot V\cdot C$ multiplications by zero, so the forward
pass is a lookup instead (`kavi/layers.py::Embedding.forward`):

```python
self.ids = ids
return self.weight.data[ids]          # (B,T) ints -> (B,T,C)
```

### Backward: the Linear formula, specialised

From chapter 03, a Linear layer's weight gradient is $dW = X^\top dY$. With $X = E$:
$$
dW_{u,:} \;=\; \sum_n E_{nu}\, dY_{n,:} \;=\; \sum_{n\,:\,\text{id}_n = u} dY_{n,:}.
$$
Row $u$ of the table receives **the sum of the upstream gradients at every position where
token $u$ appeared**. Rows of tokens absent from the batch get zero. Shapes: $dW$ is
$(V,C)$ like $W$ ✓.

That's a **scatter-add**: walk over the $N$ positions and add $dY_n$ into row $\text{id}_n$.
And since there's no meaningful derivative with respect to an integer, `backward` returns
`None`. The embedding is the bottom of the graph.

```python
def backward(self, dout):
    B.scatter_add(self.weight.grad, self.ids.reshape(-1), dout.reshape(-1, self.dim))
    return None
```

## 3. The trap: `W[ids] += dY` loses gradient

The obvious one-liner is `self.weight.grad[ids] += dout`. It's wrong whenever a token repeats in
the batch, and in real text the space character repeats in every sequence.

Tiny example: $V=4$, $C=2$, ids $= [1, 3, 1]$, upstream gradients $[[1,1],[2,2],[10,10]]$.
Token 1 appears at positions 0 and 2, so row 1 should receive $[1,1]+[10,10] = [11,11]$.

```python
>>> W = np.zeros((4, 2)); ids = np.array([1, 3, 1])
>>> d = np.array([[1., 1], [2, 2], [10, 10]])
>>> W[ids] += d;           W[1]
array([10., 10.])          # wrong: the [1, 1] from position 0 vanished
>>> W2 = np.zeros((4, 2)); np.add.at(W2, ids, d);   W2[1]
array([11., 11.])          # right
```

Why: `W[ids] += d` means `W[ids] = W[ids] + d`. NumPy first gathers a temporary copy
`W[ids]` (three rows, two of them copies of row 1), adds `d` to the copy, then scatters the
three results back. Row 1 gets written twice, and the last write wins. `np.add.at` is
*unbuffered*: it performs each addition against the live array, so repeats accumulate.

`kavi/backend.py::scatter_add` wraps this (`np.add.at` on NumPy; `add.at` or
`cupyx.scatter_add` on CuPy, [chapter 11](11-gpu.md)). Nothing crashes with the buggy version,
and the model even trains. Frequent tokens just get systematically too little gradient. That
kind of bug only a gradient check catches.

## 4. Position embeddings

### Attention alone doesn't know word order

Self-attention mixes tokens with weights that depend on *content* ($q_i\cdot k_j$), not on
*where* they are ([chapter 05](05-attention.md)). Let $P$ be a permutation matrix that reorders
the $T$ tokens, applied to the input $X\in\mathbb{R}^{T\times C}$. Without a mask:

* $Q,K,V$ are row-wise projections, so they get permuted the same way:
  $(PX)W_q = P(XW_q)$.
* Scores: $(PQ)(PK)^\top = P\,(QK^\top)\,P^\top$, which is the same matrix with rows *and* columns
  reordered.
* Row-wise softmax commutes with that reordering: $\mathrm{softmax}(PSP^\top) =
  P\,\mathrm{softmax}(S)\,P^\top$.
* Output: $P A P^\top \cdot P V = P (A V)$, because $P^\top P = I$.

So $\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$: **permuting the inputs just permutes the
outputs** (permutation *equivariance*). The MLPs and norms act per token, so they're
equivariant too. Measured on Kavi's own attention module with the mask removed:
$\max|\mathrm{Attn}(PX) - P\,\mathrm{Attn}(X)| = 4\times10^{-19}$.

Consequence: in "A beats B", the vector computed for *A* is exactly the vector computed for *A*
in "B beats A". The model can't tell the winner from the loser.

**With a causal mask** the story has a twist, and it's worth seeing. Take a gpt2-preset `GPT`,
zero its `wpe`, and compare the final-position logits for `[A, beats, B, .]` and
`[B, beats, A, .]`. The final "." sees the same *set* of tokens in both:

| model | max \|logit difference\| at "." |
|---|---|
| 1 layer, no positions | $2\times10^{-17}$ (identical) |
| 2 layers, no positions | $2.3\times10^{-3}$ |
| 2 layers, learned positions | $1.3\times10^{-3}$ |

One causal layer is blind to order, as predicted. With two layers it isn't, because in layer 1
the token at position 0 could only see itself, while the token at position 2 saw three tokens.
The mask makes earlier representations depend on *how many* tokens preceded them, which leaks a
crude sense of position into layer 2. Models can learn to exploit that, but it's weak and
indirect. Explicit positions are far more reliable.

### Learned absolute positions (GPT-2, `pos="learned"`)

The fix GPT-2 and Anton use: a second table `wpe` $\in\mathbb{R}^{T_{max}\times C}$
(`block_size` rows), one learned vector per *position*, added to the token vector
(`GPT.forward`):
$$ x_{b,t} = W^{te}_{\text{id}_{b,t}} + W^{pe}_{t}. $$
In code: `x = x + self.wpe(B.xp.arange(T)[None, :])`. The position ids have shape $(1,T)$, so
`wpe` returns $(1,T,C)$, which broadcasts over the batch.

Backward (`GPT.backward`) uses two rules from chapter 03:

* **Addition** passes $dx$ unchanged to both terms (local Jacobian $I$).
* **Broadcast ⇒ sum**: the $(1,T,C)$ tensor was reused for all $B$ sequences, so its gradient is
  `dx.sum(axis=0, keepdims=True)`. Then `wpe.backward` scatter-adds it, and since the
  position ids $0..T-1$ never repeat, that's just a row-wise add.

Limitations: the model can't handle positions beyond `block_size` (the `assert` in
`GPT.forward`). And "position 37" is learned separately from "position 38", so nothing tells the
model that relative distance is what usually matters. The Kavi preset uses **RoPE** instead
(`pos="rope"`, [chapter 05](05-attention.md)). RoPE rotates queries and keys so that their dot
product depends only on the offset $m-n$. It has zero parameters, and no `wpe` exists at all.

## 5. Weight tying: the output head is the embedding, transposed

At the top, the model turns the final hidden state $h\in\mathbb{R}^{N\times C}$ into $V$ logits.
Instead of a separate $(C,V)$ matrix, Kavi reuses the embedding (`GPT.forward`):
$$ \text{logits} = h\,(W^{te})^\top \in \mathbb{R}^{N\times V}, \qquad \text{logit}_{n,v} = h_n\cdot W^{te}_v. $$
So a token's score is the dot product between the hidden state and that token's own embedding.
"Which token comes next?" is answered by asking which token vector the hidden state points
toward. The same geometry is used in both directions. Two payoffs:

* **Parameters.** $V\cdot C$ saved. For the BPE config ($V=4096$, $C=256$) that's 1.05M of the
  5.76M parameters (18%). For the char model ($V=100$) it's only 12.8K.
* **Data efficiency.** Every row of $W^{te}$ now gets gradient at every step from the output
  side, even for tokens absent from the batch, as shown next. On a small corpus that helps rare
  tokens.

### The gradient into `wte` from both uses

`wte` is a shared parameter with two uses, so its gradient is the sum of two contributions,
which is exactly why `Param.grad` accumulates with `+=` (chapter 03).

**Use 1, the head.** $\text{logits} = h\,W^\top$ is a Linear layer with weight $M = W^\top$
(shape $C\times V$). Chapter 03 gives $dM = h^\top d\text{logits}$ $(C\times V)$, and transposing
gives the gradient for $W$:
$$ dW^{(\text{head})} = (dM)^\top = d\text{logits}^\top\, h \qquad (V\times N)(N\times C) = V\times C\ ✓ $$
Its row $v$ is $\sum_n d\text{logits}_{n,v}\,h_n$. Because softmax gives *every* token some
probability, $d\text{logits}_{n,v} = p_{n,v} - [y_n=v]$ is non-zero for every $v$
([chapter 07](07-loss.md)), so every row gets updated. The input to the head gets
$dh = d\text{logits}\,W$ $(N\times V)(V\times C)$ ✓.

**Use 2, the input lookup.** The scatter-add from §2, with $dx$ arriving all the way from the
bottom of the network.

In `GPT.backward`:

```python
W = self.wte.weight
W.grad += dlogits.reshape(-1, V).T @ self.h.reshape(-1, C)   # use 1: head
dx = self.norm_f.backward(dlogits @ W.data)                  # dh, sent down the stack
...
self.wte.backward(dx)                                        # use 2: scatter-add
```

Delete either contribution and `test_whole_model` fails on `wte.weight` with relative error
1.0, while every other parameter still passes.

**A side effect at initialisation.** At step 0 the residual stream is mostly the token's own
embedding (the blocks' output projections start tiny, [chapter 06](06-mlp.md)). So $h_n$ points
roughly along $W^{te}_{\text{id}_n}$, and the tied head scores the *current* token highest. We
measured, on the untrained char model, an average logit of 2.04 for the input token against
0.02 for the rest, which gives it probability 7% instead of the uniform 1%. Training quickly
unlearns this "repeat yourself" prior. It's also why the initial loss sits a little above
$\ln V$ ([chapter 07](07-loss.md)).

## 6. Initialisation: std 0.02

Both tables are drawn from $\mathcal N(0, 0.02^2)$ (`Embedding.__init__`, `std=0.02`, the GPT-2
value). Why small:

* With tying, logits are $h\cdot W_v$. Here $h$ has come out of the final norm, with roughly unit
  RMS per element, so a logit has standard deviation $\approx 0.02\sqrt C$: 0.23 for $C=128$,
  0.32 for $C=256$. The measured logit std at init is 0.30 (char) and 0.32 (BPE). Small logits
  mean a near-uniform first prediction and a loss near $\ln V$, a sane starting point. A large
  init would make the untrained model confidently wrong, giving huge initial losses and
  gradients.
* Pre-norm blocks normalise their input anyway ([chapter 04](04-normalization.md)), so a
  small-magnitude residual stream doesn't starve the blocks of signal.

## 7. How we know it's right

* `tests/test_gradcheck.py::test_embedding` uses ids `[[1,3,3,7],[0,3,9,1]]`. Token 3 appears
  three times and token 1 twice, **on purpose**, so the scatter-add path is exercised.
  `check_input=False`, since there's no input gradient.
* `test_whole_model[gpt2]` checks `wte` (both uses) and `wpe` (broadcast-sum) end to end, and
  `test_whole_model[kavi]` checks the RoPE variant without `wpe`.
* `tests/test_torch_parity.py` indexes `P["wte.weight"][ids_t]` and multiplies by
  `P["wte.weight"].T` in torch. Autograd accumulates both uses on its own, and our total has to
  match.

## 8. Pitfalls

* Fancy-index `+=` for scatter (§3). It never crashes, and only gradcheck with repeated ids
  catches it.
* Forgetting the batch sum for `wpe`. `wpe.forward` cached position ids of shape $(1,T)$, so
  its backward expects a $(1,T,C)$ gradient. Passing the raw $(B,T,C)$ `dx` gives `np.add.at`
  $B\cdot T$ gradient rows for only $T$ indices, and it raises a shape error. Every broadcast
  in a forward pass needs a matching explicit sum in the backward pass.
* Treating `named_params()` as "all modules' params". The tied matrix is listed once (dedup by
  `id`). Counting parameters by walking modules would count it twice.

## 9. Try it yourself

1. **Reproduce the scatter bug.** In `Embedding.backward`, replace the `B.scatter_add(...)` line
   with `self.weight.grad[self.ids.reshape(-1)] += dout.reshape(-1, self.dim)` and run
   `python -m pytest tests/test_gradcheck.py -q -k embedding`. *Expected:* fails on `weight`
   with relative error 1.0 (rows 1 and 3 are under-counted). Restore it.
2. **Order blindness.** Build `GPT(GPTConfig.preset("gpt2", vocab_size=10, n_embd=16,
   n_head=2, block_size=8, n_layer=1), seed=0)` and set `m.wpe.weight.data[:] = 0`. Compare
   `m.forward(ids)[0][0, -1]` for `[[3,5,7,1]]` and `[[7,5,3,1]]`. *Expected:* identical to
   ~1e-17. Repeat with `n_layer=2`. *Expected:* a ~1e-3 difference, the causal mask leaking
   order.
3. **Count what tying saves.** Compute `V * C` and `m.num_params()` for the BPE config.
   *Expected:* 1,048,576 of 5,758,208. Then check `[n for n, _ in m.named_params()]`: `wte.weight`
   appears once, and there is no separate head matrix.
4. **Untied gradient sanity.** After one `forward`/`backward` on the char model, compare
   `np.count_nonzero(np.abs(m.wte.weight.grad).sum(1))` with the number of distinct ids in the
   batch. *Expected:* all 100 rows are non-zero (the head touches every row), even if the batch
   contains fewer distinct tokens.
