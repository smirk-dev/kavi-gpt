# 05 · Attention

> **What you'll learn.** Attention is the only place in a transformer where tokens exchange
> information; everything else works on one token at a time. This chapter builds causal
> multi-head self-attention from the intuition up, derives its full backward pass by hand
> (including the softmax Jacobian), proves two surprising facts (the key bias gets *exactly
> zero* gradient, and attention cannot see word order on its own), and then derives rotary
> position embeddings (RoPE), forward and backward, exactly as `kavi/attention.py` implements them.

Prerequisites: [linear layers and backprop](03-linear-and-backprop.md), [embeddings](02-embeddings.md).
Shapes throughout: `B` batch, `T` sequence length, `C` model width, `H` heads, `D = C/H` head width.

---

## 1. Why tokens need to talk

After the embedding layer every token is a vector $x_t \in \mathbb{R}^C$ that knows only *which*
token it is. To predict what comes after `"ROMEO: But soft, what light through yonder"`, the
vector at `"yonder"` needs information from the other words. The MLP ([chapter 06](06-mlp.md))
can't help: it is applied to each position on its own.

The simplest way to mix in information is to average the past:

$$h_t = \frac{1}{t+1}\sum_{j \le t} x_j .$$

That works, but it is crude: every past token counts equally, whether it matters or not.
Attention keeps the "weighted average of the past" shape and lets the **weights depend on the
data**: each token decides how much to take from each earlier token.

## 2. Queries, keys and values

Each token produces three vectors by linear maps of its own state:

- a **query** $q_t$: "what am I looking for?"
- a **key** $k_j$: "what do I contain?" (what other tokens match against)
- a **value** $v_j$: "if you pick me, this is what I hand over."

Token $t$ scores every earlier token $j$ by how well its query matches their key, $q_t \cdot k_j$,
turns the scores into weights with a softmax, and takes the weighted average of the *values*.
Splitting key and value matters: what makes a token *relevant* (key) can differ from what
it should *contribute* (value).

In Kavi, all three come from one `Linear(C, 3C)` called `qkv`, whose output is split into three
`(B,T,C)` pieces. One big matmul is faster than three small ones and the maths is the same.

## 3. Scaled dot-product attention

For one head, stack the queries, keys and values of all $T$ tokens as rows of
$Q, K, V \in \mathbb{R}^{T\times D}$. Then

$$
S = \frac{QK^\top}{\sqrt{D}} + M, \qquad
P = \operatorname{softmax}_{\text{row}}(S), \qquad
O = PV,
$$

| quantity | shape | meaning |
|---|---|---|
| $S_{ij} = q_i\cdot k_j/\sqrt D + M_{ij}$ | $(T,T)$ | how much query $i$ likes key $j$ |
| $P_{ij}$ | $(T,T)$ | each row is a probability distribution over keys |
| $O_i = \sum_j P_{ij} v_j$ | $(T,D)$ | what token $i$ gathered |

$M$ is the causal mask (section 5). Softmax is applied to each row separately:
$P_{ij} = e^{S_{ij}} / \sum_{j'} e^{S_{ij'}}$.

## 4. Why divide by $\sqrt{D}$?

At initialisation, take the entries of $q$ and $k$ to be independent, mean 0, variance 1. Then

$$
q\cdot k = \sum_{d=1}^{D} q_d k_d .
$$

Each term has mean $\mathbb{E}[q_d]\mathbb{E}[k_d] = 0$ and variance
$\mathbb{E}[q_d^2 k_d^2] - 0 = \mathbb{E}[q_d^2]\,\mathbb{E}[k_d^2] = 1\cdot 1 = 1$.
The terms are independent, so variances add:

$$
\operatorname{Var}(q\cdot k) = D, \qquad \operatorname{std}(q\cdot k) = \sqrt{D}.
$$

The raw scores grow with the head width. With $D=64$ they have a standard deviation of 8, and
a softmax over numbers that far apart is almost one-hot. (A quick check: 8 random scores of std
8 gave a top probability of 0.975; divided by $\sqrt{64}$ the top probability was 0.30.)

Why is one-hot bad? Section 7 shows the softmax Jacobian is $\operatorname{diag}(P) - PP^\top$.
If $P$ is a one-hot vector $e_a$ the Jacobian is $\operatorname{diag}(e_a) - e_a e_a^\top = 0$:
**no gradient flows into the scores**, so $Q$ and $K$ can't learn. Dividing by $\sqrt D$
brings the variance back to $D/D = 1$ whatever the head width. Our two configs have
$D = 128/4 = 32$ (char model) and $D = 256/8 = 32$ (BPE model), so without scaling the scores
would have std $\approx 5.7$.

## 5. The causal mask: no peeking

Training is parallel: for a sequence of $T$ tokens we predict token $t+1$ from position $t$
**for every $t$ at once** (targets are the inputs shifted by one, see [chapter 07](07-loss.md)).
That is $T$ training examples for the price of one forward pass, but only if position $t$
cannot see positions after $t$. Otherwise the answer to "what comes next?" is sitting right
there: attention learns to copy token $t+1$, training loss drops toward zero, and generation
(where the future really doesn't exist) produces garbage. That is the "peeking" bug Anton hit
in episode 1, and he added the mask in episode 2.

The fix is to make every future score $-\infty$ before the softmax:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i\end{cases}
\quad\Longrightarrow\quad P_{ij} = \frac{e^{-\infty}}{\cdots} = 0 \text{ for } j>i .
$$

Every row keeps at least its diagonal entry ($j = i$), so no row is all $-\infty$ and the softmax
never computes $0/0$. In code:

```python
causal = xp.tril(xp.ones((T, T), dtype=bool))     # True on and below the diagonal
scores = xp.where(causal, scores, -xp.inf)
```

The mask must be applied *before* the softmax. Zeroing probabilities *after* the softmax
would leave rows that don't sum to 1, and the scores would still depend on future keys
through the denominator.

## 6. Multiple heads

One attention pattern per layer is limiting: a token might want to look at the previous
character *and* the start of the current word. Multi-head attention runs $H$ independent
attentions of width $D = C/H$ side by side and concatenates their outputs. Parameter count and
FLOPs match a single head of width $C$: `qkv` is still $C\times 3C$ and `proj` is $C\times C$.

The shapes, following `CausalSelfAttention.forward`:

```
x                    (B, T, C)
qkv(x)               (B, T, 3C)
split -> q, k, v     (B, T, C) each
reshape              (B, T, H, D)           _split_heads
transpose(0,2,1,3)   (B, H, T, D)           heads become a batch dimension
q @ kᵀ               (B, H, T, T)           scores
probs @ v            (B, H, T, D)
transpose + reshape  (B, T, C)              _merge_heads
proj                 (B, T, C)
```

The reshape must go to `(B,T,H,D)` *and then* transpose. Reshaping `(B,T,C)` directly to
`(B,H,T,D)` runs without error but mixes features from different tokens into one "head".
Once heads sit on axis 1, NumPy's batched `@` handles all $B\cdot H$ attentions in one call.

## 7. The backward pass

The module gets $\partial L/\partial y$ for its output $y$. Undoing `resid_drop` and `proj`
([chapter 03](03-linear-and-backprop.md)) and splitting heads gives
$dO = \partial L/\partial O$ of shape `(B,H,T,D)`. We work on one head; the batched code does
the same thing for all of them. Notation: $dX$ means $\partial L/\partial X$, same shape as $X$.

The forward chain, with attention dropout included:

$$
R = QK^\top \;\to\; S = sR + M \;\to\; P = \operatorname{softmax}(S) \;\to\; \tilde P = \operatorname{drop}(P) \;\to\; O = \tilde P V,
\qquad s = 1/\sqrt D .
$$

### 7.1 Through $O = \tilde P V$

Elementwise $O_{id} = \sum_j \tilde P_{ij} V_{jd}$. Each $\tilde P_{ij}$ shows up in the
outputs $O_{i1},\dots,O_{iD}$:

$$
d\tilde P_{ij} = \sum_d dO_{id}\,V_{jd} \;\Rightarrow\; d\tilde P = dO\,V^\top
\qquad (T\times D)(D\times T) = (T\times T)\;\checkmark
$$

Each $V_{jd}$ shows up in $O_{1d},\dots,O_{Td}$:

$$
dV_{jd} = \sum_i \tilde P_{ij}\,dO_{id} \;\Rightarrow\; dV = \tilde P^\top dO
\qquad (T\times T)(T\times D) = (T\times D)\;\checkmark
$$

Dropout's backward is "multiply by the same mask and $1/(1-p)$", so
$dP = \texttt{attn\_drop.backward}(d\tilde P)$. With dropout off (eval, or `dropout=0`),
$\tilde P = P$ and $dP = d\tilde P$.

### 7.2 The softmax Jacobian

Take one row: $P_i = e^{S_i}/Z$ with $Z = \sum_k e^{S_k}$ (dropping the row index). By the
quotient rule, using $\partial e^{S_i}/\partial S_j = \delta_{ij}e^{S_i}$ and
$\partial Z/\partial S_j = e^{S_j}$:

$$
\frac{\partial P_i}{\partial S_j}
= \frac{\delta_{ij}e^{S_i}\,Z - e^{S_i}e^{S_j}}{Z^2}
= \frac{e^{S_i}}{Z}\left(\delta_{ij} - \frac{e^{S_j}}{Z}\right)
= P_i(\delta_{ij} - P_j).
$$

As a matrix: $J = \operatorname{diag}(P) - PP^\top$ (symmetric). Now the chain rule. $S_j$
affects every $P_i$ in its row:

$$
dS_j = \sum_i dP_i \frac{\partial P_i}{\partial S_j}
= \sum_i dP_i P_i(\delta_{ij} - P_j)
= P_j\,dP_j - P_j \sum_i dP_i P_i .
$$

So, for every row,

$$
\boxed{\,dS = P \odot \Big(dP - \textstyle\sum_j dP_j P_j\Big)\,}
$$

where the sum is one number per row, broadcast across the row. This avoids building the
$T\times T$ Jacobian for each of the $T$ rows: the cost is $O(T)$ per row instead of $O(T^2)$.

### 7.3 Through the scale and the mask

$S = sR + M$ with $M$ constant, so $dR = s\,dS$. The code folds $s$ straight in:

```python
dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True)) * scale    # this is dR
```

**Masked entries need no special handling.** For $j > i$, $P_{ij} = 0$, so
$dS_{ij} = 0\cdot(\ldots) = 0$, as long as the bracket is finite. It is: $dP = dO\,V^\top$ involves
only finite numbers. That matches the true derivative too: `where` swaps the masked scores for
a constant $-\infty$, so they don't depend on $Q$ or $K$ at all.

### 7.4 Through $R = QK^\top$

$R_{ij} = \sum_d Q_{id}K_{jd}$. $Q_{id}$ appears in row $i$ of $R$; $K_{jd}$ in column $j$:

$$
dQ_{id} = \sum_j dR_{ij}K_{jd} \Rightarrow dQ = dR\,K, \qquad
dK_{jd} = \sum_i dR_{ij}Q_{id} \Rightarrow dK = dR^\top Q .
$$

Shapes: $(T\times T)(T\times D)$ for both. ✓

### 7.5 Back to the input

`_qkv_backward` undoes RoPE if it is on (section 11), merges each of $dQ, dK, dV$ back to
`(B,T,C)`, concatenates them into `(B,T,3C)` (the inverse of `split`) and calls
`qkv.backward`, which accumulates $dW$, $db$ and returns $dx$.

| step | forward (code) | backward (code) |
|---|---|---|
| values | `out = probs_d @ v` | `dPd = dout @ vᵀ`, `dv = Pdᵀ @ dout` |
| dropout | `probs_d = attn_drop(probs)` | `dP = attn_drop.backward(dPd)` |
| softmax+scale | `probs = softmax(where(causal, q@kᵀ·scale, -inf))` | `dS = P*(dP - Σ dP·P)*scale` |
| scores | `q @ kᵀ` | `dq = dS @ k`, `dk = dSᵀ @ q` |

## 8. A gradient that is exactly zero: the key bias

With `bias=True` (the GPT-2 and "anton" configs), `qkv` has a bias, and its middle third
$b_k$ is added to every key. **Without RoPE, $\partial L/\partial b_k = 0$ exactly**, for every
input and every loss.

**Proof 1 (invariance).** Write $k_j = k'_j + b$, where $k'_j$ is the bias-free part. Query
$i$'s score for key $j$ is

$$
S_{ij} = s\,q_i\cdot(k'_j + b) = s\,q_i\cdot k'_j + \underbrace{s\,q_i\cdot b}_{c_i} .
$$

$c_i$ depends on the query but **not on $j$**: it shifts the whole row by a constant. Softmax
doesn't care about shifts:

$$
\frac{e^{S_j + c}}{\sum_k e^{S_k + c}} = \frac{e^{c}e^{S_j}}{e^{c}\sum_k e^{S_k}} = \frac{e^{S_j}}{\sum_k e^{S_k}} .
$$

So $P$, and with it the output and the loss, doesn't depend on $b$ at all. A function that
doesn't depend on $b$ has zero derivative with respect to $b$.

**Proof 2 (through our backward formulas).** The bias gradient is the sum of $dK$ over all
positions: $db = \sum_j dK_j = \sum_j\sum_i dR_{ij}\,q_i = \sum_i q_i \big(\sum_j dR_{ij}\big)$.
Every row of $dR$ sums to zero:

$$
\sum_j dS_{ij} = \sum_j P_{ij}dP_{ij} - \Big(\sum_j P_{ij}\Big)\Big(\sum_k dP_{ik}P_{ik}\Big)
= \sum_j P_{ij}dP_{ij} - 1\cdot\sum_k dP_{ik}P_{ik} = 0 .
$$

So $db = 0$. (Row sums of zero are what shift invariance looks like in a gradient.)

Numerically: the finite-difference gradcheck in `test_attention[bias=True, rope=False]` measured
the analytic key-bias gradient at **5.6e-17**, pure float64 round-off. This is why
`kavi/gradcheck.py::_rel_err` switches to an absolute error near zero: the relative error of two
round-off values is meaningless. The query and value biases are *not* zero: $b_q$ changes how
strongly each query prefers each key, and $b_v$ adds $\sum_j P_{ij} b_v = b_v$ to every output
(it duplicates the `proj` bias, but it is not zero).

**Why RoPE breaks this.** With RoPE the bias is added *before* the rotation, and the rotation
depends on the key's position $j$:

$$
S_{ij} = s\,\big(R_i q_i\big)\cdot\big(R_j(k'_j + b)\big)
= s\,(R_i q_i)\cdot(R_j k'_j) + s\,(R_i q_i)\cdot(R_j b) .
$$

The second term now depends on $j$, so it is no longer a constant row shift, and $b_k$ acts as
a learned, position-dependent preference. In a quick check, with RoPE the largest key-bias
gradient entry was 1.3; without RoPE it was $1.7\times10^{-16}$.

## 9. Attention doesn't know word order

Look at the last position of a sequence. Its output is $\sum_j P_j v_j$ with
$P_j \propto e^{s\,q\cdot k_j}$: a sum over the visible tokens, and **a sum doesn't care about
order**. Without position information, one layer gives the last token of "A B C" and
"B A C" *identical* outputs. (Checked: with `pos="none"`, `n_layer=1`, the last-position logits
of `[1,2,3]` and `[2,1,3]` matched to $4\times10^{-16}$.)

Without the mask the statement is exact: attention is **permutation-equivariant**. Permute the
input rows and the output rows come out permuted the same way. The causal mask adds a faint
signal, since each token sees a different *number* of predecessors. With two or more layers
that can leak some order (the same check with `n_layer=2` gave a difference of 0.39), but it is
indirect and weak. "Dog bites man" and "man bites dog" deserve clearly different
representations, so we give the model position information explicitly. Anton's episode-1 model
had none; he added it in episode 2.

## 10. Option 1: learned absolute positions (GPT-2)

With `pos="learned"`, `GPT` owns a second table `wpe` of shape `(block_size, C)` and adds row
$t$ to the token embedding at position $t$: `x = wte(ids) + wpe(arange(T))`. It's simple and
needs no new backward code (it's an embedding lookup, [chapter 02](02-embeddings.md)), but:

- it costs `block_size × C` parameters, and each position must be learned separately;
- it is **absolute**: "3 tokens back" has to be learned separately at every position;
- it cannot go past `block_size`: row 128 doesn't exist.

## 11. Option 2: rotary position embeddings (RoPE)

### 11.1 The 2-D idea

Take $q, k \in \mathbb{R}^2$ and the rotation matrix
$R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}$.
Rotate the query at position $m$ by $m\theta$ and the key at position $n$ by $n\theta$. Since
rotations are orthogonal ($R(\alpha)^\top = R(-\alpha)$) and angles add
($R(\alpha)R(\beta) = R(\alpha+\beta)$):

$$
\langle R(m\theta)q,\; R(n\theta)k\rangle = q^\top R(m\theta)^\top R(n\theta)\,k
= q^\top R(-m\theta)R(n\theta)\,k = q^\top R\big((n-m)\theta\big)\,k .
$$

The score depends on the positions **only through $n - m$**. Absolute position goes in, and
relative position comes out where it's needed, in the score. No parameters, and it's
applied to $q$ and $k$ only (values aren't rotated; their job is content, not matching).

### 11.2 Many frequencies

A head has $D$ features, so RoPE makes $D/2$ two-dimensional pairs and rotates pair $i$ at its
own speed:

$$
\theta_i = \text{base}^{-2i/D}, \qquad i = 0, 1, \dots, D/2 - 1, \qquad \text{base} = 10000 .
$$

`RoPE.__init__` computes exactly this: `inv_freq = base ** (-arange(0, D, 2) / D)`, which gives
exponents $0, 2/D, 4/D, \dots, (D-2)/D$. Pair 0 turns 1 radian per token (it repeats every
$2\pi \approx 6.3$ tokens, good for fine local position). For $D=32$ the slowest pair turns
$1.8\times10^{-4}$ rad/token, a wavelength of about 35,000 tokens, which is effectively
"far vs near". The pairs work like the hands of a clock running at very different speeds.

### 11.3 The half-split pairing (`rotate_half`)

Which features form a pair? Kavi pairs feature $i$ with feature $i + D/2$, writing
$x = [x^{(1)}, x^{(2)}]$ for the two halves. The rotation of pair $i$ at position $t$ by
$\alpha_i = t\theta_i$ is

$$
\begin{aligned}
y^{(1)}_i &= x^{(1)}_i\cos\alpha_i - x^{(2)}_i \sin\alpha_i\\
y^{(2)}_i &= x^{(2)}_i\cos\alpha_i + x^{(1)}_i \sin\alpha_i .
\end{aligned}
$$

Define `rotate_half(x) = [-x2, x1]` and duplicate the angle table so that
`cos[t] = [cos α, cos α]` and `sin[t] = [sin α, sin α]` (the `concatenate([angles, angles])` line).
Then both lines above are one vector equation:

$$
y = x \odot \cos + \operatorname{rotate\_half}(x)\odot\sin \qquad \text{(RoPE.forward)}
$$

Check the first half: $x^{(1)}\cos + (-x^{(2)})\sin$ ✓. Second half: $x^{(2)}\cos + x^{(1)}\sin$ ✓.

The original RoPE paper pairs *adjacent* features $(x_{2i}, x_{2i+1})$ instead. The two
conventions differ by a fixed reordering of features, and $W_q, W_k$ are learned, so the
model absorbs it and both are equally good. Weights trained with one convention just don't
load into the other.

### 11.4 Proof that the score is relative, in $D$ dimensions

Up to that fixed reordering, $R(t)$ is block-diagonal with $2\times2$ blocks $R(t\theta_i)$.
Write $q = (q_0,\dots,q_{D/2-1})$ as a list of 2-vectors (pair $i$ = features $i$ and $i+D/2$).
The dot product splits over pairs, and each pair is the 2-D case:

$$
\langle R(m)q, R(n)k\rangle = \sum_{i} \langle R(m\theta_i)q_i, R(n\theta_i)k_i\rangle
= \sum_i q_i^\top R\big((n-m)\theta_i\big)k_i ,
$$

a function of $n - m$ only. ✓ (Numerically, a random $q,k$ at positions (10,7), (23,20) and
(3,0) all gave the score 2.126049923675…, matching to 15 digits.)

### 11.5 Backward

RoPE is linear in $x$: $y = A x$ with $A = \operatorname{diag}(\cos) + \operatorname{diag}(\sin)\,R_h$,
where $R_h$ is the matrix of `rotate_half`:

$$
R_h = \begin{pmatrix} 0 & -I \\ I & 0\end{pmatrix}, \qquad
R_h^\top = \begin{pmatrix} 0 & I \\ -I & 0\end{pmatrix} = -R_h .
$$

For a linear map the input gradient is the transpose applied to the output gradient
([chapter 03](03-linear-and-backprop.md)):

$$
dx = A^\top dy = \operatorname{diag}(\cos)\,dy + R_h^\top\operatorname{diag}(\sin)\,dy
= dy\odot\cos - \operatorname{rotate\_half}(dy \odot \sin) \qquad \text{(RoPE.backward)}
$$

This is rotation by $-\alpha$, which is what it should be: the inverse of a rotation is its
transpose. A detail: because `sin` is the same in both halves, `rotate_half(dy*sin)` happens to
equal `rotate_half(dy)*sin` here. The transpose form is the one that's correct in general,
so that's what the code writes. (Adjoint check: $\langle dy, Ax\rangle - \langle A^\top dy, x\rangle = -8.9\times10^{-16}$.)

The tables are computed once in float64 up to `max_len = block_size` and cast to the backend.
RoPE runs after `_split_heads`, on `(B,H,T,D)` tensors, so the tables of shape `(T, D)`
broadcast over batch and heads.

### 11.6 `offset`, for the KV cache

`forward(x, offset)` uses rows `offset … offset+T-1` of the tables. In training `offset=0`.
During cached generation ([chapter 10](10-sampling.md)) we feed one new token at absolute
position `start`, so it must be rotated by `start·θ`, not `0·θ`: `forward_cached` passes
`offset=start`. Keys are stored in the cache *already rotated*, which works because a key's
rotation depends only on its own position, never on who is querying it.

### 11.7 Learned positions vs RoPE

| | learned `wpe` | RoPE |
|---|---|---|
| parameters | `block_size × C` | none |
| what the score sees | absolute positions, mixed into content | relative offset $n-m$, by construction |
| where applied | once, at the input | every layer, on $q,k$ only |
| beyond `block_size` | impossible (no row) | defined, though quality drops without tricks |
| constraints | none | even head dim (`assert head_dim % 2 == 0`) |
| side effect | none | key bias is no longer dead (section 8) |

How much this matters on Shakespeare is measured by the `gpt2+rope` ablation, reported in
[the journal](journal.md).

## 12. Code walkthrough (`kavi/attention.py`)

- `softmax`: subtracts the row max before `exp`. This is safe because of the same shift
  invariance as in section 8. A row containing `-inf` stays fine, because the max is finite
  (the diagonal).
- `rotate_half`, `RoPE.forward/backward`: sections 11.3 and 11.5, line for line.
- `CausalSelfAttention.forward`: `qkv` → split → `_split_heads` → optional RoPE → (flash path,
  [chapter 12](12-flash-attention.md)) → scores × `scale` → `where(causal, …, -inf)` →
  `softmax` → `attn_drop` → `@ v` → `_merge_heads` → `proj` → `resid_drop`. It caches
  `q, k, v` (post-RoPE), `probs`, `probs_d`, `scale` for the backward pass.
- `CausalSelfAttention.backward`: section 7, in the same order as the table in 7.5.
- `_qkv_backward`: RoPE backward, merge, concatenate, `qkv.backward`.
- `forward_cached`: the inference path, covered in [chapter 10](10-sampling.md).

## 13. How we know it's right

- `tests/test_gradcheck.py::test_attention`: finite differences in float64 on all four
  combinations of `rope ∈ {False, True}` × `bias ∈ {False, True}`, parameters perturbed
  away from their init values so every term matters; relative error < 1e-6 on parameters and input.
- `test_block` and `test_whole_model`: attention inside a full block and the full GPT.
- `tests/test_torch_parity.py::test_model_matches_torch`: an independent PyTorch implementation
  (its own `rope()` written with the same half-split convention, `masked_fill` for the mask,
  torch's softmax), with autograd gradients matching ours to `rtol=1e-7`.
- `tests/test_flash.py`: the naive formulas here serve as the reference for flash attention.
- `tests/test_inference.py::test_kv_cache_matches_full_forward`: the `offset` logic.

## 14. Pitfalls

- **Forgetting the mask** gives training loss that looks great and samples that are garbage
  (section 5).
- **Masking after softmax** is wrong: the rows no longer sum to 1, and future keys still leak in
  through the denominator.
- **Using −1e9 instead of −inf** works in float32 only while real scores stay far below 1e9.
  `-inf` is exact, and safe here because the diagonal keeps every row finite.
- **Reshape without transpose** when splitting heads mixes tokens inside a head. The gradcheck
  can still pass (it's a valid function) while the model is quietly worse.
- **Forgetting $\sqrt D$** saturates the softmax at init and the gradient into $Q, K$ vanishes.
- **RoPE on $v$**, or rotating with `offset=0` during cached decoding: the second one is caught
  by `test_kv_cache_matches_full_forward` with the `kavi` preset.
- **Expecting the key bias to learn.** Without RoPE it can't (section 8). In float64 its
  gradient is round-off far below Adam's `eps=1e-8`, so it barely moves.

## 15. Try it yourself

1. **Row sums of $dS$.** Run a `CausalSelfAttention` forward/backward and recompute
   `dS` from the cached `probs`. *Expected:* `dS.sum(-1)` is ~1e-17 everywhere, which is Proof 2
   in numbers.
2. **The cost of not scaling.** Draw `q, k` of shape `(T=16, D)` from $\mathcal N(0,1)$ for
   $D \in \{16, 64, 256\}$ and print the variance of `q @ k.T` with and without `/sqrt(D)`.
   *Expected:* about 16, 64, 256 unscaled and about 1 scaled (we measured 16.2, 65.3, 259.1 and
   1.01, 1.02, 1.01 on 20 000 samples).
3. **Order blindness.** Build `GPT(GPTConfig(pos="none", n_layer=1, ...))` (neither `"learned"` nor
   `"rope"`, so no position information) and compare the last-position logits of `[1,2,3]` and
   `[2,1,3]`. *Expected:* equal to ~1e-16. With `pos="rope"` or `"learned"` they differ, and with
   `n_layer=2` they differ even without positions (the mask's faint signal).
4. **Relative position.** With `RoPE(8, 64)`, check that
   `forward(q[None], offset=m)[0] @ forward(k[None], offset=n)[0]` depends only on `m - n`, and
   that `rotate_half` written as a matrix satisfies `M.T == -M`.
