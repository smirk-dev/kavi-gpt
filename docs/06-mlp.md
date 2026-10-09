# 06 — The MLP: per-token computation

**What you'll learn.** What the feed-forward half of a transformer block does, and how its
three variants work: ReLU (Anton's), GPT-2's tanh-GELU, and Llama's gated SwiGLU. You'll derive
each backward pass down to the exact line in `kavi/layers.py`. You'll see why SwiGLU's input
gradient is a *sum* of two terms, how Kavi sizes SwiGLU so ablations compare equal-size models,
and why the layer that writes back into the residual stream starts with a deliberately tiny
init.

Prerequisites: Linear's backward and the fan-out rule ([chapter 03](03-linear-and-backprop.md)).

---

## 1. Intuition: communication vs computation

A block alternates two jobs (`kavi/model.py::Block`):

* **Attention** is *communication*. Each token gathers information from earlier tokens
  ([chapter 05](05-attention.md)).
* **The MLP** is *computation*. Each token processes what it now holds, **on its own**.

"On its own" is literal. The MLP applies the same weights at every position and never mixes
positions. Since `Linear` flattens $(B,T,C)\to(B\cdot T, C)$, the MLP is just a function applied
to $B\cdot T$ independent rows. Its Jacobian across positions is block-diagonal, so the output
at position $t$ depends only on $x_t$. The weights are shared across positions, which is why
their gradients sum over positions (the $X^\top dY$ inside every Linear).

Shape of the GPT-2 MLP (`MLP`): widen $C\to4C$, apply a non-linearity, project $4C\to C$:
$$ \text{MLP}(x) = \phi(xW_{fc} + b_{fc})\,W_{proj} + b_{proj}. $$
Without $\phi$ the two matrices would collapse into one $C\times C$ linear map. The
non-linearity in a wider space is what lets the network carve out features ("is this the end of
a word?", "are we inside a speech?") and write them back into the stream.

Backward is just the chain in reverse, exactly as `MLP.backward` reads:
`fc.backward(act.backward(proj.backward(drop.backward(dy))))`. The only new piece is the
activation's derivative, applied element-wise: $d\,\text{pre} = d\,\text{post}\odot\phi'(\text{pre})$.

## 2. ReLU — Anton's choice

$$ \mathrm{ReLU}(x) = \max(0,x), \qquad \mathrm{ReLU}'(x) = \begin{cases}1 & x>0\\ 0 & x\le 0.\end{cases} $$
At exactly $0$ the derivative is undefined. The code picks 0. `ReLU.forward` caches
`self.mask = x > 0`, and backward is `dy * self.mask`.

It's cheap and works, but a unit whose pre-activation is negative for every input gets zero
gradient forever (a "dead" neuron). And the kink makes finite differences undefined at 0, which
is why `test_activations` moves inputs with $|x|<10^{-3}$ away first. ReLU is selectable with
`mlp="relu"` (the `anton-*` runs in `configs/sweep_ablation.json`).

**What we measured.** Don't write ReLU off. In the follow-up sweep ([journal](journal.md)), the
GPT-2 model with GELU swapped for ReLU (`gpt2+relu`, two seeds) reached 1.6987 val bpb. GELU got
1.7067 and SwiGLU 1.6986. So ReLU *tied* SwiGLU at 4000 steps. SwiGLU was faster early: at step
2000 it led by 0.010. That is one model size and one budget, so it is not a general verdict. But
it is a reminder that "smooth beats kinked" is a hypothesis to test, not a law.

## 3. GELU (GPT-2's tanh approximation)

GELU is $x\,\Phi(x)$ with $\Phi$ the standard normal CDF: "keep $x$ with probability that grows
with $x$". It's smooth, it's slightly negative for negative inputs (minimum ≈ −0.170 at
$x\approx-0.75$), and it still lets some gradient through there. GPT-2 uses a tanh
approximation (max deviation from the exact erf form: $4.7\times10^{-4}$), and so does Kavi, to
match:
$$
\mathrm{GELU}(x) = \tfrac12\,x\,\big(1+\tanh u\big),\qquad u = c\,(x + k x^3),\quad c=\sqrt{2/\pi},\ k = 0.044715 .
$$

### Derivative

Let $t = \tanh u$. Product rule on $\tfrac12 x\cdot(1+t)$:
$$ \frac{dy}{dx} = \tfrac12(1+t) + \tfrac12 x\,\frac{dt}{dx}. $$
Chain rule through tanh, using $\tanh' = 1-\tanh^2$:
$$ \frac{dt}{dx} = (1-t^2)\,\frac{du}{dx}, \qquad \frac{du}{dx} = c\,(1 + 3k x^2). $$
Together:
$$
\boxed{\;\mathrm{GELU}'(x) = \tfrac12(1+t) + \tfrac12\,x\,(1-t^2)\,c\,(1+3kx^2)\;}
$$

### Code walkthrough (`GELU`)

| Math | Code |
|---|---|
| $t=\tanh(c(x+kx^3))$, cached | `self.t = B.xp.tanh(_GELU_C * (x + 0.044715 * x * x * x))` |
| $y = \frac12x(1+t)$ | `0.5 * x * (1.0 + self.t)` |
| $du/dx = c(1+3kx^2)$ | `du = _GELU_C * (1.0 + 3 * 0.044715 * x * x)` |
| boxed formula $\times\,dy$ | `dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)` |

Caching $t$ means backward needs no second `tanh`. The forward costs one transcendental per
element, and the backward costs none.

### Why `x * x * x` instead of `x ** 3`

NumPy special-cases only a few exponents (like squares and square roots) with fast paths. A
general `x ** 3` goes through the generic `pow` routine, a costly transcendental per element.
Two multiplies are much cheaper. On an $8192\times512$ float32 array we measured
`x**3` at 61 ms and `x*x*x` at 17 ms, about 3.5× faster. That matters because GELU runs on the
widest activation in the model ($4C$).

## 4. SwiGLU — the gated MLP (Kavi preset)

### Intuition

SwiGLU (`SwiGLU`) uses **three** matrices and two parallel branches:
$$
a = xW_1,\qquad b = xW_3,\qquad h = \mathrm{SiLU}(a)\odot b,\qquad \text{out} = h\,W_2,
$$
with $\mathrm{SiLU}(a) = a\,\sigma(a)$. Branch $b$ carries *content*. Branch $a$, through SiLU,
is a *gate* that decides per hidden feature, per token, how much of that content passes. The
ReLU/GELU MLP gates each unit by its *own* value. SwiGLU learns a separate, input-dependent
gate, a multiplicative interaction that a plain MLP has to approximate. Empirically it trains
to lower loss at equal parameter count, which is why Llama uses it.

### A sigmoid that never overflows

The textbook $\sigma(a) = 1/(1+e^{-a})$ computes $e^{-a}$, which overflows to `inf` for
$a < -88.7$ in float32 (NumPy warns "overflow encountered in exp"; the final answer happens to
be 0, but any expression built from the `inf` intermediates, such as a naive derivative
$e^{-a}/(1+e^{-a})^2 = \infty/\infty$, becomes NaN). `kavi/layers.py::sigmoid` uses
$$ \sigma(a) = \tfrac12\big(1+\tanh(a/2)\big). $$
**Proof.** Write $\tanh v = \dfrac{1-e^{-2v}}{1+e^{-2v}}$ and set $v = a/2$:
$$
\tfrac12\Big(1 + \frac{1-e^{-a}}{1+e^{-a}}\Big) = \tfrac12\cdot\frac{(1+e^{-a}) + (1-e^{-a})}{1+e^{-a}} = \frac{1}{1+e^{-a}} . \qquad\blacksquare
$$
`tanh` saturates gracefully at $\pm1$ for any input, so no intermediate is ever infinite.

### The sigmoid and SiLU derivatives

$\sigma = (1+e^{-a})^{-1}$, so
$$
\sigma'(a) = \frac{e^{-a}}{(1+e^{-a})^2} = \underbrace{\frac{1}{1+e^{-a}}}_{\sigma}\cdot\underbrace{\frac{e^{-a}}{1+e^{-a}}}_{1-\sigma} = \sigma(1-\sigma).
$$
Product rule on $\mathrm{SiLU}(a) = a\,\sigma(a)$:
$$
\mathrm{SiLU}'(a) = \sigma + a\,\sigma(1-\sigma) = \sigma\,\big(1 + a(1-\sigma)\big).
$$
(SiLU is non-monotonic: minimum ≈ −0.278 at $a\approx-1.28$. Its slope peaks at ≈ 1.10 near
$a\approx2.4$, so it can amplify gradients slightly, unlike ReLU's slope of at most 1.)

### SwiGLU backward

Let $s = \sigma(a)$, so $h = a\,s\,b$ element-wise. The upstream gradient first passes through
$W_2$ (a Linear): $dh = d\text{out}\,W_2^\top$, and $dW_2 \mathrel{+}= h^\top d\text{out}$.

$h$ is a product of two element-wise factors, $\mathrm{SiLU}(a)$ and $b$:
$$
db = dh\odot \frac{\partial h}{\partial b} = dh\odot (a\,s),
\qquad
da = dh\odot \frac{\partial h}{\partial a} = dh\odot b\odot s\,\big(1+a(1-s)\big).
$$
Both $a$ and $b$ came from $x$ through their own Linear layers, so each Linear's backward gives
$dW_1 \mathrel{+}= x^\top da$ and $dW_3 \mathrel{+}= x^\top db$, and *each returns a gradient for
$x$*. $x$ fans out to two branches, so by the chain rule the contributions **add**:
$$
\boxed{\;dx = da\,W_1^\top + db\,W_3^\top\;}
$$
Shapes: $da, db$ are $(N,H)$, $W_1^\top, W_3^\top$ are $(H,C)$, and $dx$ is $(N,C)$ ✓.

| Math | `SwiGLU` code |
|---|---|
| $a, b, s$ cached | `a = self.w1(x); b = self.w3(x); s = sigmoid(a)` |
| $\text{out} = (a s b)W_2$ | `self.w2(a * s * b)` |
| $dh = d\text{out}\,W_2^\top$ | `dh = self.w2.backward(self.drop.backward(dy))` |
| $db = dh\odot as$ | `db = dh * (a * s)` |
| $da = dh\odot b\odot s(1+a(1-s))$ | `da = dh * b * s * (1.0 + a * (1.0 - s))` |
| $dx = da W_1^\top + db W_3^\top$ | `return self.w1.backward(da) + self.w3.backward(db)` |

Note `self.w1.backward(da)` does two jobs at once: it accumulates $dW_1$ *and* returns
$da\,W_1^\top$.

## 5. Parameter parity: $H = 8C/3$

To compare architectures fairly (Kavi's whole ablation table, [chapter 09](09-training.md)),
the MLP variants must have the same number of parameters. Ignoring biases:

* GELU MLP: two matrices of $C\times4C$, so $8C^2$.
* SwiGLU: three matrices of $C\times H$, so $3CH$.

Set $3CH = 8C^2$, which gives $H = 8C/3$. `swiglu_hidden` rounds that to a multiple of 8 (tidy
matmul shapes) with a floor of 8:

```python
return max(8, int(round(8 * dim / 3 / 8)) * 8)
```

For $C=256$: $8\cdot256/3 = 682.7$, and $682.7/8 = 85.3 \to 85$, so $H = 680$. That's 522,240
MLP parameters vs GELU's 524,288 per block, within 0.4%. For the BPE config ($C=256$, $L=6$,
$V=4096$) the whole models measure **5.76M (kavi) vs 5.84M (gpt2)**. Of that 77,824 gap, only
$6\times2{,}048 = 12{,}288$ comes from the rounding. The other 65,536 is gpt2's learned position
table ($256\times256$), which kavi replaces with parameter-free RoPE.

## 6. The residual-branch init: $0.02/\sqrt{2\cdot n_{layer}}$

Every Linear starts at $\mathcal N(0, 0.02^2)$, *except* the projections that write into the
residual stream: `attn.proj`, `mlp.proj` (GELU/ReLU) and `mlp.w2` (SwiGLU). `Block.__init__`
gives them
```python
proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)
```
**Why.** The stream after $L$ blocks is the embedding plus $2L$ branch outputs (two per block:
attention and MLP):
$$ x_L = x_0 + \sum_{i=1}^{2L} f_i . $$
At init the $f_i$ are roughly independent and zero-mean, so variances add:
$\mathrm{Var}(x_L) \approx \mathrm{Var}(x_0) + 2L\cdot v$, where $v \propto \text{proj\_std}^2$.
Scaling each branch's output std by $1/\sqrt{2L}$ scales $v$ by $1/(2L)$. The total added
variance then no longer grows with depth, and a 12-layer model starts out as calm as a 2-layer
one. For our configs: $n_{layer}=4$ gives 0.00707, and $n_{layer}=6$ gives 0.00577. Measured on the
untrained char model, the stream RMS goes 0.020 → 0.027 across all four blocks
([chapter 04](04-normalization.md) §1). Each block starts close to the identity, which pre-norm's
gradient highway likes.

## 7. How we know it's right

* `tests/test_gradcheck.py::test_activations[GELU/ReLU]` checks the element-wise derivatives.
* `test_mlp[gelu/relu]` checks the full GELU/ReLU MLP with biases. ReLU uses `tol=1e-5`, since
  hidden pre-activations can land near the kink.
* `test_swiglu` checks $W_1, W_2, W_3$ and $dx$ after `randomise()`.
* `tests/test_torch_parity.py` compares against `F.gelu(..., approximate="tanh")`, `F.relu` and
  `F.silu`. That pins down the *forward* definitions too (the 0.044715 constant, the tanh form).

## 8. Pitfalls

* **Forgetting one branch of $dx$** in SwiGLU. The weight gradients still pass gradcheck,
  because they don't depend on $dx$. Only the input gradient and everything *below* the MLP is
  wrong.
* **Dropping a chain-rule factor** (the $c(1+3kx^2)$ in GELU, or the $(1+a(1-s))$ in SiLU).
  GELU's error is order 0.5 on the activation alone. In the whole-model check it gets diluted
  to ~1e-2–1e-3 on some parameters (others still show 1.0), which is still far above
  `TOL_DEEP`. A diluted error is still an error.
* **Naive sigmoid** in float32: overflow warnings and NaN derivatives for large negative inputs.

## 9. Try it yourself

1. **Drop GELU's cubic factor.** In `GELU.backward`, replace `du = ...` with `du = _GELU_C`
   and run `python -m pytest tests -q`. *Expected:* `test_activations[GELU]` fails at 0.46,
   along with `test_mlp[gelu]`, `test_block[gpt2]`, `test_whole_model[gpt2]` and both gpt2
   parity cases. The kavi preset doesn't use GELU, so its tests pass.
2. **Forget the second branch.** Make `SwiGLU.backward` return only `self.w1.backward(da)`
   (still call `self.w3.backward(db)` for its weight grad). *Expected:* `test_swiglu` fails on
   `<input>` only (1.0), while `w1`, `w2`, `w3` pass at ~1e-8–1e-10. Kavi block/model/parity
   tests fail too.
3. **Break the naive sigmoid.** Evaluate `1/(1+np.exp(-a))` and `kavi.layers.sigmoid(a)` for
   `a = np.array([-1000., -100, 0, 100], dtype=np.float32)`. *Expected:* the same values
   `[0, 0, 0.5, 1]`, but only the naive one raises `RuntimeWarning: overflow encountered in exp`.
4. **Check parity.** For $C\in\{128, 256, 384\}$ compute `3*C*swiglu_hidden(C)` against `8*C*C`.
   *Expected:* $C=128$ gives $H=344$ (+0.8%), $C=256$ gives $H=680$ (−0.4%), and $C=384$ gives
   $H=1024$ (exactly equal).
