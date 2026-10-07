# 03 — Linear layers, backpropagation, and how we prove gradients right

> **Start here for backprop.** Every other chapter's backward pass uses the machinery set up
> in this one.

**What you'll learn.** How a network's gradient is computed by walking a computational graph
backwards, one "upstream gradient × local derivative" step at a time, and why we never build a
Jacobian matrix to do it. You'll derive the backward pass of `Linear` (the brick that holds
most of Kavi's parameters) with every shape checked. And you'll see how Kavi *proves* every
backward pass correct with finite differences and a PyTorch cross-check, including the two
cases where the naive version of that proof would lie to you.

---

## 1. The computational graph

A forward pass is a chain (really a DAG) of functions. For one training step of Kavi:

```
ids ─▶ Embedding ─▶ x0 ─▶ Block ─▶ x1 ─▶ … ─▶ Norm ─▶ h ─▶ h·wteᵀ ─▶ logits ─▶ CE ─▶ L (scalar)
```

Every node turns its input tensor into an output tensor; at the very end sits a single number,
the loss $L$. Training needs $\partial L/\partial\theta$ for every parameter $\theta$.

**Notation used throughout the book.** For any tensor $X$ in the graph, write
$$dX \;\equiv\; \frac{\partial L}{\partial X},$$
a tensor with *the same shape as $X$*. In the code this is the `d` prefix: `dy`, `dx`, `dxhat`.
"Same shape as $X$" is the single most useful sanity check in this whole book.

### The chain rule as "upstream × local"

Take one node $y = f(x)$ with $x\in\mathbb{R}^n$, $y\in\mathbb{R}^m$. $L$ depends on $x$ only
through $y$, so by the multivariable chain rule
$$
dx_j \;=\; \frac{\partial L}{\partial x_j} \;=\; \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}\,\frac{\partial y_i}{\partial x_j}
\;=\; \sum_i dy_i \, J_{ij},
\qquad J = \frac{\partial y}{\partial x}\in\mathbb{R}^{m\times n}.
$$
In vector form, $dx = J^\top dy$. Read it as:

* $dy$ is the **upstream gradient** — how much the loss cares about each output. It arrives
  from the layer above; this node doesn't need to know anything about what is up there.
* $J$ is the **local Jacobian** — how this node's outputs move when its inputs move. It
  depends only on this node.

Backprop is just this rule applied node by node, starting from $dL = 1$ at the top. Each node
receives $dy$, produces $dx$, and hands $dx$ down as the next node's upstream gradient.

### Why we never form the Jacobian

$J^\top dy$ is a **vector–Jacobian product** (VJP), and the point of every derivation in this
book is to compute it *without* materialising $J$. Consider one `qkv` projection in a training
step of the BPE config: input $(B\cdot T, C) = (32\cdot256, 256) = (8192, 256)$, output
$(8192, 768)$. Flattened, $x$ has $2.1\times10^6$ entries and $y$ has $6.3\times10^6$, so $J$
would have about $1.3\times10^{13}$ entries — 52 TB in float32. Yet almost all of them are zero
(output row $n$ doesn't depend on input row $m\ne n$), and the non-zero ones are just copies of
$W$. Working the sum $\sum_i dy_i J_{ij}$ out *by hand* collapses it into one matrix product
(§3). That "work out the sum symbolically, end up with something cheap" pattern repeats in
every chapter: LayerNorm's Jacobian is a dense $C\times C$ matrix per token, but its VJP is
three $O(C)$ terms ([chapter 04](04-normalization.md)).

### Fan-out: gradients from several uses add up

If a tensor is used in two places, $L$ depends on it through both, and the chain rule sums
over both paths:
$$ y_1 = f(x),\; y_2 = g(x) \;\Rightarrow\; dx = J_f^\top dy_1 + J_g^\top dy_2 .$$
Three places in Kavi rely on this:

1. **Residual connections.** In `Block.backward`, `dx = dy + self.norm2.backward(...)`: the
   input of `x + mlp(norm(x))` feeds both the skip path (local Jacobian $I$) and the branch.
2. **SwiGLU**, where $x$ feeds two projections ([chapter 06](06-mlp.md)).
3. **Weight tying**, where the token embedding `wte` is both the input lookup table and the
   output projection ([chapter 02](02-embeddings.md)).

## 2. The Module contract (`kavi/module.py`)

There is no autograd, so each brick implements its own VJP. The contract in `Module`:

* `forward(x)` computes the output **and caches** whatever backward will need (inputs,
  intermediate values like LayerNorm's $\hat x$ and $r$).
* `backward(dout) -> dx` receives the upstream gradient, **accumulates** $\partial L/\partial
  \theta$ into each `Param.grad` with `+=`, and **returns** $\partial L/\partial x$.

`Param` is just `data` plus a `grad` buffer of the same shape (and a `decay` flag the optimiser
reads, [chapter 08](08-optimizers.md)).

**Why `+=` and not `=`.** Because of fan-out on parameters. The tied embedding matrix gets one
gradient contribution in `GPT.backward` from the output head
(`W.grad += dlogits... .T @ self.h...`) and a second from `self.wte.backward(dx)` (a
scatter-add). If either used `=`, it would silently erase the other. Accumulation makes the
order irrelevant and costs nothing — the price is that `zero_grad()` must be called once per
optimisation step, otherwise gradients from the previous step leak in.

`named_params()` deduplicates by `id()`, so a shared `Param` is listed (and updated by the
optimiser) exactly once even though two modules reference it.

**Caches are by reference.** `Linear.forward` stores `self.x = x` without copying. If anyone
modified `x` in place between forward and backward, the gradient would be computed against the
wrong input. Nothing in Kavi does that, but it's the first thing to suspect if a gradient is
mysteriously wrong.

## 3. `Linear`: deriving the backward pass

### Forward

`Linear` computes $y = xW + b$ with $W\in\mathbb{R}^{C_{in}\times C_{out}}$ (note: *input-major*,
so the forward is `x @ W` with no transpose) and $b\in\mathbb{R}^{C_{out}}$. Activations are
$(B, T, C_{in})$; the layer treats every one of the $N = B\cdot T$ positions identically, so
first flatten to $X\in\mathbb{R}^{N\times C_{in}}$:
$$ Y_{no} \;=\; \sum_{i=1}^{C_{in}} X_{ni}\,W_{io} + b_o, \qquad Y\in\mathbb{R}^{N\times C_{out}}. $$

### $dW$

$W_{io}$ appears in $Y_{no}$ for **every** row $n$ (the same weights are applied at every
position), and in no other output column. So the chain rule sums over $n$:
$$
dW_{io} = \sum_{n}\sum_{o'} dY_{no'}\,\frac{\partial Y_{no'}}{\partial W_{io}}
= \sum_n dY_{no}\, X_{ni}
\quad\Longrightarrow\quad
\boxed{dW = X^\top dY}
$$
Shapes: $(C_{in}\times N)(N\times C_{out}) = C_{in}\times C_{out}$ ✓, same as $W$. The sum over
batch *and* time positions is hidden inside the matrix product's inner dimension $N$ — this is
"a parameter shared across positions accumulates the gradient from every position", the same
fan-out rule as above.

### $db$

$\partial Y_{no'}/\partial b_o = \delta_{oo'}$ for every $n$, so
$$ \boxed{db_o = \sum_n dY_{no}} \qquad (C_{out},)\ ✓ $$
This is a general duality worth remembering: **broadcast in the forward pass ⇔ sum in the
backward pass.** The bias is broadcast over $N$ rows, so its gradient sums over them.
`GPT.backward` uses the same rule for learned positions: `wpe` output of shape $(1,T,C)$ is
broadcast over the batch, so its gradient is `dx.sum(axis=0, keepdims=True)`.

### $dX$

$\partial Y_{n'o}/\partial X_{ni} = \delta_{nn'}W_{io}$: row $n$ of the input only affects row
$n$ of the output.
$$ dX_{ni} = \sum_{o} dY_{no}\,W_{io} \quad\Longrightarrow\quad \boxed{dX = dY\,W^\top} $$
Shapes: $(N\times C_{out})(C_{out}\times C_{in}) = N\times C_{in}$ ✓. Then reshape back to
$(B,T,C_{in})$.

A quick way to *remember* (not derive) these: each gradient is the only product of the
available matrices whose shape works out. $dW$ must be $C_{in}\times C_{out}$ and only
$X^\top dY$ fits. Use the shape trick as a check, never as a derivation — it cannot tell you
about sums over repeated indices or element-wise factors.

### Code walkthrough (`kavi/layers.py::Linear`)

| Math | Code |
|---|---|
| flatten $(B,T,C_{in})\to(N,C_{in})$ | `x.reshape(-1, self.n_in)` |
| $Y = XW + b$ | `y = x2 @ self.weight.data; y += self.bias.data` |
| $dW \mathrel{+}= X^\top dY$ | `self.weight.grad += x2.T @ dy2` |
| $db \mathrel{+}= \sum_n dY_n$ | `self.bias.grad += dy2.sum(axis=0)` |
| $dX = dY W^\top$, unflatten | `(dy2 @ self.weight.data.T).reshape(*dy.shape[:-1], self.n_in)` |

Weights are initialised $\mathcal N(0, 0.02^2)$ (the GPT-2 convention); the output projections
of each residual branch use a smaller std, explained in [chapter 06](06-mlp.md).

### Performance note: why flatten at all

NumPy's `(B,T,C) @ (C,N)` is a *stacked* matmul: it runs $B$ separate small GEMMs. Reshaping to
one $(B\cdot T, C)$ matrix issues a single large BLAS GEMM, which keeps the CPU's vector units
and caches busy. We measured the flattened version **4.4× faster** on this laptop — and since
`Linear` is where nearly all the FLOPs are, that is close to a 4× faster training step. The
reshape is free (it's a view of contiguous memory), and the backward pass reuses it.

---

## 4. Gradient checking: proving the backward pass

A derivation can be wrong, and a wrong gradient doesn't crash — the model just trains a bit
worse, or not at all. Anton's series "double-checked the gradient code"; Kavi checks it with
numbers instead (`kavi/gradcheck.py`).

### Centred differences and their truncation error

For a scalar function $f(\theta)$, Taylor-expand around $\theta$ in both directions:
$$
f(\theta\pm h) = f(\theta) \pm h f'(\theta) + \tfrac{h^2}{2} f''(\theta) \pm \tfrac{h^3}{6} f'''(\theta) + O(h^4).
$$
Subtract, and the even-order terms cancel:
$$
\frac{f(\theta+h) - f(\theta-h)}{2h} = f'(\theta) + \frac{h^2}{6} f'''(\theta) + O(h^4).
$$
So the centred difference has **truncation error $O(h^2)$**, versus $O(h)$ for the one-sided
$(f(\theta+h)-f(\theta))/h$. That's `_numeric` in the code: perturb one element in place by
$+h$, evaluate, by $-h$, evaluate, restore.

### Rounding error and the trade-off

Each evaluation of $f$ is computed in floating point with some absolute error $\approx
\varepsilon_f$ (at best $\varepsilon\,|f|$ with machine epsilon $\varepsilon$, in practice more,
because a model's forward pass is thousands of rounded operations). The difference of two such
values carries error $\sim 2\varepsilon_f$, which we then *divide by $2h$*:
$$
\text{error}(h) \;\approx\; \underbrace{\frac{h^2}{6}\,|f'''|}_{\text{truncation}} \;+\; \underbrace{\frac{\varepsilon_f}{h}}_{\text{rounding}} .
$$
Shrinking $h$ kills the first term but blows up the second. The sum is minimised near
$h^\ast \sim (3\varepsilon_f/|f'''|)^{1/3}$ with best error $\sim \varepsilon_f^{2/3}$.

**Why float64.** Float32 has $\varepsilon\approx 1.2\times10^{-7}$, so even ideally the best
achievable error is $\sim 10^{-5}$ relative — the same size as a subtle bug. Float64 has
$\varepsilon\approx 2.2\times10^{-16}$, best error $\sim10^{-11}$, leaving many orders of
magnitude between "correct" and "buggy". The test fixture in `tests/conftest.py` switches the
backend to float64 for every test. To see what you'd get otherwise, here is the *correct*
`Linear` and `LayerNorm` checked in float32 (worst relative error over 40 sampled elements):

| $h$ | Linear $dW$ | LayerNorm $dx$ |
|---|---|---|
| $10^{-2}$ | 9.3e-04 | 2.6e-04 |
| $10^{-3}$ | 1.3e-02 | 3.7e-03 |
| $10^{-4}$ | 1.8e-01 | 8.1e-03 |

Correct code, errors up to 18%: float32 gradchecks are useless.

### Our measurement: choosing $h$

On the whole-model check (float64), one parameter element's relative error against step size:

| $h$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-7}$ |
|---|---|---|---|---|---|
| rel. error | 1.3e-07 | **1.5e-08** | 5.8e-07 | 8.8e-06 | 5.8e-05 |

Read it right to left: from $10^{-7}$ up to $10^{-4}$ the error falls roughly $10\times$ per
decade of $h$ — the $\varepsilon_f/h$ rounding regime. From $10^{-4}$ to $10^{-3}$ it rises
again — truncation ($\propto h^2$) taking over. The optimum is larger than the textbook
float64 $h^\ast\approx10^{-5}$ because the loss of a whole transformer accumulates far more
rounding than one $\varepsilon|f|$. So `check()` defaults to **$h = 10^{-4}$**.

### The relative error and the tolerances

`_rel_err` compares analytic $a$ with numeric $n$ as
$$ \text{rel}(a,n) = \frac{|a-n|}{|a|+|n|}. $$
Scale-free (a gradient of $10^{-6}$ and one of $10^{3}$ are judged alike), and bounded by 1: a
value of **1.0 means opposite signs or one side zero** — the backward pass is not just
inaccurate but pointing the wrong way. For small errors it is about half the
$|a-n|/|a|$ you may have seen elsewhere.

The tests (`tests/test_gradcheck.py`) use two thresholds:

* `TOL = 1e-6` for single bricks. A correct brick lands at $10^{-9}$–$10^{-12}$.
* `TOL_DEEP = 1e-5` for composites (`test_block`, `test_whole_model`). Errors compound
  through many layers and the $h$ trade-off leaves $\sim10^{-6}$ of noise: the correct gpt2
  whole-model check measures a worst element of 1.6e-06.

Real bugs are nowhere near either: deleting LayerNorm's mean term produces a median relative
error of 0.22 and a max of 1.0. There is a factor of $10^4$ of daylight.

### The exactly-zero gradient (and why `_rel_err` has a special case)

Run the attention gradcheck with biases but without RoPE, and one parameter fails with
relative error ≈ 1.0 even though the code is right: the **key bias**, the middle third of the
`qkv` bias vector.

Why: with key bias $b$, query $i$'s score against key $j$ is
$$ s_{ij} = \frac{q_i\cdot(k_j + b)}{\sqrt D} = \frac{q_i\cdot k_j}{\sqrt D} + \underbrace{\frac{q_i\cdot b}{\sqrt D}}_{\text{same for every } j}. $$
The second term is a constant added to *every* entry of query $i$'s score row, and softmax is
shift-invariant: $\mathrm{softmax}(s + c\mathbf 1) = \mathrm{softmax}(s)$. So the attention
weights, the output and the loss don't depend on $b$ at all, and $\partial L/\partial b \equiv
0$ exactly. (Derivation of attention itself: [chapter 05](05-attention.md).)

Our analytic backward pass agrees — it returns values like $5.6\times10^{-17}$ and $2\times
10^{-19}$, which are just rounding residue from sums that cancel. The finite difference
returns pure noise, $\sim10^{-12}$ (we measured $0$, $-4.4\times10^{-12}$, $8.9\times
10^{-12}$ on three elements). Both are "zero", but $|a-n|/(|a|+|n|)$ of two tiny unrelated
numbers is $\approx 1$. Relative error is meaningless when the true value is 0.

So `_rel_err` switches to an absolute test: if $|a|+|n| < 10^{-7}$ it reports 0 when
$|a-n| < 10^{-8}$ and 1 otherwise. A truly wrong gradient that happens to be tiny still has
to agree in absolute terms. With RoPE the case vanishes: keys are rotated by a
position-dependent angle *after* the bias is added, so $q_i\cdot R_j b$ varies with $j$ and the
key bias gets a real gradient (max $|g| = 0.64$ in the same test). Kavi defaults to
`bias=False` anyway, so in practice there is no wasted parameter.

### Why the objective is $L = \sum \text{out}\odot G$ with random $G$

`check_module` needs a scalar objective for a module that outputs a tensor. It uses
$L = \sum_{\ldots} \text{out} \odot G$ with $G$ drawn once from $\mathcal N(0,1)$. Then
$\partial L/\partial\,\text{out} = G$, so calling `backward(G)` *is* the VJP with an arbitrary
upstream gradient — exactly what a real layer above would send.

The tempting $L = \sum \text{out}$ ($G = \mathbf 1$) is a weak test. Every output element gets
the same weight, so bugs that permute or mix output positions can cancel. Concretely, for
LayerNorm with unit gains, $\sum_i \hat x_i = 0$ for every row, so $L = \sum\text{out}$ is
constant and the true $dx$ is zero. A LayerNorm backward with the whole $\sigma$ term
deleted *also* returns exactly zero for $G=\mathbf 1$ (measured: max $|dx| = 0.0$) and would
pass. With random $G$ it fails immediately.

### Why the tests randomise gains and biases

A fresh LayerNorm has $g\equiv1$, $b\equiv0$. If backward forgets to multiply by $g$
(`dxhat = dy` instead of `dy * self.weight.data`), the bug is invisible because multiplying
by 1 does nothing: we measured a worst error of 4.5e-07, which **passes** `TOL`. After
`randomise()` perturbs every parameter, the same bug scores 1.0. That is why every brick test
calls `randomise(m)` first, and the torch-parity test perturbs all parameters too. The
lesson generalises: test at a *generic* point, never at a special one where terms vanish.

## 5. The second oracle: torch parity

Finite differences prove that backward is the derivative of *our* forward. They cannot tell
you whether our forward is the function we *meant*. Mistype GELU's constant as `0.04715` and
gradcheck still passes, because forward and backward are consistently wrong.

`tests/test_torch_parity.py` closes that gap. `torch_loss` rebuilds the whole model from
standard PyTorch ops (`F.layer_norm`, `F.gelu(approximate="tanh")`, `F.silu`,
`F.cross_entropy`, its own RoPE), loads our exact weights in float64, and lets autograd produce
the gradients. `test_model_matches_torch` then demands, for all four preset/bias combinations,
that the loss agrees to $10^{-10}$ and every parameter gradient to `rtol=1e-7, atol=1e-11`.
The two oracles fail in different ways, so passing both is strong evidence. (Torch is a
test-only dependency; the test skips if it's missing.)

## 6. Pitfalls

* **Gradchecking in float32.** See the table above: correct code shows errors up to $10^{-1}$.
* **Non-differentiable points.** ReLU has a kink at 0; a finite difference straddling it is
  meaningless. `test_activations` moves inputs with $|x|<10^{-3}$ to 0.5 first.
* **Randomness inside forward.** With dropout on, every forward draws a new mask, so the
  "function" being differentiated changes between the $+h$ and $-h$ evaluations. Gradcheck
  with `p=0` (the default in the tests).
* **Forgetting `zero_grad()`.** Because grads accumulate, the second step adds to the first.
  `check_module` calls `module.zero_grad()` before its analytic backward for this reason.
* **Checking at a special point.** Unit gains, zero biases, $G=\mathbf 1$: see §4.

## 7. How we know it's right

`tests/test_gradcheck.py::test_linear[True/False]` (with and without bias) checks $dW$, $db$
and $dx$ on a $(2,3,6)$ input — the 3-D shape matters, since it exercises the flattening and
the sum over positions. `test_whole_model` checks every parameter of both presets end to end,
and `test_model_matches_torch` cross-checks against autograd.

## 8. Try it yourself

1. **Lose the batch sum.** In `Linear.backward`, change `x2.T @ dy2` to
   `self.x[0].T @ dy[0]` (use only the first batch element). Run
   `python -m pytest tests/test_gradcheck.py -q -k linear`. *Expected:* both `test_linear`
   cases fail on `weight` with relative errors of order 0.1–1; `bias` and `<input>` still pass.
2. **Sweep $h$ yourself.** Call `check_module(m, x, h=h)` on a randomised `LayerNorm(8)` for
   $h\in\{10^{-2},\dots,10^{-8}\}$, first in float64, then after
   `B.set_precision("float32")`. *Expected:* float64 bottoms out around $10^{-9}$–$10^{-10}$
   for $h\approx10^{-4}$–$10^{-5}$ and gets worse on both sides; float32 never gets below
   $\sim10^{-4}$.
3. **Remove the zero-gradient guard.** Temporarily replace `_rel_err`'s body with plain
   `abs(a-n)/(abs(a)+abs(n))` (guard against 0/0) and run `-k attention`. *Expected:* exactly
   one of the four cases fails — no RoPE, with bias — on `qkv.bias` with error 1.0. Then print
   `m.qkv.bias.grad[C:2*C]` and confirm it's $\sim10^{-16}$.
4. **Break weight tying.** In `GPT.backward`, comment out the line
   `W.grad += dlogits.reshape(-1, V).T @ ...`. *Expected:* `test_whole_model` fails on
   `wte.weight` with relative error 1.0; every other parameter still passes, because only the
   tied matrix lost a path.
