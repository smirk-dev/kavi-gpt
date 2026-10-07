# 08 · Optimizers: from SGD to AdamW

> **What you'll learn.** Backprop gives us the gradient. The optimizer decides what to *do*
> with it. This chapter derives why plain gradient descent struggles on transformers, then
> builds momentum, Adam (with a proof of its bias correction) and AdamW (with a proof that
> "decoupled" weight decay is different from L2 inside Adam), and finishes with the
> warmup + cosine schedule and global-norm gradient clipping, all mapped line by line onto
> `kavi/optim.py`.

Prerequisites: [backprop](03-linear-and-backprop.md), [the loss](07-loss.md). The training loop
that calls all of this is in [chapter 09](09-training.md).

---

## 1. Gradient descent and its weak spot

The gradient $g = \nabla_\theta L$ points uphill, so we step the other way:

$$
\theta \leftarrow \theta - \eta\, g .
$$

Why does this work, and how big can $\eta$ be? A second-order Taylor expansion with Hessian $H$:

$$
L(\theta - \eta g) \approx L(\theta) - \eta\,\|g\|^2 + \tfrac{\eta^2}{2}\, g^\top H g .
$$

For small $\eta$ the linear term wins and the loss goes down. For large $\eta$ the curvature
term wins and the loss goes *up*.

**Ill-conditioning.** Look at a quadratic bowl with very different curvatures along its axes,
$L = \tfrac12\sum_i \lambda_i \theta_i^2$. Then $g_i = \lambda_i\theta_i$, and one step gives

$$
\theta_i \leftarrow (1 - \eta\lambda_i)\,\theta_i .
$$

- Stability in the steepest direction requires $|1 - \eta\lambda_{\max}| < 1$, so $\eta < 2/\lambda_{\max}$.
- The flattest direction then shrinks by at most a factor $1 - 2\lambda_{\min}/\lambda_{\max}$
  per step. If the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ is $10^4$, that
  takes on the order of $10^4$ steps.

One global learning rate must be small enough for the sharpest direction, so it is far
too small for the flat ones. A transformer is full of parameters that live on very different
scales: the embedding row of a rare character gets a gradient only when that character shows
up; RMSNorm gains, attention matrices and the tied output head see gradients of very different
sizes; and deep and shallow layers differ too. No single $\eta$ suits them all.

**Noise.** We never compute the true gradient over all 4.8 M training characters. We use a
minibatch of 32 × 128 tokens. The minibatch gradient is an unbiased estimate of the true one,
with variance shrinking like $1/\text{batch size}$, so each step is partly random.

That's what happened to Anton: his first run with plain SGD produced garbage, and switching
to AdamW fixed it. Kavi keeps both so the comparison can be re-run (section 9).

## 2. Plain SGD in code

`SGD.step` with `momentum=0` is exactly $\theta \leftarrow \theta - \eta g$:

```python
p.data -= self.lr * g
```

`SGD` also has an optional `weight_decay` (added to the gradient). `scripts/train.py` doesn't
pass one, so the sweep's SGD run has no decay. For SGD, adding $\lambda\theta$ to the gradient
and decaying the weights directly are the *same thing*:
$\theta - \eta(g + \lambda\theta) = \theta - \eta\lambda\theta - \eta g$. Section 5 shows that
equivalence breaks for Adam.

## 3. Momentum: a ball rolling downhill

Anton's analogy is a skier: instead of re-deciding the direction at every step, you build up
speed in the direction the slope keeps pointing. `SGD` with `momentum=μ` does

$$
v \leftarrow \mu v + g, \qquad \theta \leftarrow \theta - \eta v .
$$

Unrolling (with $v_0 = 0$):

$$
v_t = g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \dots = \sum_{s \le t} \mu^{t-s} g_s .
$$

- If the gradient keeps pointing the same way ($g_s = g$), then $v_t \to g/(1-\mu)$: the
  effective learning rate becomes $\eta/(1-\mu)$, i.e. **10×** for $\mu = 0.9$. That speeds up
  the flat, consistent directions.
- If the gradient flips sign every step (the sharp direction of the bowl, bouncing between the
  walls), the terms alternate and largely cancel, which damps the oscillation.

The same quantity, normalised, is an **exponential moving average** (EMA):
$m_t = \beta m_{t-1} + (1-\beta) g_t$ satisfies $m_t = (1-\beta) v_t$ when $\mu = \beta$. Adam uses
the EMA form.

## 4. Adam: a separate step size for every parameter

Adam keeps two EMAs per parameter (code: `AdamW.m`, `AdamW.v`, both start at 0):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t &&\text{(mean of the gradient: direction)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 &&\text{(mean of the squared gradient: scale)}\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

with all operations elementwise, and the bias-corrected $\hat m, \hat v$ defined next.

### 4.1 Bias correction, derived

Unroll $m$ from $m_0 = 0$:

$$
m_t = (1-\beta_1)\sum_{s=1}^{t}\beta_1^{t-s} g_s .
$$

Suppose the gradients are drawn from a fixed distribution with mean $\mathbb{E}[g]$. Take the
expectation and sum the geometric series $\sum_{s=1}^t \beta_1^{t-s} = (1-\beta_1^t)/(1-\beta_1)$:

$$
\mathbb{E}[m_t] = (1-\beta_1)\,\mathbb{E}[g]\sum_{s=1}^{t}\beta_1^{t-s}
= (1-\beta_1)\,\mathbb{E}[g]\,\frac{1-\beta_1^t}{1-\beta_1}
= (1-\beta_1^t)\,\mathbb{E}[g] .
$$

So $m_t$ is biased toward zero by the factor $1-\beta_1^t$, because it started at 0. Dividing it
out removes the bias. The same argument works for $v$ with $g^2$:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} .
$$

In code these are `c1 = 1 - b1**t`, `c2 = 1 - b2**t`, and the update is
`p.data -= lr * (m / c1) / (sqrt(v / c2) + eps)`.

**The first step.** At $t=1$: $m_1 = (1-\beta_1)g$, so $\hat m_1 = g$, and likewise
$\hat v_1 = g^2$. The update is $\eta\, g/(|g|+\epsilon) \approx \eta\,\operatorname{sign}(g)$:
**every parameter moves by about $\eta$ at once**, however small its gradient. Remember this
for warmup (section 7).

### 4.2 Why this fixes ill-conditioning

- **Scale invariance.** Multiply one parameter's gradients by $c$: $\hat m$ scales by $c$,
  $\sqrt{\hat v}$ by $|c|$, and the ratio doesn't change. Each parameter's step is about $\eta$
  whether its raw gradients are $10^{-5}$ or $10$. That's a diagonal approximation of
  "divide by the curvature", and it is what lets the rare-character embedding and the
  busy attention matrix share one learning rate.
- **Signal to noise.** $\hat m/\sqrt{\hat v} \approx \mathbb{E}[g]/\sqrt{\mathbb{E}[g^2]}$, which
  has magnitude at most 1 (Cauchy-Schwarz). If the gradient is consistent the ratio is near
  $\pm1$ and the step is large. If it is mostly noise ($\mathbb{E}[g] \approx 0$ but
  $\mathbb{E}[g^2]$ large), the ratio is small and the parameter barely moves.
- $\epsilon = 10^{-8}$ prevents division by zero and caps the step of a parameter whose
  gradient is essentially zero. (Example: the provably-zero key bias of
  [chapter 05](05-attention.md), whose float64 round-off gradient is far below $\epsilon$.)

## 5. AdamW: why weight decay must be decoupled

Weight decay pulls weights toward zero each step, a preference for small weights that
regularises the model. The classical way is **L2 regularisation**: add $\tfrac{\lambda}{2}\|\theta\|^2$
to the loss, so the gradient becomes $g + \lambda\theta$. Put that into Adam:

$$
m \leftarrow \beta_1 m + (1-\beta_1)(g + \lambda\theta), \qquad
\Delta\theta = -\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon} .
$$

The decay term now passes through the same $1/\sqrt{\hat v}$ as the gradient. In steady state its
contribution is roughly $\eta\lambda\theta/\sqrt{\hat v}$:

- parameters with large gradient history (big $\hat v$) get *weak* decay;
- parameters with tiny gradients (small $\hat v$) get *huge* decay.

That's the opposite of a uniform "shrink every weight by a fixed fraction", and $\hat v$ includes
the decay term too, which muddles things further. Loshchilov & Hutter's fix is to keep the
decay out of the adaptive part and apply it directly to the weights:

$$
\theta \leftarrow \theta - \eta\lambda\,\theta - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon} .
$$

Now every decayed weight shrinks by the same fraction $\eta\lambda$ per step, and the
gradient part is unaffected. In `AdamW.step`:

```python
if self.wd and p.decay:
    p.data -= self.lr * self.wd * p.data          # decoupled decay, scaled by the current lr
p.data -= self.lr * (m / c1) / (xp.sqrt(v / c2) + self.eps)
```

The decay is multiplied by the *scheduled* lr, so it follows warmup and cosine decay. This is
the same convention as `torch.optim.AdamW`, which is why the parity test can match it to 1e-10.
With our settings ($\eta = 2\cdot10^{-3}$, $\lambda = 0.1$) that's $2\cdot10^{-4}$ of each weight
per step at peak lr.

### 5.1 Which parameters decay: `Param.decay`

Every `Param` carries a `decay` flag (`kavi/module.py`), set where the parameter is created in
`kavi/layers.py`:

| parameter | `decay` | why |
|---|---|---|
| `Linear.weight` (qkv, proj, MLP) | True | the big matrices: decay limits their norm |
| `Embedding.weight` (`wte`, tied to the head; `wpe`) | True | matrices too |
| `Linear.bias` | False | few parameters, no real capacity to overfit; zero isn't special for a bias |
| `LayerNorm`/`RMSNorm` weight (gain) and bias | False | a gain's natural value is 1, not 0; decaying it would shrink every activation |

## 6. Our betas: (0.9, 0.95)

$\beta_1 = 0.9$ averages the gradient over about $1/(1-\beta_1) = 10$ steps. For $\beta_2$ the
textbook default is 0.999 (about 1000 steps); we use **0.95** (about 20 steps), as is common for
transformer language models. A shorter memory for $v$ means that when gradient scales change
(and in early training they change fast), the denominator catches up within tens of steps
instead of thousands. With a stale, too-small $\hat v$, a sudden larger gradient produces an
oversized step, which is a classic cause of loss spikes. A side effect is that bias correction
fades fast: $1 - 0.95^{t}$ is above 0.95 after 59 steps.

These are the `beta1=0.9, beta2=0.95` defaults in `scripts/train.py` and `AdamW.__init__`.

## 7. Learning-rate schedule: warmup then cosine (`lr_at`)

```python
if step < warmup:  return max_lr * (step + 1) / warmup
if step >= total:  return min_lr
progress = (step - warmup) / max(1, total - warmup)
return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

$$
\eta(t) =
\begin{cases}
\eta_{\max}\,\dfrac{t+1}{W} & t < W\\[6pt]
\eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\big(1 + \cos(\pi p)\big),\quad p = \dfrac{t-W}{T_{\text{tot}}-W} & W \le t < T_{\text{tot}}\\[6pt]
\eta_{\min} & t \ge T_{\text{tot}}
\end{cases}
$$

Checks: at $p=0$, $\cos 0 = 1$ gives $\eta_{\max}$, so the two pieces join (both pieces equal
$\eta_{\max}$ at $t = W-1$ and $t = W$). At $p = 1$, $\cos\pi = -1$ gives $\eta_{\min}$. The
`+1` in the warmup means step 0 already has a non-zero lr.

For the char run (`max_lr 2e-3, min_lr 2e-4, warmup 150, total 2000`):

| step | 0 | 1 | 149 | 150 | 1000 | 1999 | 2000 |
|---|---|---|---|---|---|---|---|
| lr | 1.33e-5 | 2.67e-5 | 2.0e-3 | 2.0e-3 | 1.214e-3 | 2.0e-4 | 2.0e-4 |

The training log (`runs/char_kavi_cpu.log`) agrees: `step 0 | lr 1.33e-05`, `step 150 | lr 2.00e-03`.

**Why warmup helps Adam.** Section 4.1 showed the first update is about $\eta\operatorname{sign}(g)$ for
*every* parameter at once, based on a single noisy gradient, and $\hat v$ stays a rough estimate
for the first tens of steps. At initialisation the gradients are also at their largest and
change fastest: our log shows a gradient norm of 4.22 at step 0 and about 0.5 by step 500. Starting
with a tiny lr and ramping up means the steps only get large once the moment estimates are
reliable and the model has left the chaotic initial region.

**Why cosine.** Large steps make fast progress far from a minimum. Near the end, minibatch noise
keeps a large-lr model bouncing around the bottom of the valley, and shrinking the step lets it
settle. Cosine decay is a smooth way to do this: it stays near $\eta_{\max}$ for a while, then
drops, and flattens out at $\eta_{\min}$.

## 8. Global-norm gradient clipping (`clip_grad_norm`)

One unlucky batch can produce a gradient many times larger than usual, and a single huge step
can undo hours of training. Clipping caps the step:

$$
\|g\| = \sqrt{\sum_{\text{params } p}\ \sum_{\text{elements}} g_{p}^2}, \qquad
g \leftarrow g\cdot\frac{c}{\|g\| + 10^{-6}} \quad\text{if } \|g\| > c .
$$

- **Global**: one norm over *all* parameters together, and every gradient is multiplied by the
  same factor. The update **direction is unchanged**, only its length shrinks. Clipping each
  element to $[-c, c]$ instead would change the direction.
- It returns the **pre-clip norm**, which `train.py` logs as `gnorm`. Spikes in `gnorm` are the
  earliest warning of instability, often visible before the loss jumps.
- The sum stays on the device and is converted to a Python float once, because on a GPU every
  `float()` forces a host/device sync ([chapter 11](11-gpu.md)).

In our char run, `grad_clip = 1.0`: step 0 had `gnorm 4.22` (clipped by about 4×), and from step
50 onward it hovered between 0.44 and 0.73, so clipping was inactive. That's the normal
picture: clipping is insurance against rare spikes, not a constant brake. A subtle point is
that Adam is scale-invariant (section 4.2), so clipping *every* step by the same factor would
change nothing. Clipping matters for steps that are large *compared to the recent history*,
because it keeps one bad batch from polluting $m$ and $v$.

Order in the training loop (`scripts/train.py`): `zero_grad` → `forward` → `backward` →
`clip_grad_norm` → set `opt.lr = lr_at(step)` → `opt.step()`.

## 9. How we know it's right

- **`tests/test_torch_parity.py::test_adamw_matches_torch`**: a `(4,3)` matrix with decay and a
  `(3,)` vector with `decay=False`, 5 steps of random gradients through our `AdamW` and through
  `torch.optim.AdamW` (with matching weight-decay groups). Weights agree to `rtol=1e-10`. This
  checks the moments, bias correction, $\epsilon$ placement, decoupled decay, and the `decay` flag.
- **`tests/test_training.py::test_adamw_overfits_one_batch`**: for both presets, 300 AdamW
  steps on one fixed batch (vocab 20, so the initial loss is about $\ln 20 \approx 3.0$) must reach a loss below 0.05.
  A sign error, a missing gradient or a stale cache anywhere in the stack shows up as a stalled loss.
- **`test_sgd_is_much_slower_than_adamw`**: the Anton lesson as a test. Same model and batch,
  150 steps, SGD at lr 0.5 vs AdamW at 3e-3, both clipped to norm 1. Measured: SGD drops below
  loss 1.0 at step 60 and AdamW at step 35. Below 0.1 at step 108 vs 73. Final loss 0.0133 vs
  0.0050 (the test requires AdamW < 0.5 × SGD; the ratio is 0.37). On one tiny batch, a
  well-tuned SGD does get there eventually; the gap is speed, and it widens on real data.
- **The ablation sweep** (`configs/sweep_ablation.json`) includes `kavi-sgd-s1`: the full Kavi model
  trained with plain SGD (`lr 0.1 → 0.01`, no momentum) on the same budget as `kavi-s1`.
  Result: **2.238 val bpb vs 1.627** for AdamW. After 4000 steps SGD still hadn't reached where
  AdamW was at step 400 (2.003). Details and samples are in [the journal](journal.md).

## 10. Pitfalls

- **L2 inside Adam is not weight decay.** Adding `wd * θ` to the gradient before Adam (section 5)
  gives uneven decay. Use the decoupled form.
- **Decaying norm gains and biases** pulls gains toward 0 and fights the normalisation. That's why
  `decay=False` exists.
- **No warmup with a high lr** risks the sign-like first steps (section 4.1) wrecking the
  initialisation. Watch for a loss that rises in the first few dozen steps.
- **Forgetting bias correction** leaves the factor $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$ in every early
  update. With our betas that's 0.45 at step 1 and about 1.1 around step 20, so early steps come
  out first too small and then too large (exercise 1).
- **Resuming without optimizer state.** $m$, $v$ and $t$ are part of the training state.
  `AdamW.state_dict` saves them, and the checkpoint stores them ([chapter 09](09-training.md)).
- **Clipping the wrong thing.** Clip *after* `backward` and *before* `step`, on the gradients,
  with one global norm.

## 11. Try it yourself

1. **Bias correction by hand.** Feed `AdamW` a constant gradient `g = 0.01` for 5 steps on one
   scalar parameter (`weight_decay=0`) and print each update. *Expected:* every update is
   $\eta\cdot 0.01/(0.01+10^{-8}) \approx \eta$, from step 1 on. Then comment out the `/ c1` and
   `/ c2`: the update becomes $\eta\,(1-0.9^t)/\sqrt{1-0.95^t}$, which is $0.45\eta$ at step 1,
   overshoots to about $1.1\eta$ around step 20, and only settles at $\eta$ after about 100 steps.
2. **The condition number.** Minimise $L = \tfrac12(\theta_1^2 + 1000\,\theta_2^2)$ from $(1,1)$
   with `SGD` (pick the largest stable lr, just under $2/1000$) and with `AdamW(weight_decay=0)`
   at lr 0.01. *Expected:* SGD shrinks $\theta_1$ by only 0.2% per step and needs thousands of steps;
   Adam moves both coordinates at about the same speed and gets both near zero in a few hundred
   steps (then jitters at a scale of about lr).
3. **Plot the schedule.** Plot `lr_at(s, 2e-3, 2e-4, 150, 2000)` for `s in range(2200)`.
   *Expected:* a straight ramp to 2e-3 over 150 steps, a half-cosine down to 2e-4 at step
   2000, then flat.
4. **Clipping preserves direction.** Make random grads on three `Param`s, record the flattened
   concatenation, call `clip_grad_norm(params, 0.1)`, and compare. *Expected:* the returned value
   is the old norm, the new norm is about 0.1, and the cosine similarity between old and new is 1.0.
