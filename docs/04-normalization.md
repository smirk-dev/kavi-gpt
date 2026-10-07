# 04 — Normalization: LayerNorm and RMSNorm

**What you'll learn.** Why a deep network needs to keep resetting the scale of its activations,
and how LayerNorm and RMSNorm do it. You'll get the full derivation of both backward passes,
the one Anton said "took a whole afternoon to understand". The derivation is done carefully
enough that RMSNorm's formula falls out as "LayerNorm minus one term". Then you'll see where
the norm goes in a block (pre-norm vs post-norm) and why that placement decides whether a
transformer trains at all.

Prerequisites: VJPs and the Module contract from [chapter 03](03-linear-and-backprop.md).

---

## 1. Why normalise

Each block *adds* its output to the residual stream: $x_{l+1} = x_l + F_l(\cdot)$. Nothing
stops the stream's magnitude from drifting as the additions pile up. We measured it on the
char run's best checkpoint (8 validation sequences, RMS over all elements):

| after | embedding | block 1 | block 2 | block 3 | block 4 |
|---|---|---|---|---|---|
| untrained | 0.020 | 0.021 | 0.023 | 0.024 | 0.027 |
| trained | 0.087 | 0.659 | 1.041 | 1.530 | 2.431 |

After training, the stream grows **28×** in four blocks. Hand that to a Linear layer directly
and its outputs, and everything downstream, change scale with depth. Softmax saturates on large
inputs, and step sizes tuned for one scale are wrong for another. A normalization layer makes
each sublayer see inputs of a *fixed* scale, whatever the stream is doing. It also decouples
"which direction is this vector pointing" (what the norm preserves) from "how long is it"
(what it throws away), which makes the optimisation better conditioned.

Both norms here work on **each token's $C$-vector independently**, over the last axis. No
statistics are shared across the batch (unlike BatchNorm). So it behaves the same in training
and inference, and with batch size 1.

## 2. LayerNorm

### Forward

For one token's vector $x\in\mathbb{R}^C$ (`kavi/layers.py::LayerNorm.forward`):
$$
\mu = \frac1C\sum_j x_j,\qquad
\text{var} = \frac1C\sum_j (x_j-\mu)^2,\qquad
r = \frac{1}{\sqrt{\text{var}+\epsilon}},
$$
$$
\hat x_i = (x_i - \mu)\,r,\qquad y_i = g_i\,\hat x_i + b_i .
$$
$\hat x$ has mean 0 and (up to $\epsilon = 10^{-5}$) variance 1. The learned **gain** $g$ and
**bias** $b$ (init 1 and 0) let the network undo the normalisation per feature if it wants.
Note $r$ is the reciprocal of the standard deviation. The code caches `self.r` and
`self.xhat`, which is all the backward pass needs.

### Backward, step 1: through the affine part

$y_i = g_i\hat x_i + b_i$ is element-wise, so with upstream $dy$:
$$
d\hat x_i = dy_i\, g_i, \qquad dg_i \mathrel{+}= \sum_{\text{positions}} dy_i\,\hat x_i, \qquad db_i \mathrel{+}= \sum_{\text{positions}} dy_i .
$$
$g$ and $b$ are shared across all $B\cdot T$ positions, so their gradients sum over positions,
the fan-out rule again. That's `_sum_to_last`, which reshapes to $(-1, C)$ and sums axis 0.

### Backward, step 2: through the normalisation (the hard part)

Every $\hat x_i$ depends on *every* $x_j$, through $\mu$ and through $r$. So the local Jacobian
$\partial\hat x_i/\partial x_j$ is a dense $C\times C$ matrix. Build it one piece at a time.

**(a) Derivative of the mean.**
$$\frac{\partial\mu}{\partial x_j} = \frac1C .$$

**(b) Derivative of the centred value.** Write $c_i = x_i - \mu$:
$$\frac{\partial c_i}{\partial x_j} = \delta_{ij} - \frac1C .$$

**(c) Derivative of the variance.** $\text{var} = \frac1C\sum_k c_k^2$, so
$$
\frac{\partial\,\text{var}}{\partial x_j} = \frac1C\sum_k 2c_k\Big(\delta_{kj} - \frac1C\Big)
= \frac2C c_j - \frac{2}{C^2}\underbrace{\sum_k c_k}_{=\,0} = \frac2C c_j .
$$
The centred values sum to zero, so the term coming through $\mu$ vanishes.

**(d) Derivative of $r = (\text{var}+\epsilon)^{-1/2}$.**
$$
\frac{\partial r}{\partial x_j} = -\tfrac12(\text{var}+\epsilon)^{-3/2}\cdot\frac2C c_j = -\frac{r^3}{C}\,c_j .
$$

**(e) Assemble, using the product rule on $\hat x_i = c_i\,r$:**
$$
\frac{\partial\hat x_i}{\partial x_j}
= r\Big(\delta_{ij}-\frac1C\Big) + c_i\Big(-\frac{r^3}{C}c_j\Big)
= r\Big(\delta_{ij} - \frac1C - \frac{(c_i r)(c_j r)}{C}\Big)
= \frac{r}{C}\Big(C\,\delta_{ij} - 1 - \hat x_i\hat x_j\Big).
$$
The Jacobian is symmetric, and its three pieces have clear meanings:

* $\delta_{ij}$ is the direct path, as if $\mu$ and $\sigma$ were constants;
* $-\frac1C$ is the path through $\mu$;
* $-\hat x_i\hat x_j/C$ is the path through $\sigma$.

The result is **exact, $\epsilon$ included**: we never assumed $\text{var}+\epsilon = \text{var}$,
because $c_i r = \hat x_i$ holds by definition.

**(f) The VJP.** Contract with the upstream gradient, $dx_j = \sum_i d\hat x_i\,
\partial\hat x_i/\partial x_j$:
$$
dx_j = r\Big( d\hat x_j - \frac1C\sum_i d\hat x_i - \hat x_j\,\frac1C\sum_i d\hat x_i\,\hat x_i \Big)
$$
$$
\boxed{\;dx = r\odot\Big(d\hat x - \operatorname{mean}(d\hat x) - \hat x\odot\operatorname{mean}(d\hat x\odot\hat x)\Big)\;}
$$
where `mean` is over the $C$ features of each token, broadcast back. Shapes: $r$ is
$(\ldots,1)$, the two means are $(\ldots,1)$, and everything else is $(\ldots,C)$ ✓. Cost:
$O(C)$ per token instead of the $O(C^2)$ the Jacobian would need.

**A sanity property.** Sum the result over $j$: $\sum_j dx_j = r(\sum d\hat x - \sum d\hat x -
m\sum_j\hat x_j) = 0$, since $\sum_j\hat x_j = 0$ (here $m$ is the second mean). So the
gradient is orthogonal to $\mathbf 1$. That makes sense: shifting all of $x$ by a constant
doesn't change $\hat x$, so the loss can't care about that direction. Likewise
$\sum_j dx_j\hat x_j = r\,m\,(C - \sum_j\hat x_j^2) \approx 0$, since rescaling $x$ (almost) doesn't
change $\hat x$ either. You can check both numerically (exercise 3).

### Code walkthrough

| Math | `LayerNorm` code |
|---|---|
| $\mu$, $c$, var, $r$, $\hat x$ | `mu = x.mean(...)`, `xc = x - mu`, `var = (xc*xc).mean(...)`, `self.r = 1/xp.sqrt(var+eps)`, `self.xhat = xc*self.r` |
| $dg \mathrel{+}= \sum dy\odot\hat x$ | `self.weight.grad += _sum_to_last(dy * xhat)` |
| $db \mathrel{+}= \sum dy$ | `self.bias.grad += _sum_to_last(dy)` |
| $d\hat x = dy\odot g$ | `dxhat = dy * self.weight.data` |
| boxed formula | `r * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat*xhat).mean(-1, keepdims=True))` |

`bias=False` (the default in `GPTConfig`) drops $b$. The derivation is unchanged.

## 3. RMSNorm

### Forward

RMSNorm (used by Llama, and by the Kavi preset) skips the centring and has no bias
(`RMSNorm.forward`):
$$
r = \frac{1}{\sqrt{\frac1C\sum_j x_j^2 + \epsilon}},\qquad \hat x_i = x_i\,r,\qquad y_i = g_i\hat x_i .
$$
It rescales each vector to unit RMS but leaves its mean alone. It's cheaper (one reduction
instead of two) and in practice just as good. The usual reading is that LayerNorm's benefit
comes mostly from re-scaling, not re-centring.

### Backward

Same steps, minus everything involving $\mu$. Let $s = \frac1C\sum_k x_k^2$, so
$\partial s/\partial x_j = \frac2C x_j$ and
$$
\frac{\partial r}{\partial x_j} = -\tfrac12(s+\epsilon)^{-3/2}\cdot\frac2C x_j = -\frac{r^3}{C}x_j,
$$
$$
\frac{\partial\hat x_i}{\partial x_j} = r\,\delta_{ij} + x_i\Big(-\frac{r^3}{C}x_j\Big) = r\Big(\delta_{ij} - \frac{\hat x_i\hat x_j}{C}\Big).
$$
Contract with $d\hat x = dy\odot g$:
$$
\boxed{\;dx = r\odot\Big(d\hat x - \hat x\odot\operatorname{mean}(d\hat x\odot\hat x)\Big)\;}
$$
Compare the two Jacobians side by side:
$$
\text{LayerNorm: } r\big(\delta_{ij} - \tfrac1C - \tfrac{\hat x_i\hat x_j}{C}\big)
\qquad
\text{RMSNorm: } r\big(\delta_{ij} - \tfrac{\hat x_i\hat x_j}{C}\big).
$$
The $-\frac1C$ term, LayerNorm's "path through $\mu$", is gone because there is no $\mu$.
Everything else is literally the same formula (with $\hat x$ defined without centring). So RMSNorm's
backward is **LayerNorm's backward minus the mean term**, which is what
`RMSNorm.backward` implements:

```python
self.weight.grad += _sum_to_last(dy * xhat)
dxhat = dy * self.weight.data
return r * (dxhat - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
```

The gradient is still (nearly) orthogonal to $\hat x$, since RMSNorm is scale-invariant. It is
*not* orthogonal to $\mathbf 1$, since RMSNorm is not shift-invariant.

## 4. Pre-norm vs post-norm

Where does the norm go? The original Transformer (2017) used **post-norm**:
$$ x_{l+1} = \mathrm{Norm}\big(x_l + F(x_l)\big). $$
GPT-2 onward, and Kavi (`kavi/model.py::Block`), use **pre-norm**:
$$ x_{l+1} = x_l + F\big(\mathrm{Norm}(x_l)\big), $$
plus one final norm before the output head (`GPT.norm_f`), because the stream itself is never
normalised.

Look at the gradient through one block. Pre-norm:
$$ \frac{\partial x_{l+1}}{\partial x_l} = I + J_F\,J_{\text{Norm}}
\;\Rightarrow\;
dx_l = dx_{l+1} + J_{\text{Norm}}^\top J_F^\top\,dx_{l+1}. $$
That's exactly the code in `Block.backward`:

```python
dx = dy + self.norm2.backward(self.mlp.backward(dy))
return dx + self.norm1.backward(self.attn.backward(dx))
```

Unroll it across $L$ blocks and the product $\prod_l (I + A_l)$ always contains the bare
identity: $dx_0 = dx_L + (\text{branch terms})$. The loss gradient reaches the embedding
**untouched** along a clean "residual highway", however deep the stack or however badly
initialised the branches.

Post-norm instead gives $\partial x_{l+1}/\partial x_l = J_{\text{Norm}}(I + J_F)$. Every
block's norm Jacobian sits *on* the highway, so the gradient to layer 0 is multiplied by $L$
of them. Each one scales by $r = 1/\sigma$ of a stream that, as we measured, varies a lot in
scale, and each one projects out some directions. The result is gradients whose size depends
strongly on depth. Post-norm can be made to work, but it needs careful learning-rate warmup
and tuning (Xiong et al., 2020 analyse exactly this).

**Suryansh hit this himself.** In his earlier `transformer-repro` project (a from-scratch
encoder-decoder trained on Multi30k En→De), the faithful post-norm model underfit: train loss
stuck around 4.5–5.6, and one scheduled-LR run diverged. Test BLEU was about 4.8. Switching to
pre-norm was the single biggest fix, alongside gradient clipping at 1.0, dropout 0.1 and a
properly scaled warmup. It took test BLEU to **36.2**. Kavi uses pre-norm from day one.

## 5. Why no weight decay on gains (and biases)

`LayerNorm` and `RMSNorm` create their gains with `Param(..., decay=False)`, and so does every
bias. `AdamW.step` applies $\theta \leftarrow \theta - \text{lr}\cdot\lambda\theta$ only when
`p.decay` is true ([chapter 08](08-optimizers.md)). Reasons:

* **Decay pulls toward 0, but a gain's neutral value is 1.** Decaying $g$ shrinks every
  normalised activation. That doesn't make the function simpler, just smaller, and the next
  Linear layer has to grow its (decayed) weights to compensate. That fights itself.
* **Scale is exactly what the norm removed.** Weight decay regularises a matrix by limiting
  its norm. A gain vector *is* a scale, so penalising it has no "simpler model" meaning.
* **They're tiny.** $C$ numbers per norm, a negligible share of capacity, so there's nothing
  to regularise.

In the trained char model the gains stay near 1 (from 0.65 in the first block's `norm1` to
1.39 in `norm_f`). That's free to move, and it hasn't run off.

## 6. How we know it's right

* `tests/test_gradcheck.py::test_layernorm` and `test_rmsnorm` check weight, bias (LayerNorm)
  and input gradients on a $(2,3,8)$ input, **after `randomise()`**. With $g\equiv1$, a
  backward that forgets the `* self.weight.data` scores 4.5e-07 and passes. Randomised, it
  scores 1.0 ([chapter 03](03-linear-and-backprop.md) §4).
* `test_block[gpt2/kavi]` and `test_whole_model` check the norms inside the residual wiring.
* `tests/test_torch_parity.py` compares our LayerNorm against `F.layer_norm` and our RMSNorm
  against `x * rsqrt(mean(x²)+eps) * g`. This checks the *forward definition* too, e.g. that
  we use the biased variance ($1/C$, not $1/(C-1)$) and put $\epsilon$ inside the square root,
  as torch does.

## 7. Pitfalls

* **$1/(C-1)$ variance.** `np.var` defaults to $1/C$ but `torch.var` defaults to unbiased.
  Normalisation layers use $1/C$. A mismatch passes gradcheck and fails torch parity.
* **$\epsilon$ outside the root** ($1/(\sigma+\epsilon)$ instead of $1/\sqrt{\text{var}+\epsilon}$)
  changes the function, and also the derivative of $r$. Our derivation relied on
  $r = (\text{var}+\epsilon)^{-1/2}$.
* **Dropping a term that "should be small".** The $\sigma$ path is not small, it's $O(1)$.
  Delete either subtracted term and the input gradient's relative error is order 1.
* **Copying LayerNorm's formula into RMSNorm.** Including the mean term there is wrong (error
  1.0). RMSNorm has no $\mu$ to backprop through.

## 8. Try it yourself

1. **Delete the mean term** from `LayerNorm.backward` (keep the $\hat x$ term) and run
   `python -m pytest tests -q`. *Expected:* `test_layernorm` fails on `<input>` with max
   relative error 1.0 (median over elements ≈ 0.22); `weight` and `bias` still pass at
   ~1e-11, since they don't involve $dx$. The failure propagates to `test_block[gpt2]`,
   `test_whole_model[gpt2]` and both gpt2 torch-parity cases. All kavi-preset tests pass,
   because they use RMSNorm.
2. **Add the mean term to RMSNorm**, i.e. paste LayerNorm's formula into `RMSNorm.backward`.
   *Expected:* the mirror image: `test_rmsnorm` fails on `<input>` at 1.0, plus
   `test_block[kavi]`, `test_whole_model[kavi]` and both kavi parity cases.
3. **Check the orthogonality properties.** Run a randomised `LayerNorm(8)` forward/backward
   with a random upstream `G` and print `dx.sum(-1)` and `(dx * m.xhat).sum(-1)`. *Expected:*
   the first is ~1e-16 (exactly orthogonal to $\mathbf 1$). The second is small but not
   zero: it's proportional to $C - \sum\hat x^2 = C\epsilon/(\text{var}+\epsilon)$. For
   RMSNorm, `dx.sum(-1)` is *not* zero.
4. **Watch the stream grow.** Load `runs/char_kavi/best.npz` with
   `kavi.checkpoint.load_model`, push a batch of validation ids through `m.wte` and then each
   block, and print the RMS after each. *Expected:* about 0.09 → 0.66 → 1.04 → 1.53 → 2.43.
   Then print the RMS of `blk.norm1(x)` at each depth. *Expected:* close to the gain's RMS
   (≈ 1) everywhere. That's the point of the norm.
