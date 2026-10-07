# 07 — The loss: softmax cross-entropy

**What you'll learn.** How "predict the next token" becomes a classification problem, solved at
every position at once, and why cross-entropy is the right score for it. You'll derive the
famously clean gradient $\partial L/\partial z = (\mathrm{softmax}(z) - \mathrm{onehot}(y))/N$
and see why softmax and log must be computed together. And you'll learn to read a loss number:
what $\ln V$ means at step 0, what perplexity is, and how to turn the loss into bits per byte.

Prerequisites: VJPs ([chapter 03](03-linear-and-backprop.md)). The logits come from the tied
output head ([chapter 02](02-embeddings.md)).

---

## 1. Next-token prediction is classification, $B\cdot T$ times over

`TokenData.batch` (`kavi/data.py`) cuts random windows from the corpus and returns
`x = data[i : i+T]` and `y = data[i+1 : i+T+1]`: the targets are the inputs shifted by one.
For each position $t$, the model sees tokens $0..t$ (the causal mask forbids peeking,
[chapter 05](05-attention.md)) and must output a score for each of the $V$ possible next
tokens. So one batch holds $N = B\cdot T$ separate **$V$-way classification problems**. For the
char model that's $32\times128 = 4096$ hundred-way problems per step, all trained in parallel
from one forward pass.

The model's output for position $n$ is a logit vector $z_n\in\mathbb{R}^V$. Softmax turns it
into a probability distribution:
$$ p_{n,k} = \frac{e^{z_{n,k}}}{\sum_{j} e^{z_{n,j}}}. $$

## 2. Cross-entropy = negative log-likelihood

We want the model to assign high probability to what actually came next. The probability of
the whole batch of targets (treating positions as independent) is $\prod_n p_{n,y_n}$.
Maximising it is the same as minimising the average negative log:
$$ \boxed{\;L = -\frac1N\sum_{n=1}^{N} \log p_{n,y_n}\;} $$
That's the cross-entropy between the one-hot "truth" distribution and the model's $p$. Why the
log:

* It turns a product of thousands of small numbers (which would underflow to 0) into a sum.
* It punishes confident mistakes without limit: $p_y = 0.5$ costs $0.69$, while $p_y = 10^{-6}$
  costs $13.8$.
* $\log p$ measures *information*: $-\log_2 p_y$ is how many bits an ideal compressor using the
  model would spend encoding the true token. That's where bits-per-byte comes from (§6).

Dividing by $N$ makes the loss, and the gradient, independent of batch size and context
length, so the learning rate doesn't have to be retuned when they change.

## 3. Computing it stably: log-sum-exp and max-subtraction

Plug softmax into the log:
$$ \log p_{n,k} = z_{n,k} - \underbrace{\log\sum_j e^{z_{n,j}}}_{\mathrm{lse}(z_n)}. $$
Computed naively, both halves can fail.

**Overflow.** Float32 tops out at $3.4\times10^{38} = e^{88.7}$. Logits of 90, 100, 110 are
unremarkable for a confident model, but:

```python
>>> z = np.array([90., 100., 110.], dtype=np.float32)
>>> np.exp(z)                  # [inf inf inf]
>>> np.exp(z) / np.exp(z).sum()   # [nan nan nan]   inf/inf
```

**Underflow, then $\log 0$.** With logits $[0, -200]$, $e^{-200}$ underflows to exactly 0 in
float32, so the naive probability is `[1, 0]` and its log is `[0, -inf]`. One `-inf` in the
loss and training is dead. The true log-probability is a perfectly ordinary $-200$.

**The fix** rests on the fact that subtracting any constant $m$ from all logits changes nothing:
$$
\frac{e^{z_k - m}}{\sum_j e^{z_j - m}} = \frac{e^{-m}e^{z_k}}{e^{-m}\sum_j e^{z_j}} = p_k,
\qquad
\mathrm{lse}(z) = m + \mathrm{lse}(z - m).
$$
Choose $m = \max_j z_j$. Then every shifted logit is $\le 0$, so `exp` can't overflow. The
largest term is $e^0 = 1$, so the sum is $\ge 1$ and its log is never $-\infty$. Tiny terms
may still underflow to 0, but harmlessly, since they're added to something $\ge1$. With the
trick, $[90,100,110]$ with target 2 gives a loss of $4.54\times10^{-5}$, and $[0,-200]$ gives
log-probabilities of exactly $[0, -200]$.

### Code walkthrough (`kavi/loss.py::CrossEntropy.forward`)

| Math | Code |
|---|---|
| flatten to $N\times V$ | `z = logits.reshape(-1, V); y = targets.reshape(-1)` |
| $z \leftarrow z - \max_j z_j$ | `z = z - z.max(axis=1, keepdims=True)` |
| $\mathrm{lse}$ of the shifted logits | `lse = xp.log(xp.exp(z).sum(axis=1, keepdims=True))` |
| $\log p = z - \mathrm{lse}$ (log-softmax) | `logp = z - lse` |
| $p$, cached for backward | `self.probs = xp.exp(logp)` |
| $L = -\frac1N\sum_n \log p_{n,y_n}$ | `-logp[rows, y].mean()` |

The loss is read straight off the log-softmax and never goes through $\log(p)$. That's the
whole point.

## 4. The gradient: $(p - \mathrm{onehot}(y))/N$

### Derivation via log-sum-exp

Write the loss in logits only:
$$ L = -\frac1N\sum_n\Big(z_{n,y_n} - \mathrm{lse}(z_n)\Big). $$
Two derivatives are needed. First, $\partial z_{n,y_n}/\partial z_{n,k} = \delta_{k,y_n}$.
Second, the derivative of log-sum-exp is softmax:
$$ \frac{\partial\,\mathrm{lse}(z_n)}{\partial z_{n,k}} = \frac{1}{\sum_j e^{z_{n,j}}}\cdot e^{z_{n,k}} = p_{n,k}. $$
Row $n$ of the loss involves only row $n$ of the logits, so
$$
\frac{\partial L}{\partial z_{n,k}} = -\frac1N\big(\delta_{k,y_n} - p_{n,k}\big)
\quad\Longrightarrow\quad
\boxed{\;dz = \frac{p - Y}{N}\;}
$$
where $Y$ is the $N\times V$ one-hot matrix of targets. Shape $(N,V)$, the same as the logits ✓.
In words: **predicted probabilities minus the truth.** The correct class is pushed up by
$(1-p_y)/N$, and every wrong class is pushed down in proportion to how much probability it
stole.

### The same result the long way (and why fusing matters)

Treat softmax and the log as separate layers. Log layer: $L_n = -\log p_{y}$, so
$\partial L_n/\partial p_j = -\delta_{jy}/p_y$. Softmax layer: its Jacobian is
$\partial p_j/\partial z_k = p_j(\delta_{jk} - p_k)$ (differentiate the quotient). Chain them:
$$
\frac{\partial L_n}{\partial z_k} = \sum_j \Big(-\frac{\delta_{jy}}{p_y}\Big)\,p_j(\delta_{jk}-p_k) = -\frac{1}{p_y}\,p_y(\delta_{yk} - p_k) = p_k - \delta_{yk}. \checkmark
$$
The $1/p_y$ from the log cancels the $p_y$ from the softmax Jacobian. In exact arithmetic,
that is. In floating point, the separate route computes $-1/p_y$ first. When the model is
confidently wrong, $p_y$ might be $10^{-30}$, or exactly 0 after underflow, and the
intermediate gradient is $10^{30}$ or `inf`. Then it's multiplied by an equally extreme small
number, or `0 * inf = nan`. The fused formula never forms either extreme: every entry of
$p - Y$ lies in $[-1, 1]$. Fusing is also cheaper. No $V\times V$ Jacobian per row (even
implicitly), just one subtraction.

Two properties that are good to know:

* **Each row of $dz$ sums to zero** ($\sum_k p_k = 1 = \sum_k Y_k$). Adding a constant to all
  logits doesn't change the loss (§3), so the gradient has no component along $\mathbf 1$. It's
  the same shift-invariance that gave the attention key bias its zero gradient
  ([chapter 03](03-linear-and-backprop.md) §4).
* **It's bounded**: $|dz_{n,k}| \le 1/N$. The loss itself can never produce an exploding
  gradient. Explosions come from the layers below, which is what gradient clipping
  ([chapter 08](08-optimizers.md)) is for.

### Code walkthrough (`CrossEntropy.backward`)

```python
d = self.probs.copy()              # p
d[xp.arange(n), self.y] -= 1.0     # p - Y  (subtract 1 at each row's target)
d /= n                             # / N
self.probs = None                  # free the (N, V) buffer early
return d.reshape(self.shape)       # back to (B, T, V)
```

Freeing `probs` matters for BPE: with $N = 32\times256$ and $V=4096$ it's $3.4\times10^7$
floats, 134 MB in float32. `GPT.backward` then routes `dlogits` into the tied head
([chapter 02](02-embeddings.md)).

## 5. Reading a loss value

### At initialisation: $\ln V$

An untrained model whose logits are all equal predicts the uniform distribution,
$p_{y} = 1/V$, so the loss is $-\log(1/V) = \ln V$:

| tokenizer | $V$ | $\ln V$ |
|---|---|---|
| char | 100 | 4.605 |
| BPE | 4096 | 8.318 |

Kavi's init (small embeddings, tied head, [chapter 02](02-embeddings.md) §6) gives logits with
std ≈ 0.3, close to equal. The char run's first evaluation read **train 4.593 / val 4.583**,
within 0.02 of $\ln 100$. That's the check: **if your step-0 loss is far from $\ln V$, something
is wrong** (init too large, logits mis-scaled, targets misaligned).

Why not *exactly* $\ln V$? For i.i.d. logits $z\sim\mathcal N(0,\sigma^2)$,
$\sum_j e^{z_j}\approx V\,\mathbb E[e^z] = V e^{\sigma^2/2}$. So $\mathrm{lse}\approx\ln V +
\sigma^2/2$, and on random targets the loss is about $\ln V + \sigma^2/2$. That's ≈ 4.65 for
$\sigma = 0.3$. We measured 4.68 on random char targets and 8.39 on random BPE targets. The
small remainder is the tied head's initial "repeat the current token" bias
([chapter 02](02-embeddings.md) §5). On real text, the frequent patterns can nudge the number
slightly below $\ln V$ instead.

### Perplexity $= e^{L}$

$e^{L}$ is the **perplexity**: the model is as uncertain as if it were choosing uniformly among
$e^L$ options at every step. At init it's $e^{\ln V} = V$, a uniform guess over the vocabulary.
The char model's best validation loss so far, 1.597, is a perplexity of **4.94**: about five
equally plausible next characters instead of 100. Perplexity, like the loss, depends on the
tokenizer. A BPE token carries about 3 characters of text, so its perplexity naturally sits
much higher.

### Bits per byte

To compare char-level and BPE models fairly, Kavi reports **bits per byte**
(`TokenData.bpb`, [chapter 01](01-tokenization.md) and [chapter 09](09-training.md)):
$$ \text{bpb} = \frac{L/\ln 2}{\text{bytes per token}}. $$
$L/\ln2$ converts nats to bits per token, and dividing by bytes per token spreads them over the
UTF-8 bytes that token stands for. On the validation split, char tokens average 1.013 bytes and
BPE tokens 3.025 bytes. Check against the log: val loss 4.583 → $4.583/0.6931/1.013 = 6.53$
bpb, exactly the `val bpb 6.5289` printed at step 0. A uniform guess is 6.56 bpb for char but
only 3.97 for BPE. Bigger tokens already "know" common character sequences before any training.

## 6. How we know it's right

There is no stand-alone CrossEntropy gradcheck. The loss is tested where it lives, at the top of
the full model:

* `tests/test_gradcheck.py::test_whole_model[gpt2/kavi]` uses the **true loss** as the
  objective (not $\sum \text{out}\odot G$), so every parameter's gradient passes through
  `CrossEntropy.backward` first. Any error in $dz$ shows up everywhere.
* `tests/test_torch_parity.py::test_model_matches_torch` compares our loss to
  `F.cross_entropy` to $10^{-10}$ and every gradient to `rtol=1e-7`. That checks the *mean*
  reduction and the log-softmax definition independently.

## 7. Pitfalls

* **Separate softmax then `log`**: `-inf` and NaN for confident mistakes (§3, §4).
* **Forgetting $/N$** (or using sum instead of mean). Every gradient is $N$ times too large.
  Gradcheck catches it instantly, and with Adam it's partly hidden in training, since Adam
  normalises gradient scale. Silent bugs are the worst kind.
* **Off-by-one targets.** If `y` isn't `x` shifted by one, the model learns to copy its input.
  The loss collapses toward 0 suspiciously fast, but the samples are garbage.
* **Fancy-index assignment on repeated rows** is not a problem *here*: `d[arange(n), y] -= 1`
  touches each row exactly once, so no index repeats (contrast the embedding,
  [chapter 02](02-embeddings.md) §3).

## 8. Try it yourself

1. **Drop the $1/N$.** Delete `d /= n` from `CrossEntropy.backward` and run
   `python -m pytest tests -q`. *Expected:* both `test_whole_model` cases fail with relative
   error **0.846 on every parameter**, and all four torch-parity cases fail. Why that exact
   number: the test uses $N = 2\times6 = 12$ positions, so analytic $= 12\times$ numeric, and
   $|12a - a|/(12a + a) = 11/13 = 0.846$.
2. **See the overflow.** Compute softmax naively on
   `np.array([90., 100., 110.], dtype=np.float32)`, then call
   `CrossEntropy().forward(z[None], np.array([2]))`. *Expected:* `[nan nan nan]` vs a finite
   loss of $4.5\times10^{-5}$.
3. **Predict the initial loss.** Build the BPE-config kavi model, measure the logit std
   $\sigma$ on a random batch, and predict $\ln V + \sigma^2/2$ before measuring the loss on
   random targets. *Expected:* $\sigma\approx0.32$, prediction ≈ 8.37, measured ≈ 8.39.
4. **Gradient rows sum to zero.** After any `forward`/`backward`, check
   `abs(d.sum(-1)).max()` on the returned `dlogits`. *Expected:* ~1e-8 in float32 and ~1e-17
   in float64. Then explain why from §4.
