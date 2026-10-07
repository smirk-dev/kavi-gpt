# 00 · Overview: the whole machine on one page

> **What you'll learn.** How a string of Shakespeare becomes a single number (the loss), and
> how that number is turned back into a gradient for every one of Kavi's ~6 million weights,
> with the tensor shape written down at each step. You'll also see the small `Module`/`Param`
> design that makes hand-written backprop manageable, how the `gpt2` and `kavi` presets
> differ, where every file lives, and how to run the whole pipeline end to end.

Kavi is a GPT written in NumPy with no autograd. Every layer has a `forward` that you could
find in any textbook and a `backward` that was derived by hand (and then proven correct by
tests). This chapter is the map. Later chapters zoom in on one brick each.

---

## 1. The intuition in one paragraph

A language model is a next-token guesser. Show it "To be, or not to" and it should put a lot
of probability on " be". Training means showing it millions of such prefixes, measuring how
surprised it was by the true next token (the **cross-entropy loss**), and nudging every weight
in the direction that makes it a little less surprised next time. The nudge direction is the
gradient, and computing the gradient is what backpropagation does: walk the computation
backwards and apply the chain rule at each step.

A transformer is just a particular choice of computation: turn each token into a vector, let
the vectors **talk to each other** (attention), let each vector **think on its own** (the
MLP), repeat a few times, then read a probability distribution over the vocabulary off the
final vectors.

---

## 2. Notation

| Symbol | Meaning | `char_kavi.json` | `bpe_base.json` |
|---|---|---|---|
| $B$ | batch size (sequences per step) | 32 | 32 |
| $T$ | sequence length (≤ `block_size`) | 128 | 256 |
| $C$ | model width, `n_embd` | 128 | 256 |
| $H$ | attention heads, `n_head` | 4 | 8 |
| $D$ | head width, $C/H$ | 32 | 32 |
| $V$ | vocabulary size | 100 | 4096 |
| $L$ | number of blocks, `n_layer` | 4 | 6 |
| $F$ | MLP hidden width | 344 (SwiGLU) | 680 (SwiGLU) or 1024 (GELU) |

$F$ is $4C$ for the GELU/ReLU MLP and `swiglu_hidden(C)` $\approx 8C/3$ (rounded to a multiple
of 8) for SwiGLU, chosen so both MLPs have almost exactly the same parameter count
([06-mlp](06-mlp.md)).

---

## 3. The data flow, with shapes

```
 "ROMEO: But soft!"                                        raw text
        │  tokenizer.encode            (01-tokenization)
        ▼
 ids            (B, T)   int64        e.g. [2061, 58, 885, 2527, 33]
        │  wte = Embedding(V, C)       (02-embeddings)
        ▼
 x              (B, T, C)             + wpe[0:T]  (1, T, C)   only if pos="learned"
        │  Dropout
        ▼
 ┌──────────────── Block × L  (pre-norm residual) ─────────────────────────┐
 │  a  = Norm1(x)                      (B, T, C)       (04-normalization)   │
 │  qkv = a @ W_qkv                    (B, T, 3C)                           │
 │  q,k,v  split + reshape             (B, H, T, D) each                    │
 │        [RoPE rotates q, k           if pos="rope"]                       │
 │  scores = q kᵀ / √D  + causal mask  (B, H, T, T)    (05-attention)       │
 │  probs  = softmax(scores)           (B, H, T, T)                         │
 │  out    = probs @ v                 (B, H, T, D) → merge → (B, T, C)     │
 │  x = x + Dropout(out @ W_proj)      (B, T, C)       ← residual add       │
 │                                                                         │
 │  m  = Norm2(x)                      (B, T, C)                            │
 │  MLP: (B,T,C) → (B,T,F) → (B,T,C)   GELU 4C or SwiGLU 8C/3  (06-mlp)     │
 │  x = x + Dropout(MLP(m))            (B, T, C)       ← residual add       │
 └─────────────────────────────────────────────────────────────────────────┘
        │  norm_f
        ▼
 h              (B, T, C)
        │  h @ wteᵀ                    tied head: the embedding table reused
        ▼
 logits         (B, T, V)
        │  CrossEntropy vs targets (B, T) = ids shifted left by one   (07-loss)
        ▼
 loss           scalar  (mean over the B·T positions)
```

Three details worth noticing:

1. **Every brick works on the last axis.** Linear, the norms and the activations all see
   `(..., C)` and don't care how many leading axes there are. Only attention needs to know
   about $T$, because it is the only place where positions interact.
2. **The causal mask** sets every score where the key position is later than the query
   position to $-\infty$ before the softmax, so position $t$ can only use tokens $0..t$. That
   is what makes "predict the next token" a fair task: the answer is never visible.
3. **The head has no weights of its own.** `GPT.forward` computes
   `logits = h @ self.wte.weight.data.T`. This is **weight tying**: the matrix that turns a
   token id into a vector is reused, transposed, to turn a vector back into token scores. It
   saves $V \cdot C$ parameters (1,048,576 for the BPE model, about 18% of it) and tends to
   help on small data.

At initialisation every logit is close to zero, so the predicted distribution is close to
uniform and the loss starts near $\ln V$: $\ln 100 = 4.61$ for the char model, $\ln 4096 =
8.32$ for BPE. The real CPU char run logged 4.58 at step 0, and the Kaggle BPE probes logged
8.30 to 8.35. If your step-0 loss is far from $\ln V$, something is wrong with the init or
the loss.

### Parameter counts

| Config | `gpt2` preset | `kavi` preset |
|---|---|---|
| `char_kavi.json` (V=100, C=128, L=4, T=128) | 816,768 | 804,480 |
| `bpe_base.json` (V=4096, C=256, L=6, T=256) | 5,836,032 | 5,758,208 |

The BPE-model gap of 77,824 is the learned position table ($256 \times 256 = 65{,}536$) plus
the small rounding difference between $3CF$ (SwiGLU) and $8C^2$ (GELU): $2{,}048$ per layer
× 6. Anton was also about 6M parameters, so the BPE model is the like-for-like comparison.

---

## 4. Backward: the same picture, read right to left

`GPT.backward()` walks the diagram upside down:

```
dlogits (B,T,V) = (softmax − onehot)/N       CrossEntropy.backward
  ├─► wte.grad += dlogitsᵀ · h               the head's use of wte
  └─► dh = dlogits @ wte  (B,T,C) ─► norm_f.backward
          ─► Block_L.backward ─► … ─► Block_1.backward
          ─► Dropout.backward
          ├─► wpe.backward( sum over batch )  only if pos="learned"
          └─► wte.backward(dx)  (scatter-add)  the input's use of wte
```

Inside a block the residual connections make backward easy to write. Because
$y = x + f(x)$ has $\partial y / \partial x = I + f'(x)$, the incoming gradient is copied to
both the skip path and the branch, and the two contributions are added (`Block.backward`):

```python
dx = dy + self.norm2.backward(self.mlp.backward(dy))
return dx + self.norm1.backward(self.attn.backward(dx))
```

The skip path is a straight highway from the loss to every layer. That is why deep stacks
train at all ([03-linear-and-backprop](03-linear-and-backprop.md)).

---

## 5. The Module / Param design (`kavi/module.py`)

There is no autograd, so every brick follows one contract:

* **`forward(x)`** computes the output and **caches** whatever its backward will need
  (`Linear` keeps `self.x`, `LayerNorm` keeps `self.xhat` and `self.r`, attention keeps
  `q, k, v, probs`).
* **`backward(dout) -> dx`** receives $\partial L / \partial \text{output}$, **adds**
  $\partial L / \partial \theta$ into each parameter's `grad`, and returns
  $\partial L / \partial \text{input}$ for the module below.

A `Param` is just `data` + `grad` + a `decay` flag (AdamW decays matrices but not gains and
biases, [08-optimizers](08-optimizers.md)). `Module` gives you traversal for free:
`named_params()` walks attributes, lists and tuples to find every `Param`, and also provides
`zero_grad`, `train`/`eval`, `num_params` and `state_dict`/`load_state_dict`.

### Why backward accumulates with `+=`

`Linear.backward` does `self.weight.grad += x2.T @ dy2`, not `=`. The reason is **weight
tying**. The table `wte.weight` is used twice in one forward pass, once as the input lookup
and once as the output head. By the multivariable chain rule, its total gradient is the
**sum** of the gradients from each use:

$$
\frac{\partial L}{\partial W_{te}} =
\underbrace{\frac{\partial L}{\partial W_{te}}\Big|_{\text{head}}}_{\texttt{GPT.backward}}
+ \underbrace{\frac{\partial L}{\partial W_{te}}\Big|_{\text{lookup}}}_{\texttt{Embedding.backward}}
$$

If either site overwrote the buffer, one contribution would be lost without any error. With
`+=` each site simply adds its share, and the price is one rule: **call `model.zero_grad()`
once per optimisation step**, before the forward pass. `scripts/train.py` does this. The same
property would also allow gradient accumulation over several micro-batches (call
forward/backward several times, then step), though `train.py` doesn't use it.

`named_params()` also deduplicates by `id()`, so a parameter shared between modules is only
counted, decayed and checkpointed once.

### Consequences of caching

* One forward must be followed by its own backward. A second forward (for example an eval
  batch) overwrites the caches. `train.py` always runs forward → backward → step back to
  back, so this never bites, but it would if you interleaved them.
* The caches hold every activation of the batch. The single biggest one is the
  cross-entropy's `(B·T, V)` probability matrix: $32 \cdot 256 \cdot 4096$ floats ≈ 128 MiB in
  float32 for the BPE model. `CrossEntropy.backward` frees it as soon as it is used.

---

## 6. Two presets, three knobs

`GPTConfig.preset(name, **overrides)` in `kavi/model.py`:

| Knob | `gpt2` (Anton-style baseline) | `kavi` (Llama-style) | Chapter |
|---|---|---|---|
| `pos` | `"learned"`: add a trained `wpe` row per position | `"rope"`: rotate $q, k$ by position-dependent angles, no params | [02](02-embeddings.md), [05](05-attention.md) |
| `norm` | `"layernorm"`: centre, scale, gain (+bias) | `"rmsnorm"`: scale only, gain | [04](04-normalization.md) |
| `mlp` | `"gelu"`: $C \to 4C \to C$ | `"swiglu"`: gated, $C \to 8C/3 \to C$ | [06](06-mlp.md) |

Everything else (width, depth, heads, dropout, `bias`) is shared and set by the run config.
Because the knobs are independent, the ablation sweep can flip **one at a time**
(`gpt2+rope`, `gpt2+rmsnorm`, `gpt2+swiglu`) and measure what each is worth. Anton's own
architecture is `gpt2` with `mlp="relu"` and `bias=true`, which is the `anton-*` entry in
`configs/sweep_ablation.json`.

Two more config fields exist: `attn="flash"` swaps in tiled online-softmax attention
([12-flash-attention](12-flash-attention.md)), and `rope_base` sets RoPE's frequency base.

---

## 7. Repo map

```
kavi-gpt/
├── PLAN.md                    goals, the "edge" over Anton, phases, verification strategy
├── kavi/                      the library
│   ├── __init__.py            exports GPT, GPTConfig, backend
│   ├── backend.py             numpy ↔ cupy switch, dtype, host-side init, scatter_add, RNG
│   ├── module.py              Param + Module base: traversal, zero_grad, train/eval, state_dict
│   ├── layers.py              Linear, Embedding, LayerNorm, RMSNorm, ReLU, GELU, Dropout, MLP, SwiGLU
│   ├── attention.py           softmax, RoPE, CausalSelfAttention (+ KV cache)
│   ├── flash.py               tiled online-softmax attention, forward + backward
│   ├── loss.py                fused softmax + cross-entropy
│   ├── model.py               GPTConfig (+ presets), Block, GPT, generate(), sample()
│   ├── optim.py               SGD, AdamW, warmup+cosine lr_at, clip_grad_norm
│   ├── data.py                TokenData (loads the .bin token files), batching, bits-per-byte
│   ├── checkpoint.py          .npz save (atomic) / load / rebuild model from checkpoint
│   ├── gradcheck.py           finite-difference gradient checker
│   └── tokenizers/
│       ├── char.py            one token per character
│       └── bpe.py             byte-level BPE: split regex, incremental trainer, encode/decode
├── scripts/
│   ├── prepare_data.py        download Gutenberg #100, strip licence, 100-shard train/val split
│   ├── tokenize_data.py       train a tokenizer on train, write data/<name>/*.bin + meta.json
│   ├── train.py               the training loop (09-training)
│   ├── sweep.py               run many single-knob variants of one base config
│   ├── summarize.py           runs/*/log.jsonl → markdown table + val-bpb plot
│   ├── generate.py            sample text from a checkpoint; --bench KV cache vs none
│   ├── attention_viz.py       heatmaps + per-head distance/entropy (13-interpretability)
│   └── backend_parity.py      prove CPU and GPU compute identical loss and grads
├── configs/
│   ├── char_kavi.json         small char-level model for the CPU
│   ├── bpe_base.json          ~5.8M-param BPE model (the Anton-sized one)
│   ├── sweep_probe.json       2 short runs, a smoke test for the GPU pipeline
│   └── sweep_ablation.json    the full ablation: anton, gpt2, +rope, +rmsnorm, +swiglu, kavi, char, sgd
├── tests/
│   ├── conftest.py            puts every test in float64 by default
│   ├── test_gradcheck.py      finite-difference check of every brick and the whole model
│   ├── test_torch_parity.py   same forward/backward re-done in PyTorch (skipped if no torch)
│   ├── test_training.py       overfit-one-batch, SGD-vs-AdamW
│   ├── test_inference.py      KV cache == full recompute, sampling filters
│   ├── test_flash.py          flash == naive attention, flash gradcheck
│   └── test_bpe.py            lossless split, round trip, incremental == naive trainer
├── kaggle/
│   ├── build_kernel.py        bundle the repo into one Kaggle script; push/status/output
│   ├── build/                 generated kernel + metadata (git-ignored)
│   └── output/                downloaded kernel results (git-ignored)
├── data/                      raw + split text and tokenised .bin files (git-ignored)
├── runs/                      one folder per training run (git-ignored)
└── docs/                      this book
```

---

## 8. Reading order

The chapters follow the data flow, so reading them in number order works. If you have never
derived a backward pass, read [03](03-linear-and-backprop.md) first, because everything else
uses its pattern.

1. [00-overview](00-overview.md): this map
2. [01-tokenization](01-tokenization.md): text → ids, char vs byte-level BPE, bits-per-byte
3. [02-embeddings](02-embeddings.md): ids → vectors, learned positions, tying
4. [03-linear-and-backprop](03-linear-and-backprop.md): the chain rule as code; gradcheck
5. [04-normalization](04-normalization.md): LayerNorm vs RMSNorm, their backward passes
6. [05-attention](05-attention.md): causal multi-head attention, RoPE, KV cache
7. [06-mlp](06-mlp.md): GELU MLP vs SwiGLU
8. [07-loss](07-loss.md): fused softmax cross-entropy and its clean gradient
9. [08-optimizers](08-optimizers.md): SGD, AdamW, warmup + cosine, clipping
10. [09-training](09-training.md): the loop, evaluation, checkpoints, sweeps
11. [10-sampling](10-sampling.md): temperature, top-k, top-p, KV-cached generation
12. [11-gpu](11-gpu.md): the same NumPy code on CuPy, and the Kaggle pipeline
13. [12-flash-attention](12-flash-attention.md): online softmax, tiled, in NumPy
14. [13-interpretability](13-interpretability.md): what the heads look at
15. [journal](journal.md): every run, with numbers

---

## 9. Quickstart: run everything

All commands run from the repo root. You need Python 3.10+ with NumPy. matplotlib is used
for plots, PyTorch only by the optional parity tests, and CuPy only on a GPU.

```bash
# 1. data: download, strip the Gutenberg licence, make the 90/10 interleaved split
python scripts/prepare_data.py           # -> data/shakespeare.txt, train.txt, val.txt

# 2. tokenise (each writes data/<name>/{train.bin,val.bin,meta.json,tokenizer.json})
python scripts/tokenize_data.py char             # -> data/char     (vocab 100)
python scripts/tokenize_data.py bpe --vocab 4096 # -> data/bpe4096  (about 10 s)

# 3. prove the maths before spending compute on it
python -m pytest -q

# 4. train (outputs runs/<name>/: config.json, log.jsonl, best.npz, last.npz, samples.txt)
python scripts/train.py configs/char_kavi.json
python scripts/train.py configs/char_kavi.json --set train.max_steps=200 model.n_layer=2
python scripts/train.py configs/char_kavi.json --resume          # continue from last.npz

# 5. sample from the best checkpoint
python scripts/generate.py runs/char_kavi/best.npz --prompt "ROMEO:" --tokens 300
python scripts/generate.py runs/char_kavi/best.npz --bench       # KV cache speed-up

# 6. look inside
python scripts/attention_viz.py runs/char_kavi/best.npz --out docs/img/attention.png

# 7. ablations: many runs, then one table
python scripts/sweep.py configs/sweep_probe.json
python scripts/summarize.py runs/ --plot docs/img/curves.png

# 8. the same thing on a free Kaggle GPU (11-gpu)
python kaggle/build_kernel.py build --sweep configs/sweep_ablation.json
python kaggle/build_kernel.py push
python kaggle/build_kernel.py status
python kaggle/build_kernel.py output      # -> kaggle/output/
```

The run name defaults to the config file's stem (`char_kavi`), so a run lives in
`runs/char_kavi/`. Override it with `--set name="my-run"`.

---

## 10. How we know it's right

Kavi's main claim over Anton is that its gradients are *proven*, not just believed. The
proof has four layers, cheapest first:

| Test | What it proves |
|---|---|
| `tests/test_gradcheck.py` (`test_linear` … `test_whole_model`) | every backward matches finite differences of its own forward, in float64 |
| `tests/test_torch_parity.py::test_model_matches_torch` | our forward *and* gradients match PyTorch's definitions and autograd |
| `tests/test_training.py::test_adamw_overfits_one_batch` | the full stack (model + loss + optimiser + clipping) can memorise a batch |
| `tests/test_inference.py`, `tests/test_flash.py`, `tests/test_bpe.py` | the fast paths (KV cache, flash attention, incremental BPE) equal the slow, obviously correct ones |

## 11. Pitfalls

* **Forgetting `zero_grad()`** doesn't crash anything. Each `grad` silently becomes a
  running sum of all past gradients. With clipping and Adam, training can even look like
  it works, just worse ([09-training, exercise 2](09-training.md#12-try-it-yourself)).
* **Comparing loss across tokenizers.** A loss of 1.6 per character and 4.5 per BPE token
  can't be compared directly. Use bits-per-byte ([01](01-tokenization.md#7-bits-per-byte-the-fair-metric)).
* **Building the model before `set_backend`.** Parameters are allocated with the active
  backend when the `GPT` is built. Switch the backend first ([11-gpu](11-gpu.md)).

## 12. Try it yourself

1. **Check $\ln V$.** Build `GPT(GPTConfig.preset("kavi", vocab_size=4096))`, feed a random
   `(2, 16)` batch with random targets, and print the loss. *Expected:* about 8.3
   ($\ln 4096 = 8.318$).
2. **Count the tie.** Print `model.num_params()` for the BPE config, then work out by hand
   what it would be without tying. *Expected:* 5,758,208 tied, and $5{,}758{,}208 + 4096
   \cdot 256 = 6{,}806{,}784$ untied.
3. **Break the accumulation.** In a scratch copy, add `self.weight.grad.fill(0)` as the
   first line of `Embedding.backward`, so the embedding overwrites the buffer instead of
   adding to it. Then run `pytest tests/test_gradcheck.py -k "embedding or whole_model"`.
   *Expected:* `test_embedding` still passes, because a lone embedding has only one use, but
   `test_whole_model` fails on `wte.weight`: the head's contribution, written a moment
   earlier in `GPT.backward`, is wiped out. (Revert afterwards.)
4. **Flip one knob.** Train `char_kavi.json` for 300 steps twice, once as is and once with
   `--set model.preset="gpt2"`, and compare `val_bpb` in the two `log.jsonl` files.
   *Expected:* both numbers are close, with the size and sign of the gap left for you to
   measure (the BPE probe on Kaggle favoured `kavi`, 2.156 vs 2.335 bpb).
