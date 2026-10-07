# Kavi — a GPT built from absolute scratch in NumPy

> *Kavi* (कवि) — Sanskrit/Hindi for "poet". Its first job is to learn to write like Shakespeare.

Inspired by Green Code's "Anton" series (a GPT in pure NumPy, trained on Shakespeare, later
scaled with PyTorch on GPUs). Kavi follows the same arc but is built to be **better and more
rigorous**, and to be a **learning project**: every layer's forward *and* backward pass is
derived by hand, written down, and proven correct.

Constraints: free compute only (this laptop's CPU + free Kaggle GPUs). No PyTorch in the
model — PyTorch is used only as an *optional test oracle* to cross-check our gradients.

---

## 1. What makes Kavi better than the original ("the edge")

| # | Edge | Anton (original) | Kavi |
|---|------|------------------|------|
| 1 | **Proven gradients** | "I double-checked my gradient code" | Every layer *and* the whole model are verified two ways: finite-difference gradcheck (float64) **and** parity against PyTorch autograd. Tests, not vibes. |
| 2 | **Modern architecture** | GPT-2 style: learned positions, LayerNorm, ReLU MLP | Switchable: GPT-2 baseline **vs** a Llama-style "Kavi" block — **RoPE** rotary positions, **RMSNorm**, **SwiGLU** MLP — all with hand-derived backward passes. |
| 3 | **Honest evaluation** | Eyeballs the samples, no validation set, 6M params on 319k tokens | Held-out validation split, **bits-per-byte** (a metric that is comparable across tokenizers, so char-level vs BPE is a fair fight), loss curves, overfitting tracked. |
| 4 | **Ablations, not guesses** | "train a bunch of Antons and see" | Each architectural upgrade is a single-knob config diff, trained on identical budgets, reported as a table. |
| 5 | **Same NumPy code on a free GPU** | NumPy = CPU only (11 h run); rewrites in PyTorch to use a GPU | A backend switch (`numpy` ↔ `cupy`) runs the *same hand-written model* on a Kaggle GPU. |
| 6 | **Fast generation** | Re-runs the whole sequence for every token | **KV-cache** incremental decoding + temperature / top-k / top-p sampling. |
| 7 | **Flash attention, understood** | Planned for ep. 3 (PyTorch) | Tiled online-softmax attention implemented in NumPy and proven equal to naive attention — the algorithm, without the CUDA. |
| 8 | **Interpretability** | — | Attention-pattern dumps & plots: see what each head looks at. |
| 9 | **Documentation** | Videos | A `docs/` book: one chapter per component, math derivation → code, plus a lab journal of every run. |

---

## 2. Architecture

```
text ──tokenizer──▶ ids (B,T)
ids ──Embedding (wte)──▶ x (B,T,C)            [+ learned wpe  if pos="learned"]
repeat n_layer times (pre-norm residual block):
    x = x + Attention( Norm(x) )              causal, multi-head, [RoPE on q,k if pos="rope"]
    x = x + MLP( Norm(x) )                    GELU 4C  |  SwiGLU (8/3)C  (equal params)
x = Norm(x)
logits = x @ wteᵀ                             (weight tying)
loss = CrossEntropy(logits, ids shifted by 1)
```
`Norm` = LayerNorm | RMSNorm. Dropout after attention/MLP outputs and embeddings.

Every module implements `forward(x)` (caches what it needs) and `backward(dout) -> dx`
(accumulates parameter grads). No autograd anywhere.

---

## 3. Phases

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| **0. Foundations** | Repo, backend switch, Module/Param base, gradcheck harness, data download + 90/10 split | `pytest` green, data prepared |
| **1. Lego bricks** | Linear, Embedding, LayerNorm, RMSNorm, GELU, SwiGLU-MLP, GELU-MLP, causal MHA (+RoPE), Dropout, softmax-CE | Every brick passes finite-diff gradcheck (rel err < 1e-6, float64) and torch parity |
| **2. Assemble** | `GPT` model, untrained generation (gibberish milestone) | Whole-model gradcheck passes; generates |
| **3. Train** | SGD vs AdamW, warmup+cosine LR, grad clipping, overfit-one-batch test, char-level Shakespeare run on CPU | Overfit test hits ~0 loss; val bpb drops; coherent-ish samples |
| **4. BPE** | Byte-level BPE trained from scratch (incremental pair-count algorithm), encode/decode round-trip | Round-trip exact on full corpus; retrain, compare bpb vs char-level |
| **5. GPU** | CuPy backend, Kaggle kernel bundler, push/poll/download loop | Same model trains on Kaggle GPU; ablation sweep run there |
| **6. Ablations** | GPT-2 baseline vs +RoPE vs +RMSNorm vs +SwiGLU vs full Kavi | Table in `docs/journal.md` |
| **7. Inference & insight** | KV-cache, top-k/top-p, flash attention (NumPy), attention visualisation | KV-cache output == full recompute; flash == naive |
| **8. (stretch)** | PyTorch port + bigger corpus (e.g. TinyStories / FineWeb-Edu sample) on Kaggle | — |

## 4. Data

Project Gutenberg eBook #100, *The Complete Works of William Shakespeare* (public domain,
~5.4 MB — about 5× the "Tiny Shakespeare" file). Gutenberg header/licence stripped.
**Split:** text cut into 100 contiguous shards; every 10th shard is validation. That keeps
val text unseen while still covering comedies, tragedies, histories and sonnets.

## 5. Verification strategy

1. **Finite differences** — `gradcheck(module)` perturbs every parameter/input element (or a
   random sample of them) by ±h in float64 and compares to the analytic gradient.
2. **Torch parity** (optional; skipped if torch absent) — copy weights into an equivalent
   torch module, compare outputs and grads.
3. **Overfit one batch** — the full training stack must drive loss on one batch to ~0.
4. **Equivalence tests** — KV-cache vs recompute, flash vs naive attention, BPE round-trip.

## 6. Layout

```
kavi/            the library (backend, modules, model, optim, tokenizers, sampling)
scripts/         prepare_data, train_bpe, train, generate, plot
configs/         JSON run configs (one knob changed per ablation)
tests/           pytest suite (gradchecks, parity, equivalence)
kaggle/          kernel bundler + metadata
docs/            the book (one chapter per component) + journal.md
runs/            checkpoints/logs (git-ignored)
```
