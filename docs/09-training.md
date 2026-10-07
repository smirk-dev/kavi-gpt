# 09 · Training: the loop that turns gradients into a model

> **What you'll learn.** How `scripts/train.py` turns a tokenised corpus into a trained
> checkpoint: how the data is split and batched, how evaluation is kept honest, how
> overfitting is detected and contained, and how runs are checkpointed, resumed, swept and
> summarised. You'll also learn the two cheap tests that prove the training stack is wired
> correctly, and how to read a loss curve.

Everything up to this chapter computes one gradient. Training is the loop around it:
*sample a batch → forward → loss → backward → clip → step*, repeated thousands of times, with
enough bookkeeping to know whether it's working.

---

## 1. The data split

`scripts/prepare_data.py` cuts the cleaned corpus into **100 contiguous shards**, each ending
at a line break, and puts **every 10th shard** (indices 9, 19, …, 99) into validation:

```python
pos = text.find("\n", max(len(text) * i // n_shards, pos)) + 1   # cut just after a newline
val   = "".join(s for i, s in enumerate(shards) if i % 10 == 9)
train = "".join(s for i, s in enumerate(shards) if i % 10 != 9)
```

Each shard is about 54,000 characters, roughly a few scenes. Why this shape?

* **Not the last 10%.** The Complete Works isn't uniform: sonnets, comedies, histories,
  tragedies and the long poems come in blocks. Holding out the final 10% would make val a
  different *genre* from train, and val loss would partly measure genre shift.
* **Not random lines.** Neighbouring lines are strongly correlated (same speakers, same
  scene, repeated refrains). With a random line-level split, almost every val line would
  have its neighbours in train and val loss would be optimistic. Contiguous 54k-character
  shards keep val text genuinely unseen, with leakage limited to the 20 shard edges.
* **Interleaved.** Ten val shards spread evenly through the book give val the same mix of
  genres as train.

Result: 4,823,350 train characters, 535,993 val characters (10.0%). Anton had no validation
set at all, so he could not tell learning from memorising. Kavi's every evaluation reports
both.

---

## 2. Batching: random windows, targets shifted by one

`TokenData.batch` in `kavi/data.py`:

```python
ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
x = np.stack([data[i:i + block_size]         for i in ix])    # (B, T)
y = np.stack([data[i + 1:i + block_size + 1] for i in ix])    # (B, T), shifted by one
```

Each row is a random window of $T$ consecutive tokens, and `y` is the same window shifted
one token to the right. So `y[b, t]` is the token that follows `x[b, 0..t]`.

**One window gives $T$ training examples, not one.** Because of the causal mask
([05-attention](05-attention.md)), position $t$ only sees tokens $0..t$, so the model makes
$T$ independent next-token predictions per window, all in one forward pass:

```
x:  T  o  ␣  b  e  ,
y:  o  ␣  b  e  ,  ␣        predict y[t] from x[0..t]
```

A batch is therefore $B \cdot T$ predictions: $32 \cdot 256 = 8192$ for `bpe_base.json`, and
the loss is their mean. Random windows (instead of walking through the data in order) give
roughly independent batches, and every offset of every token appears in some window.

How much data does a run see? Steps × $B$ × $T$ token predictions:

| Config | Train tokens | Predictions per run | ≈ epochs |
|---|---|---|---|
| `char_kavi.json` (3000 steps, B=32, T=128) | 4,823,350 | 12.3 M | 2.5 |
| `bpe_base.json` (4000 steps, B=32, T=256) | 1,578,753 | 32.8 M | **20.8** |

The BPE model goes over its data about 21 times. That's why its config uses `dropout=0.2`
where the char config uses 0.1 (§5).

---

## 3. `scripts/train.py`, top to bottom

**Config.** `load_config` reads a JSON file (`data`, `model`, `train`), fills `train` from
`DEFAULT_TRAIN`, and applies `--set key.sub=value` overrides. Values are parsed as JSON when
possible (`train.lr=3e-4` becomes a float, `model.bias=true` a bool) and kept as raw strings
otherwise, so `model.pos="rope"` and `model.pos=rope` both work. `name` defaults to the
config file's stem.

**Setup.** In order:

1. `B.set_backend(args.backend, precision="float32")` and then `B.seed(seed)` for dropout
   masks. This must happen *before* the model is built, because parameters are allocated on
   whatever backend is active ([11-gpu](11-gpu.md)).
2. `TokenData(...)` loads `train.bin`/`val.bin`. `GPTConfig.preset(preset, **model,
   vocab_size=data.vocab_size)` builds the config, with vocab size always taken from the
   data so a config can't disagree with its tokenizer.
3. `GPT(mcfg, seed=seed)` and the optimiser: `"adamw"` gives `AdamW(lr, betas,
   weight_decay)`, `"momentum"` gives SGD with momentum, and anything else (e.g. `"sgd"`)
   gives plain SGD ([08-optimizers](08-optimizers.md)).
4. Write `runs/<name>/config.json`, with the *resolved* model config, so a checkpoint can
   always be rebuilt.
5. Print parameter count and **tokens per parameter**.

**The loop.** For `step` in `start..max_steps`:

```python
if step % eval_interval == 0 or step == max_steps:   # evaluate, log, checkpoint
    ...
    if step == max_steps: break
lr = lr_at(step, ...)            # warmup + cosine
opt.lr = lr
x, y = data.batch("train", bs, T, rng)
model.zero_grad()                # grads accumulate (00-overview §5), so clear them first
_, loss = model.forward(x, y)
model.backward()
gnorm = clip_grad_norm(params, grad_clip)   # returns the pre-clip norm
opt.step()
```

The eval at step $s$ happens *before* update $s$. So the checkpoint saved at step $s$ holds
the weights after exactly $s$ updates, and a run with `max_steps=N` performs $N$ updates
and $N/\text{eval\_interval} + 1$ evaluations.

**The end.** Reload `best.npz` (not the last weights), generate `sample_tokens` tokens from
`sample_prompt` with temperature 0.8 and top-k 50, and write `samples.txt`. That file's
existence is how the sweep runner knows a run finished (§8).

---

## 4. Evaluation with fixed batches

```python
def evaluate(model, data, tc, block_size):
    model.eval()                               # dropout off
    for split in ("train", "val"):
        rng = np.random.default_rng(1234)      # the SAME windows every time
        losses = [float(model.forward(*data.batch(split, bs, T, rng))[1])
                  for _ in range(eval_iters)]
```

Every evaluation of a run, and every run with the same data and block size, scores
**exactly the same windows**. With fresh random batches each time, eval loss would wobble
by sampling noise, and a 0.01 improvement would be indistinguishable from luck. With fixed
batches, a change in eval loss can only come from the weights changing. That's essential
when comparing ablations whose differences are small.

The sample is `eval_iters × batch_size` windows: 40 × 32 windows of 256 tokens for BPE
(327,680 positions, more than the 179,413-token val split, so with overlap it covers most
of it), and 20 × 32 windows of 128 for char (81,920 of 535,993 positions).

Two more details:

* **Train loss is also measured in eval mode** on fixed train batches, so `train_loss` and
  `val_loss` in the log are directly comparable. The per-step `loss` printed every
  `log_interval` steps is different: one fresh batch, dropout on. It is noisier and
  slightly higher.
* **Model selection uses val, reporting uses bpb.** `val_bpb = data.bpb(val_loss)`
  ([01-tokenization §7](01-tokenization.md#7-bits-per-byte-the-fair-metric)), so char and
  BPE runs land on one axis.

---

## 5. Overfitting, tokens per parameter, and dropout

A model with enough parameters can **memorise** its training text instead of learning
patterns that transfer. The symptom: train loss keeps falling while val loss stalls, then
rises. The gap `val_loss − train_loss` measures it, and `summarize.py` reports the final gap
for every run.

A useful single number is **tokens per parameter**, which `train.py` prints at start-up:

| Model | Train tokens | Params | Tokens/param |
|---|---|---|---|
| Anton (BPE 10k) | ≈ 319 k | ≈ 6 M | **≈ 0.05** |
| Kavi `char_kavi.json` | 4.82 M | 0.80 M | 6.00 |
| Kavi `bpe_base.json` (`kavi`) | 1.58 M | 5.76 M | 0.27 |

Anton had about 20 parameters per training token, so memorisation was almost guaranteed,
and without a val set it was invisible. Kavi's BPE model has about 5× more data per
parameter than Anton but is still heavily over-parameterised, which is why val tracking
matters. For scale, compute-optimal training of large models is often quoted at around 20
tokens per parameter. Nobody trains a 6M model on 5 MB of text and gets there.

**Dropout** is the regulariser ([`kavi/layers.py`](../kavi/layers.py) `Dropout`). During
training it zeroes each activation with probability $p$ and scales survivors by $1/(1-p)$,
so no single feature can be relied on. The model is pushed towards redundant, more general
representations. It's applied to the embeddings, the attention probabilities, and the
outputs of the attention and MLP branches. `model.eval()` turns it off, which is why
evaluation calls `model.eval()` and then `model.train()`. Weight decay (AdamW, 0.1 by
default) is the other regulariser.

---

## 6. Best checkpoint = early stopping

```python
if ev["val"] < best:
    best = ev["val"]
    save_checkpoint(run / "best.npz", ...)      # lowest val loss so far
save_checkpoint(run / "last.npz", ...)          # always: for --resume
```

Classic early stopping halts training when val loss stops improving. Kavi lets the run
continue (the cosine schedule is fixed in advance) but **keeps the weights from the best
evaluation**. You get the same result, the model at the val minimum, without having to
guess a patience parameter, and the log still shows what happened after the minimum. The
final sample and every downstream use (`generate.py`, `attention_viz.py`) read `best.npz`.
Selecting on val makes val slightly optimistic. A third, test split would remove that bias;
at this scale it's a deliberate simplification.

**Atomic writes.** `save_checkpoint` in `kavi/checkpoint.py` writes to `<path>.tmp.npz` and
then calls `os.replace(tmp, path)`. A rename within one filesystem is atomic, so if the
process dies mid-save (Kaggle timeout, Ctrl-C, power cut) you're left with the *old* complete
checkpoint, never a half-written one. A checkpoint holds every parameter (`param/<name>`),
the optimiser state (`opt/t`, `opt/m.i`, `opt/v.i`), and a JSON `meta` with step, best val
and the full config.

**Resume.** `--resume` loads `last.npz` into the model *and* the optimiser. AdamW's moment
estimates and step counter `t` matter: restarting Adam from zero moments is a different
optimiser trajectory. The loop restarts at the saved `step`. The learning rate is a pure
function of `step`, so the schedule continues exactly. The batch RNG is reseeded with
`seed + start`, so the batches after a resume are not the ones an uninterrupted run would
have drawn. That's statistically equivalent, but not bit-identical.

---

## 7. Logging

`runs/<name>/log.jsonl` gets one JSON object per evaluation:

| Field | Meaning |
|---|---|
| `step` | updates performed so far |
| `train_loss`, `val_loss` | mean CE (nats/token) on the fixed eval batches, dropout off |
| `val_bpb` | `val_loss` converted to bits per byte |
| `lr` | learning rate of the most recent update (`null` at step 0) |
| `time` | Unix timestamp, so wall-clock time is a difference of two rows |

JSON-lines can be appended to by a running process and read by another while training
continues, and a crash loses at most the line being written (each line is flushed). The
console additionally prints, every `log_interval` steps, the batch loss, lr, **pre-clip
gradient norm** (spikes are the first sign of instability) and tokens/s.

---

## 8. Sweeps and summaries

**`scripts/sweep.py`** runs a list of variants of one base config:

```json
{"base": "configs/bpe_base.json", "set": [...overrides for every run...],
 "runs": [{"name": "gpt2-s1",      "set": ["model.preset=\"gpt2\"", "train.seed=1"]},
          {"name": "gpt2+rope-s1", "set": ["model.preset=\"gpt2\"", "model.pos=\"rope\"", "train.seed=1"]}]}
```

Each run is launched as `train.py base --set name=... <common set> <its own set>` in a
subprocess. Each ablation is therefore a **single-knob diff you can read off the file**:
`gpt2+rope` differs from `gpt2` by exactly `model.pos`. A run is skipped if
`runs/<name>/samples.txt` exists, so a sweep killed by a timeout can be relaunched and
carries on with the unfinished runs. `--only name ...` picks runs, and `--shard k/n` runs
every $n$-th run starting at $k$, so $n$ GPUs can split one sweep ([11-gpu](11-gpu.md)).

`configs/sweep_ablation.json` contains Anton's architecture (`gpt2` + ReLU + biases), the
`gpt2` baseline, the three single upgrades, full `kavi`, `kavi` on char data, and `kavi`
with SGD, with two seeds for the architecture runs. The SGD run necessarily changes the lr
too (0.1 instead of 1e-3), because the two optimisers live on different lr scales.

**`scripts/summarize.py runs/ --plot docs/img/curves.png`** reads every `log.jsonl` and
prints a markdown table sorted by best val bpb: run, data, best val loss, **best val bpb**,
the step it happened at, final val−train gap, and minutes. Runs whose names differ only by
an `-s<seed>` suffix are also averaged into mean/min/max over seeds. That matters: if two
configs differ by less than the seed-to-seed spread, the difference isn't real. `--plot`
draws val bpb against step for every run.

> **Results** (tables, curves, samples) live in the lab journal: see
> [journal.md](journal.md).

---

## 9. How we know it's right: the training gates

`tests/test_training.py` builds a tiny model (vocab 20, $C=32$, 2 layers, $T=16$; 25,888
parameters for `kavi`) and one fixed batch of **random** tokens, $4 \times 16 = 64$ targets.

**`test_adamw_overfits_one_batch`** (both presets): 300 AdamW steps (lr 3e-3, no weight
decay, clipping at 1.0). It asserts the first loss is above 2.5 (≈ $\ln 20 = 3.0$, which
also checks the init) and the last below 0.05.

Why does memorising one batch prove anything? The targets are random, so there is no
pattern to learn. The only way to drive the loss to zero is to fit those 64 answers exactly,
and the model has about 400 parameters per answer, so a correct stack must succeed. If
something in the chain is badly wrong (a sign error, a missing term in a backward pass,
a cache holding stale values, an optimiser that updates the wrong buffer), the loss stalls
or diverges. In practice the AdamW loss goes 3.03 → 0.44 (step 50) →
0.005 (step 150) → 0.00008 (step 300).

It is a **necessary, not sufficient** test. A slightly wrong gradient still points roughly
downhill and can still memorise a batch. The precise check is finite-difference gradcheck
([03-linear-and-backprop](03-linear-and-backprop.md)). The overfit test catches the wiring
around the gradients, which gradcheck never exercises.

**`test_sgd_is_much_slower_than_adamw`**: the same batch, 150 steps each, plain SGD at
lr 0.5 vs AdamW. It asserts AdamW's final loss is under half of SGD's. This is the lesson of
Anton's episode 2 (SGD failed, AdamW worked) turned into a test. In the actual numbers SGD
reaches 0.013 and AdamW 0.005: SGD isn't stuck on this toy problem, just behind. The large
real gap shows up on the full model, which is what the `kavi-sgd` sweep run measures.

---

## 10. Reading a loss curve

Real data first. The first 1000 steps of the CPU char run (`runs/char_kavi/log.jsonl`,
`kavi` preset, 0.80 M params, lr 2e-3, 2000 steps planned):

| step | train | val | gap | val bpb |
|---|---|---|---|---|
| 0 | 4.593 | 4.583 | −0.010 | 6.53 |
| 250 | 1.927 | 1.953 | +0.026 | 2.78 |
| 500 | 1.720 | 1.758 | +0.039 | 2.51 |
| 750 | 1.620 | 1.660 | +0.040 | 2.36 |
| 1000 | 1.546 | 1.597 | +0.051 | 2.28 |

How to read it:

* **Step 0 ≈ $\ln V$.** 4.58 vs $\ln 100 = 4.61$, so the init is sane.
* **The big drop is learning the easy statistics.** By step 250 the model has learned
  character frequencies, common bigrams, spaces and newlines (6.53 → 2.78 bpb). After that,
  each further 0.1 bpb is learning words, then syntax, and costs more steps.
* **The gap is small and growing slowly.** At 6 tokens/param and 2.5 epochs this model
  is still mostly *underfit*. More capacity or more steps would help before regularisation
  would.

The shapes to recognise:

| What you see | Diagnosis | What to try |
|---|---|---|
| Train and val both high and flat, close together | **underfitting**: too small, too few steps, or lr too low | bigger model, more steps, higher lr |
| Train keeps falling, val flattens then **rises** | **overfitting**: memorising | rely on `best.npz`; more dropout/weight decay; smaller model; more data |
| Loss falls, then a sudden spike, maybe a slow recovery | **instability**: an outlier batch or too-high lr | check gnorm spikes; lower lr; longer warmup; keep clipping on |
| Loss **rises from the start** or becomes `nan` | **lr far too high** (or a bug) | lower lr by 3–10×; run the overfit test |
| Loss falls very slowly in a straight-ish line | **lr too low** (or plain SGD on a transformer) | raise lr; use AdamW |
| Train loss in log ≪ printed batch loss | not a bug: the log is dropout-off on fixed batches | compare like with like |
| Curve flattens exactly when the cosine lr reaches `min_lr` | the schedule, not the model, ended progress | longer run or higher `min_lr` |

Compare bpb, not loss, when the tokenizers differ. And compare at equal *tokens seen* when
batch sizes differ.

---

## 11. Pitfalls

* **Re-running a config without `--resume`.** `log.jsonl` is opened in append mode, so a
  fresh run into an existing `runs/<name>/` adds its records after the old ones, and
  `summarize.py` will mix them. Delete the folder or change `name`.
* **A sweep restarts unfinished runs from scratch.** It skips only runs with `samples.txt`
  and doesn't pass `--resume`, so a half-finished run starts again at step 0 (and appends to
  its old log, as above).
* **Shrinking `max_steps` without shrinking `warmup`.** `sweep_probe.json` runs 300 steps
  with the base config's 200-step warmup, so most of a probe is spent warming up. That's
  fine for a smoke test, but misleading as a comparison.
* **Fixed eval batches are a sample.** Two runs within ~0.005 nats of each other may still
  be noise. Use the seed aggregation.

## 12. Try it yourself

1. **Overfit on purpose.** Train `char_kavi.json` with
   `--set model.dropout=0.0 train.max_steps=1500` on a cut-down corpus (point `data` at a
   copy of `data/char` whose `train.bin` holds only the first 50,000 tokens). *Expected:*
   train loss keeps dropping while val loss bottoms out and rises; `best.npz` records the
   minimum step.
2. **Find the test's blind spot.** In a scratch copy of `train_one_batch`
   (`tests/test_training.py`), remove `model.zero_grad()` and print the losses.
   *Expected, and surprising:* it **still passes**, reaching about 0.007 at step 300
   (against 0.00008 with `zero_grad`). Without clearing, `p.grad` becomes a running sum of
   every past gradient. `clip_grad_norm` rescales that buffer in place, which turns the sum
   into something like heavy momentum, and Adam divides out the overall scale. Training
   gets worse, but it doesn't fail. That's the "necessary, not sufficient" point made
   concrete. A missing `zero_grad` is caught by reading the code, not by this test.
3. **lr too high.** Run 200 steps of `char_kavi.json` with `--set train.lr=0.05
   train.warmup=10`. *Expected:* gnorm spikes in the console and a loss that plateaus high
   or diverges, compared with the default lr.
4. **Resume equivalence.** Start a 1000-step run of `char_kavi.json`
   (`--set train.max_steps=1000 name="resume-test"`) and stop it with Ctrl-C a little after
   the step-500 eval. Re-run the identical command with `--resume`, then compare with an
   uninterrupted 1000-step run under another name. *Expected:* `resumed at step 500`; close
   but not identical curves after step 500 (different batches after the resume point, same
   lr schedule); and a duplicated step-500 row in the resumed `log.jsonl`, because the
   resumed loop evaluates again at its first step. (Changing `max_steps` when resuming
   would also change the cosine schedule, since `lr_at` depends on the total.)
