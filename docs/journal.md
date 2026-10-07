# Lab journal

What we actually ran, what came out, and what we think it means. Chapters 00–13 explain how
the machine works; this page records what it did. Every number here comes from a file in
[`results/`](../results/), and every table can be regenerated with
`python scripts/summarize.py results/<name> --plot docs/img/<name>.png`.

---

## 2026-10-07 · Day 1: does it train at all? (CPU)

**Setup.** `configs/char_kavi.json`: 0.80M params, char vocabulary (100), T=128,
4 layers × 4 heads × 128 wide, on the laptop CPU in NumPy.

**Result.** The run was killed for lack of RAM around step 1075, after the best checkpoint at step 1000
had been saved. Val loss fell 4.58 → 1.597 nats per character
(**6.53 → 2.275 bpb**). The step-0 loss of 4.59 matches $\ln 100 = 4.605$, so the model starts
as a uniform guesser, as [chapter 07](07-loss.md) predicts. Throughput was only 1.4–3.8k tokens/s,
because the machine had about 2 GB of free RAM and was busy with other work. That was enough to see it learn,
and too slow for real experiments. This run is the source of the loss curve in
[chapter 09](09-training.md).

## 2026-10-07 · Day 1: the GPU probe (Kaggle, 4 minutes)

Before spending GPU hours we pushed a tiny sweep (`configs/sweep_probe.json`: gpt2 vs kavi,
300 steps each) to check three things:

| question | answer |
|---|---|
| Does our NumPy code run unmodified on CuPy? | Yes. Kaggle assigned **2× Tesla T4**, and CuPy 14.0.1 was preinstalled |
| Are CPU and GPU gradients the same? | `backend_parity.py` in float64: losses identical to 12 digits, worst gradient difference **7.6e-16** |
| How fast? | **32–33k tokens/s per T4**, about 10–20× the laptop |

After 300 steps, kavi was already ahead: **2.156 vs 2.335 val bpb**
([`results/probe/`](../results/probe/)). The kernel was then changed to run one sweep shard
per GPU in parallel ([chapter 11](11-gpu.md)).

## 2026-10-07/08 · The ablation sweep

**Question.** Anton's model was a GPT-2-style block. Kavi's default is a Llama-style block. Which
of the three changes (RoPE, RMSNorm, SwiGLU) actually helps, by how much, and is the
difference bigger than seed noise?

**Setup** (`configs/bpe_base.json` + `configs/sweep_ablation.json`):

- data: all of Shakespeare, byte-level BPE with 4096 tokens, 1.58M train tokens and 179k val tokens;
- model: C=256, 6 layers, 8 heads, T=256, dropout 0.2, no biases, about **5.8M params** for
  every config (SwiGLU's hidden width is set to 8C/3 so the parameter count matches, see
  [chapter 06](06-mlp.md));
- training: AdamW (β=0.9/0.95, wd 0.1), lr 1e-3 with 200 warmup steps and cosine decay to 1e-4, clip 1.0,
  batch 32 × 256 = 8192 tokens, **4000 steps = 32.8M tokens ≈ 21 epochs**;
- every config differs from `gpt2` by **one knob**, and the six main configs run with 2 seeds each;
- hardware: 14 runs on two T4s in **160 min** (18–23 min per run). Free tier.

### Results

| config | what changed vs `gpt2` | val bpb (mean of 2 seeds) | seed spread | Δ vs gpt2 |
|---|---|---|---|---|
| **kavi** | RoPE + RMSNorm + SwiGLU | **1.6269** | 0.0004 | **−0.080** |
| gpt2+rope | learned positions → RoPE | 1.6321 | 0.0012 | −0.075 |
| anton | GELU → ReLU, biases on | 1.6974 | 0.0009 | −0.009 |
| gpt2+swiglu | GELU MLP → SwiGLU | 1.6986 | 0.0012 | −0.008 |
| gpt2+rmsnorm | LayerNorm → RMSNorm | 1.7065 | 0.0003 | −0.000 |
| gpt2 | baseline | 1.7067 | 0.0008 | — |
| kavi-char (1 seed) | char vocabulary instead of BPE | 1.8027 | — | — |
| kavi-sgd (1 seed) | AdamW → plain SGD (lr 0.1→0.01, no momentum) | 2.2381 | — | — |

![validation curves](img/ablation_curves.png)

*Left: the whole run. Right: zoom on steps 2000–4000. The shaded bands are the min–max over seeds and are
barely visible because the seeds agree so closely.*

### What we learned

1. **RoPE accounts for nearly all of the gain.** Swapping learned position embeddings for RoPE
   alone gives −0.075 bpb, about 94% of the full kavi improvement. Our reading: RoPE gives every head
   relative position for free, and with only 1.58M training tokens, a learned 256×256 position table is
   a lot to learn from scratch ([chapter 05](05-attention.md)).
2. **The ranking is real, not noise.** The two seeds of a config differ by at most 0.0012 bpb.
   The gaps between gpt2+rope, gpt2+swiglu and gpt2 are 6–60× that.
3. **RMSNorm ties LayerNorm.** We got 1.7065 vs 1.7067, well within seed noise. (Seed 1 of both even
   matched to 4 decimals at step 4000. We checked: the configs differ and the curves differ at
   every earlier step, so this is a coincidence.) RMSNorm's selling point in big models is
   speed and simplicity, not quality, and that's consistent with this result.
4. **SwiGLU and RMSNorm mostly buy faster early progress.** At step 2000, kavi leads
   gpt2+rope by 0.018 bpb; by step 4000 the lead is 0.005. gpt2+swiglu's lead over gpt2
   shrinks from 0.025 to 0.008 over the same stretch. RoPE's lead doesn't shrink.
5. **A surprise: Anton's block beats the GPT-2 block.** `anton` (ReLU + biases) is 0.009
   better than `gpt2` (GELU, no biases). Two knobs changed together, so we can't say which
   one did it. A follow-up should run `gpt2+bias` and `gpt2+relu` separately. One hypothesis is that with
   learned positions, the query and key biases let a head score keys by position in a way
   that doesn't depend on content. (The key bias alone is provably useless, see
   [chapter 05 §8](05-attention.md); the query bias isn't.)
6. **SGD fails the way it failed for Anton.** Same model and same budget, with plain SGD
   instead of AdamW: 2.238 vs 1.627 bpb. After 4000 steps it still hasn't reached where AdamW was
   at step 400. The sample below shows what that gap reads like. [Chapter 08](08-optimizers.md)
   explains why.
7. **BPE beats characters at equal steps.** Kavi on characters reaches 1.803 bpb vs 1.627 on BPE.
   Caveat: at equal steps, a 256-token window covers about 776 bytes of BPE text but only
   256 bytes of characters, so the BPE model also saw about 3× more text. This is a
   comparison at equal compute, not at equal data.
8. **The better model overfits more.** At the end, the val−train gap was 0.60 nats for kavi
   vs 0.42 for gpt2, after about 21 passes over the data. Val loss was still falling at step
   4000 in every run (best checkpoint = last or second-to-last eval), so dropout 0.2 is holding.
   More data, rather than more steps, is the obvious next lever ([chapter 09](09-training.md)).

### Samples

Generated at the end of each run from the best checkpoint's weights (`train.sample_tokens=400`,
temperature 0.8, top-k 50, as `train.py` does). The prompt is a newline. Excerpts are unedited.

**kavi (1.627 bpb).** It holds one scene (Hamlet and Horatio) with consistent speakers and
asides:
```
HORATIO.
O my lord, you are not yet in it.

HORATIO.
This is a very grievous thing to th’ unseen.

HORATIO.
[_Aside._] Though I call thee this, boy, I had rather have beat thee.

HAMLET.
I am going to my lord.
```

**gpt2 (1.707 bpb).** The verse is fluent, but the history-play names pile up without sense:
```
SUFFOLK.
Then were the first prince so, when was his son
Hath got the Duke of Salisbury.

SURREY.
Where’s the Earl of York?
```

**kavi-sgd (2.238 bpb).** The form is still there, but the grammar falls apart:
```
KING.
SECOND LORD.
Why, I had a man, and thy love
And I’er a word, then,
That they will tell me.
Nay, he hath that I see my son?
```

**kavi-char (1.803 bpb).** Character level: well-formed words and plausible names, with loose sense:
```
LADY MACBETH.
As thou art not as long as do make me any.

CLEOPATRA.
What kneel you here?
```

And with the KV cache from a prompt (`python scripts/generate.py ckpt/kavi-s2.npz --prompt
"ROMEO:" --tokens 120 --seed 1`):
```
JULIET.
I am sure I am, and fear not how to love.

ROMEO.
Put on my poor achievement to my soul.
```

## 2026-10-08 · Looking inside the best model

### Attention maps

`python scripts/attention_viz.py ckpt/kavi-s2.npz` on *"ROMEO:\nBut soft, what light through
yonder window breaks?"*:

![attention patterns of kavi-s2](img/attention.png)

Things that stand out (see [chapter 13](13-interpretability.md) for how to read the grid):

- **Previous-token heads.** Layer 4 head 5 is an almost perfect line one step below the
  diagonal. Layer 3 heads 2–3 are softer versions of the same thing.
- **One-token "anchor" heads.** Layer 3 head 6 sends almost every query to one early token
  (the newline after `ROMEO:`), and several layer-1 and layer-2 heads put much of their weight
  on the first token. This is the "attention sink" pattern: when a head has nothing useful to
  do, it parks its attention somewhere harmless.
- Layer 0 heads are diffuse, averaging over recent tokens.

### Searching for induction heads: a negative result

An *induction head* implements "the last time I saw the current token, what came next? Predict
that." It's the classic copy circuit behind in-context learning, and it's built from a previous-token head
plus a head that matches on the previous token's output. We wrote
[`scripts/induction.py`](../scripts/induction.py) to look for one. It feeds the model a
50-token sequence followed by the *same* 50 tokens and measures, per head, how much attention
position $i$ pays to position $i-50+1$ (the token after the earlier occurrence).

| model | loss on 1st copy | loss on 2nd copy | strongest induction score | strongest prev-token score |
|---|---|---|---|---|
| kavi-s2 | 3.119 | 2.973 | 3.3% | 42.8% (L4 H5) |
| gpt2-s2 | 3.211 | 3.023 | 7.4% | 37.1% |
| anton-s2 | 3.189 | 2.981 | 8.7% | 41.2% |

*(Real val passages, 8 sequences. Uniform attention would score about 1.3%. With uniformly random token
ids the kavi losses are 14.83 → 14.82, and with tokens drawn at random from the val split 10.23 → 10.11.)*

**No model has an induction head yet.** A model with one would score tens of percent on
the induction target, and its second-copy loss would collapse toward 0. Here the
second copy is only 0.15–0.21 nats easier, which is ordinary context help, not copying. All three
models *do* have strong previous-token heads, the first half of the circuit. In the
literature, induction heads form through a fairly sudden phase change during training, and our
models (5.8M params, 33M tokens) haven't reached it. This is a good experiment to rerun on
longer or bigger training.

## Known limitations of these results

- **One budget.** Every run is 4000 steps. A ranking at 4000 steps can change at 40,000
  (point 4 above shows the gaps already moving).
- **Two seeds** for the main configs and one for char and SGD. Enough to show the gaps are
  real, not enough for error bars on small differences.
- **SGD wasn't tuned, and it ran without momentum.** `optimizer: "sgd"` builds plain SGD; only
  `optimizer: "momentum"` uses `train.momentum`. The run's `config.json` still records
  `momentum: 0.9` (the default), but nothing read it, and SGD gets no weight decay. A `momentum`
  run with a tuned lr would narrow the gap, though Anton's experience and the theory both say
  it won't close it.
- **`TokenData.batch` never picks the very last token of a split as a target** (an off-by-one in
  the random start, `rng.integers(0, len(data) - block_size - 1)`). We left it unfixed so these
  numbers stay exactly reproducible. It affects one token out of 1.58M.

## What's next

- [ ] `gpt2+bias` and `gpt2+relu` single-knob runs, to explain finding 5
- [ ] `kavi-momentum` (SGD + momentum 0.9, lr sweep), the fair version of finding 6
- [ ] A longer run of the best config (e.g. 20k steps) to look for the induction-head phase change
- [ ] More data: a bigger public-domain corpus, to fight the 21-epoch overfitting (PLAN phase 8)
