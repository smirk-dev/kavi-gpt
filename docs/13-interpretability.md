# 13 · Interpretability: looking inside the heads

> **What you'll learn.** Every attention head produces an explicit table of "who looked at whom",
> and `scripts/attention_viz.py` draws all of them. This chapter explains what the plot and its
> two summary numbers mean, which patterns to look for (previous-token heads, sink heads,
> induction heads), why heads specialise at all, how to turn a hunch into a test, and why an
> attention map is weaker evidence than it looks.

Prerequisites: [attention](05-attention.md). Anton's series never looked inside the model;
this is one of Kavi's additions.

---

## 1. What the script does

```
python scripts/attention_viz.py runs/<run>/best.npz --text "ROMEO:\nBut soft, ..." --out docs/img/attention.png
```

1. Loads the checkpoint and its tokenizer, and encodes the text (truncated to `block_size`).
2. If the model was trained with `attn="flash"`, it switches every block back to the naive path.
   Flash attention never builds the probability matrix ([chapter 12](12-flash-attention.md)),
   and we need it here. Both paths compute the same function, so the picture is the same.
3. Runs one forward pass in eval mode (no dropout) and grabs `blk.attn.probs[0]` from every block:
   the cached softmax output, shape `(H, T, T)` per layer.
4. Prints two statistics per head and saves a grid of heatmaps.

**Reading a heatmap.** Rows are layers, columns are heads. In each small image, **row $i$ is the
attention distribution of query token $i$**: column $j$ is $P_{ij}$, how much token $i$ took from
token $j$. Every row sums to 1. The upper-right triangle is always black because of the causal mask.
The colour scale is fixed to $[0, 1]$ (`vmin=0, vmax=1`), so brightness is comparable across heads.

## 2. The two printed statistics

For a head with probabilities $P$ over $T$ tokens, with distance $i - j \ge 0$:

$$
\text{look-back} = \frac1T\sum_{i}\sum_{j\le i} P_{ij}\,(i-j), \qquad
\text{entropy} = \frac1T\sum_i \Big(-\sum_{j\le i} P_{ij}\log P_{ij}\Big)\ \text{nats}.
$$

(In code: `(p * dist).sum(-1).mean()` and `-(p * log(p + 1e-12)).sum(-1).mean()`. The `1e-12`
avoids $\log 0$ on masked entries, which contribute $0\cdot\log(10^{-12}) = 0$.)

- **Look-back** is the average distance a token reaches back. 0 means "attends to itself", 1 means
  "the previous token".
- **Entropy** measures how spread out the attention is. 0 means all weight on one token; row $i$
  can reach at most $\log(i+1)$ (uniform over everything visible).

**Baselines** help when reading these. For $T = 57$, a head attending *uniformly* has
look-back $(T-1)/4 = 14$ and entropy $\frac1T\sum_i \log(i+1) \approx 3.1$ nats. A perfect
previous-token head has look-back $\approx 1$ and entropy $\approx 0$. Early rows have few
choices (row 0 always has entropy 0), which pulls both averages down a little.

## 3. Patterns to look for

**Previous-token heads.** A bright line just below the diagonal: token $i$ looks at $i-1$.
These are the simplest useful heads. For a character model the previous character is the
strongest single clue to the next one. With RoPE ([chapter 05 §11](05-attention.md)) "one
step back" is a fixed rotation offset, which makes such heads easy to learn.

**First-token / "attention sink" heads.** A bright first column (or a bright column at some
landmark token). Softmax forces each row to sum to 1, so a head that has nothing useful to
contribute for a token still has to put its weight somewhere. A fixed, always-visible token
is a convenient dump, and if that token's value vector is small the head effectively
switches itself off. Note that the first few rows of *every* head look like this trivially, since
row 0 can only see column 0.

**Induction (copy) heads.** In text like `... Romeo said ... Romeo s`, after the second `Romeo` the
model can predict `said` by finding the earlier occurrence of the current token and copying what
came **after** it. An induction head at position $i$ (current token $A$) attends to position
$j+1$, where $j < i$ is an earlier occurrence of $A$. In the heatmap this is a bright off-diagonal
stripe. Mechanistically it takes two layers: a previous-token head in an earlier layer writes "the
token before me was $A$" into position $j+1$, and the induction head's query ("I am $A$") matches
that key. This is the simplest form of *in-context learning*.

**Broad / averaging heads.** High entropy, bright smear over a whole region, typically in later
layers, gathering a summary of the recent context.

## 4. Why heads specialise

Nothing in the code tells head 2 to track the previous token. Specialisation emerges:

- Heads start from different random weights, so from step 1 they receive different gradients.
- Once a head is slightly useful for some pattern, the gradient pushes it to do that pattern
  *better*. That's a rich-get-richer dynamic.
- All heads write into the same residual stream. If one head already supplies "previous
  character", a second head copying it lowers the loss much less than a head that supplies something
  new, so the gradient favours diversity (though in practice some redundancy remains).

## 5. A first look: the 1000-step char model

Here are the statistics from the CPU checkpoint `runs/char_kavi/best.npz` (4 layers × 4
heads, 1000 steps, val bpb 2.275) on the default text
`"ROMEO:\nBut soft, what light through yonder window breaks?"` (57 characters):

| layer | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|
| 0 | 1.28 / 0.36 | 4.75 / 0.47 | 2.54 / 0.72 | 3.19 / 0.86 |
| 1 | 4.41 / 1.46 | 5.42 / 1.64 | 4.43 / 1.37 | **17.05** / 1.64 |
| 2 | 6.01 / 1.64 | 5.75 / 1.84 | 2.47 / 0.86 | 2.67 / 0.95 |
| 3 | 7.77 / 2.02 | 6.92 / 1.85 | 5.24 / 1.52 | 3.39 / 1.19 |

(each cell: look-back distance / entropy in nats)

What the numbers and the heatmap show:

- **Layer 0 is sharp and local.** All four heads have entropy below 0.9 nats (uniform would be
  about 3.1). Layer 0 head 0 is close to a previous-token head: on average 53% of each row's
  weight lands on $i-1$ and 33% on the token itself. In the heatmap it's a crisp line hugging the diagonal.
- **Layer 1 head 3 is a landmark head.** Its look-back of 17 is far above any other head. Its argmax
  is column 6, the `\n` after `ROMEO:`, for most rows, with 46% of the weight on average. It
  marks "start of the line / after the speaker name", which is useful in a play script where
  line structure carries a lot of information. It behaves like a sink on a meaningful token.
- **Deeper layers spread out** (entropy 1.2–2.0), mixing broader context.

**Testing for induction heads (section 6's method):** none yet. On a random sequence of 40
lowercase letters repeated twice, every head put at most 0.02 of its weight on the induction
target $i - 40 + 1$, and the loss on the second copy (6.56 nats/char) was no better than on the
first (6.50). Real induction heads would make the second copy nearly free. A 1000-step model on
character data hasn't developed them.

### The fully trained GPU model (kavi-s2)

The best model from the ablation sweep (BPE, 6 layers × 8 heads, 4000 steps, 1.627 val bpb):

![Attention patterns of kavi-s2](img/attention.png)

The same patterns, sharper. Layer 4 head 5 is an almost perfect previous-token head. Layer 3 head 6
is a landmark head that parks almost every query on the newline after `ROMEO:`. Several heads in
layers 1–2 sink onto the first token. With `scripts/induction.py` (section 6's test, automated)
the previous-token heads put 40–55% of their weight on $i-1$, but **no head scores above 3.3% on
the induction target**, and the 5.8M-parameter model trained on 33M tokens still has no induction
head. The numbers, plus the gpt2 and anton baselines (7–9%), are in
[the journal](journal.md#searching-for-induction-heads-a-negative-result).

> **Update (2026-10-09).** The scale-up runs found them. Trained on 11.9× more text, the same
> 6.8M-parameter shape reaches a 0.13 induction score, and the 29M model 0.26 with a 1 nat copy
> gain. The same small model trained on Shakespeare alone never gets past 0.02. What was missing
> was diverse data, not training length. See [chapter 14, section 7](14-scaling.md#7-results).

## 6. How to test a hypothesis

A heatmap suggests; a test decides. The pattern:

1. **State the hypothesis precisely**, e.g. "layer $\ell$ head $h$ is an induction head: at
   position $i$ it attends to the token after the previous occurrence of token $i$."
2. **Construct input where the hypothesis makes a sharp prediction and confounds are removed.**
   For induction: a random token sequence of length $L$ (no natural-language regularities to
   exploit) repeated twice. In the second copy, the previous occurrence of position $i$'s token is
   at $i - L$, so an induction head should attend to $i - L + 1$.
3. **Measure**: the average attention $P_{i,\,i-L+1}$ over the second copy, per head, compared with
   a baseline (e.g. $P_{i,\,i-L}$, or the uniform value $1/(i+1)$).
4. **Check the behaviour, not just the attention**: the per-token loss on the second copy should
   drop sharply if the model really copies.
5. **Intervene**: ablate the candidate head (exercise 3 in section 9) and confirm the second-copy loss
   rises again. That turns "this head attends there" into "this head is *why* the model copies".

## 7. Limitations of reading attention maps

- **Attention weight is not importance.** Head output is $\sum_j P_{ij} v_j$, followed by `proj`. A
  large $P_{ij}$ with a tiny $v_j$ contributes nothing. Sink heads exploit exactly this.
- **Later layers attend to mixtures.** In layer 3, "position $j$" holds a residual-stream vector that
  already absorbed information from many earlier tokens. A layer-3 head looking at $j$ is not
  looking at the *character* $j$ in any simple sense.
- **One input is an anecdote.** A head can behave differently on dialogue, stage directions and
  sonnets. Average over many inputs before concluding.
- **Summary statistics hide structure.** A head that splits its weight between "previous token" and
  "the `\n`" has a medium look-back that describes neither behaviour.
- **Correlation, not causation.** Only interventions (ablating or patching heads) show that a head
  matters for the output.
- **Not the whole model.** The MLPs, roughly two-thirds of each block's parameters, don't show up in attention maps at all.

## 8. Pitfalls

- **Visualising a flash-trained model without switching paths**: there is no `probs` to read. The
  script handles this; custom notebooks must too.
- **Visualising with dropout on**: call `model.eval()` first (the script does), or the maps are
  noisy and the `probs` cache differs from what a real forward computes.
- **Over-reading the first rows**: row 0 can only look at column 0. That isn't a sink head.
- **Text longer than `block_size`** gets truncated by the script; patterns at long range need a
  model with a long enough context.

## 9. Try it yourself

1. **Baselines.** Make a fake head with uniform causal attention for $T=57$ and run the script's two
   formulas on it. *Expected:* look-back 14.0, entropy about 3.1 nats. Then a perfect previous-token head
   (row 0 on itself, row $i$ on $i-1$): look-back $56/57 \approx 0.98$, entropy 0.
2. **Induction probe.** Encode a random 40-letter lowercase string twice, run
   `model.forward`, and for each head compute the mean of `probs[0, h, i, i - 39]` over
   `i in range(40, 80)`. *Expected on the 1000-step char checkpoint:* every value ≤ 0.02 (section 5).
   On a model with induction heads, one or more heads would stand far above that baseline.
3. **Ablate a head.** Head $h$'s output occupies columns `h*D:(h+1)*D` of the merged tensor
   (`_merge_heads`), so zeroing rows `h*D:(h+1)*D` of `blocks[ℓ].attn.proj.weight.data` removes its
   contribution. Do it for layer 0 head 0 (the previous-token head) and measure the loss on a fixed
   validation batch. Restore the weights afterwards (copy them first). *Expected* (measured on the
   1000-step char checkpoint, 16 × 128 validation tokens, base loss 1.650): removing any **single**
   head costs only about 0.01 nats (L0H0 +0.009, L1H3 +0.010, L3H0 +0.016), so the heads are
   redundant enough that no one of them is critical. Zeroing a **whole layer's** `proj` hurts far more:
   layer 0 +0.38, layer 1 +0.07, layer 2 +0.14, layer 3 +0.09. The sharp local heads of layer 0
   matter most as a group. This is also a lesson in humility: the head that looks most striking
   (L1H3) isn't the one whose removal hurts most.
4. **Different text, different heads.** Run the script on a stage direction
   (`"[_Exeunt._]\n\nSCENE II."`) and on a sonnet line. *What to look for:* whether layer-0
   head 0 keeps its look-back near 1 (a position-based head should be content-agnostic), and where
   layer 1 head 3 puts its weight when the text has several newlines, or none. If it sticks to the
   first newline, or falls back to column 0, that tells you whether it tracks "a newline" or "a
   fixed landmark". (Not yet measured. Record what you find in the journal.)
