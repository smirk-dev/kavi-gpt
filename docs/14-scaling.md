# 14 · Scaling up: more text, a bigger model, a budget

> **What you'll learn.** Why the ablation models stopped improving, and why the cure is *more
> data* before *more parameters*. How to assemble a bigger corpus without contaminating your
> validation set. Two bugs in the first version of our own corpus builder, and how we caught them.
> How to estimate, before spending a single GPU-hour, whether a run fits into a free Kaggle
> session.

Prerequisites: [tokenization](01-tokenization.md), [training](09-training.md),
[GPU](11-gpu.md). This is PLAN phase 8; the numbers it produces go in the [journal](journal.md).

---

## 1. Why scale, and why data first

The ablation ([journal](journal.md)) trained 5.8M-parameter models for 4000 steps of
32 × 256 tokens. That is 32.8M tokens seen, from a training split of only 1.58M tokens, so
**every token was seen about 21 times**. The logs show the consequence. By step 4000, kavi's
train loss is 2.82 nats while its val loss has flattened at 3.41 (the best checkpoint is at
step 3800–4000). A 0.6-nat gap means the model is increasingly memorising the training
split rather than learning more English.

Adding parameters to that setup would make memorising *easier*, not harder. There are two
levers:

| lever | effect on overfitting | cost |
|---|---|---|
| more parameters | worse (more capacity to memorise) | more compute per token |
| more *distinct* text | better (fewer repeats) | finding the text |

The rough rule from Hoffmann et al. 2022 ("Chinchilla") is that compute-optimal training uses
about **20 tokens per parameter**. A 29M model would want roughly 590M tokens. Nothing close to
that exists for Elizabethan English. What we can do is cut the repetition: with about 12× more
distinct text, the same 20k-step run repeats each token about 9 times instead of about 100.
Muennighoff et al. 2023 ("Scaling Data-Constrained Language Models") found that up to about 4
epochs of repeated data is almost as good as fresh data, with returns falling off beyond that.
So 9 epochs is in the "still useful, but diminishing" zone. That is acceptable for a free-tier
learning run, and worth remembering when you read the curves.

## 2. Which text? A rule, not a wish-list

We want text that *helps the model at Shakespeare*. Validation stays Shakespeare's val split,
so the bits-per-byte number stays directly comparable with the ablation's 1.627. Modern English
would teach the model a different language. What we want is the same idiom: Marlowe, Jonson,
Fletcher, Webster, Spenser, Donne, Milton, and the 1611 Bible.

Hand-picking 140 books invites bias and is hard to reproduce. `scripts/build_corpus.py select`
applies a rule to Project Gutenberg's machine-readable catalog (`pg_catalog.csv`):

- `Language` is `en` and `Type` is `Text`;
- the Library of Congress class starts with **`PR`** (English literature);
- the **first listed author was born 1530–1625**, which covers the generation before
  Shakespeare (b. 1564), his contemporaries, and the one after;
- no author string contains "Shakespeare";
- plus an explicit `EXTRA_IDS` list, each entry with a reason. Today it holds one item: the King
  James Bible (#10), which is in class BS, not PR, but is the most-read English text of 1611.

The output is 137 texts, frozen into `configs/corpus_early_modern.json`, so anyone rebuilding
gets the same list even if the catalog changes. The biggest contributors by count are Beaumont
(17 volumes), Jonson 12, Marlowe 11, Milton 10, Fletcher 8, and Spenser 6. By *size* the corpus
is lopsided: the KJV is 7.3% of it, Burton's *Anatomy of Melancholy* 5.6%, and one volume of
*The Faerie Queene* 4.7%. Prose and epic outweigh drama. Keep that in mind when the model starts
sounding like a sermon.

The downloader is deliberately polite. It waits 1.5 s between requests, sends a descriptive
User-Agent, and is resumable: files already on disk are skipped. It also retries. On the first
run *Paradise Lost* failed with `IncompleteRead`, a dropped connection mid-file.

## 3. Contamination: the validation set is hiding in the training data

This is the subtle part. Shakespeare's contemporaries **quote** him, and their 19th-century
editors quote him even more, in footnotes like:

> "But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun!"
> ROMEO AND JULIET, act ii. sc. 2.

Any validation line that appears in the training data makes val loss lie: the model scores
well on text it has *memorised*, not predicted. The guard is **n-gram overlap**.

```python
def words(s):  return re.findall(r"[a-z]+", s.lower())          # case + punctuation blind
def ngrams(ws): return {hash(tuple(ws[i:i + 8])) for i in range(len(ws) - 7)}

banned = ngrams(words(shakespeare_complete_works))              # train AND val
drop paragraph p  if  ngrams(words(p)) & banned
```

Why 8 words? Shorter n-grams flag innocent phrases. "*I pray you, sir*" is everywhere in the
period. Longer n-grams miss quotations with a word changed. With 8-grams, the KJV's
"*God save the king, God save the king*" does get flagged, a false positive that costs us one verse.
That is the price of a guard that errs toward caution. Why the *complete works*, not just val? A
paragraph matching train text would be harmless duplication, but we would rather not double-count
Shakespeare in a corpus meant to be *other* voices. The check is cheap either way: a Python set of
about a million hashes.

### The collaboration we didn't expect

Running the guard, one volume dropped 471 paragraphs while every other text dropped 50 or
fewer: Beaumont & Fletcher, vol. 9. The reason is that it prints ***The Two Noble Kinsmen***,
which Fletcher co-wrote with Shakespeare, and Gutenberg #100 includes it too. The play is in
*both* corpora.

The 8-gram guard caught the long speeches. But plays are full of short ones ("*Fair cousin, I am
glad.*") with fewer than 8 words, so they have no 8-gram and sailed through between the flagged
paragraphs. The fix is a **span fill** (`leak_mask`). If two flagged paragraphs are at most
`SPAN_GAP = 10` paragraphs apart, everything between them is dropped too. A leaked play is a
dense run of hits, so it gets removed whole. An isolated footnote quotation is a lone hit, so it
removes only itself. `tests/test_corpus.py` pins both behaviours.

How do we know it worked? 195 lines from the canon's text appear *verbatim* in the raw B&F volume.
The two editions differ in spelling and line breaks, so most lines don't match character for
character. After the span fill, **1** of those 195 survives in the built corpus. We also checked where the
play sits in our own split: 0 of 400 sampled lines are in `val.txt`. So even the old leak only
duplicated *training* text, but we had no way to know that until we looked.

## 4. Deduplication, and the bug that deleted "Exeunt"

The same text appears more than once on Gutenberg: two editions of *Faustus*, a play printed in
a "Works" volume and again on its own. Repeats are bad for the same reason as epochs: they
tilt the model toward memorising. The first version of `build` therefore dropped any paragraph
whose normalised text had been seen before.

The statistics looked wrong: 22,585 paragraphs dropped as duplicates, and 8 whole texts skipped
as "another edition". Counting what was dropped:

```
'exeunt' 381   'footnotes' 354   'arb' 335   'madam' 323   'exit' 302
'illustration' 283   'scene ii' 130   'sir' 149   ...
```

Under 8 words: 30,987 drops. 8 words or more: 4,951. The dedup was deleting every "*Exeunt.*",
every "*Madam.*" and every speaker-name line after its first appearance anywhere in the corpus.
These are not duplicates in any useful sense; they are the grammar of a play. Without them, the
model would see plays in which characters never exit. The whole-text skip was polluted the same
way: a play is mostly short lines, so "more than half of its paragraphs already seen" fired on
plays that were never duplicated.

The fix applies the same threshold as the leakage guard: only paragraphs of **8+ words** take part
in dedup and in the whole-text test. After it, 4,942 real duplicates are dropped and 0 texts are
skipped.

The general lesson: **read what your filter removes**, not just how much. A count of 22k looks
like thoroughness until you look at the top of the list.

## 5. The built corpus and its tokenizer

`python scripts/build_corpus.py all` then
`python scripts/tokenize_data.py bpe --vocab 8192 --extra data/extra.txt`:

| | Shakespeare only (ablation) | + early-modern corpus (phase 8) |
|---|---|---|
| training text | 4.8M chars | 4.8M + 57.8M chars (**11.9×**) |
| tokenizer | BPE 4096 | BPE 8192 |
| train tokens | 1.58M | **18.69M** |
| val tokens (same Shakespeare val text) | 179,413 | 171,711 |
| val bytes/token | 3.03 | 3.16 |

`--extra` appends the corpus to the **training text only**; val is untouched. Why a bigger
vocabulary? With 12× more text, rarer words and spellings ("*doth*", "*hath*", "*Faustus*")
occur often enough to earn a token of their own, and each val token now covers 3.16 bytes
instead of 3.03. The cost is a bigger embedding matrix: 8192 × C parameters, which is 4.2M of
the 29.4M in the largest model.

Because bits-per-byte divides by the *val split's own* bytes per token
(`TokenData.bpb`, [chapter 7](07-loss.md)), the switch from BPE 4096 to 8192 doesn't break the
comparison. Loss *per token* would. That is exactly why we report bpb.

## 6. Three model sizes, and a budget before a run

`configs/scale_base.json` is the kavi preset (RoPE, RMSNorm, SwiGLU) with T = 256, dropout
0.1 instead of 0.2 (more data means less need to regularise), and AdamW with a lower peak lr,
since wider models want smaller steps.

| name | C | L | H | params |
|---|---|---|---|---|
| S | 256 | 6 | 8 | 6.8M |
| M | 384 | 8 | 6 | 17.3M |
| L | 512 | 8 | 8 | 29.4M |

Training costs about **6 FLOPs per parameter per token**. The forward pass costs 2: a
multiply and an add per weight. The backward pass costs 4, because every `Linear.backward` in
[chapter 3](03-linear-and-backprop.md) does *two* matmuls the size of the forward one:
`dx = dy @ W.T` for the input gradient and `dW = x.T @ dy` for the weight gradient. Attention adds a T-dependent term
that is small at T = 256. Calibrating against what we *measured* in the ablation:

$$
\frac{6 \times 5.8\text{M params} \times 32.8\text{M tokens}}{\approx 23 \text{ min}}
\approx 0.8\ \text{TFLOP/s on a T4}
$$

A T4's fp32 peak is about 8 TFLOP/s, so we achieved roughly **10% of peak**. That is unsurprising
for unfused NumPy-style kernels: every elementwise op is a separate pass over memory. At that
rate, 20k steps (164M tokens) cost:

| size | 6·N·D | estimate at 0.8 TFLOP/s |
|---|---|---|
| S 6.8M | 6.7e15 | ~2.2 h |
| M 17.3M | 1.7e16 | ~5.7 h |
| L 29.4M | 2.9e16 | ~9.7 h |

Kaggle's free tier gives about 30 GPU-hours a week and 12 hours per session, so L at 20k steps
*just* fits, with no margin for the evaluation passes. Bigger matmuls usually run closer to peak,
so the real number may be better. **Measure it, don't trust it.**
`configs/sweep_scale_probe.json` runs 300 steps of each size and prints the throughput, and only
then do we size the long run.

### What the probe measured (Kaggle v4, 2026-10-09, `results/scale_probe/`)

Median tokens per second on one T4, after the first logged interval:

| size | tok/s | achieved 6·N·tok/s | 20k steps, training only |
|---|---|---|---|
| S 6.8M | 29,828 | 1.22 TFLOP/s | 1.5 h |
| M 17.3M | 17,823 | 1.85 TFLOP/s | 2.6 h |
| L 29.4M | 11,657 | **2.06 TFLOP/s** | **3.9 h** |

The estimate said 9.7 h for L, and the measurement says 3.9 h. Efficiency rose with width, from
1.2 to 2.1 TFLOP/s, about 25% of the T4's peak. The 0.8 TFLOP/s calibration came from a
C = 256 model whose matmuls are too small to keep the GPU busy, so per-op overheads dominate:
kernel launches, elementwise passes, Python. At C = 512 each matmul does 4× the work for
roughly the same overhead. This is the most useful lesson of the probe: **a throughput number
measured at one size doesn't transfer to another.** Re-measure whenever the shapes change.

With the real numbers, all three sizes fit at 20k steps, with room for a control. GPU 0 runs L
and then S trained on *Shakespeare only*. GPU 1 runs S and then M. That is about 6 h of wall time
(`configs/sweep_scale.json`). That is 6 hours of the ~30-hour weekly quota if Kaggle counts session time, or up to
12 if it counts each GPU. Check the quota bar after the run. The control uses the same
8192-token BPE trained on Shakespeare alone: 1.44M train tokens, so 20k steps is about 114
passes. Same model, same tokenizer size, same steps, 13× less data. Any difference between it
and S is the data's doing.

**What it actually took (v5).** L 4.1 h, M 3.1 h, S 1.8 h, the control 1.6 h; the whole kernel
5.9 h. Against the probe's training-only numbers, that is +6% for L, +7% for the control, +17% for S
and +21% for M. The extra is evaluation (40 batches × 2 splits every 500 steps), the induction
probe and checkpoint writes. Why the two GPU-1 runs paid more is not clear; one guess is that the
two processes share the CPU, which matters most for small, overhead-bound models. Budget about
20% on top of a probe's number.

## 7. Results

Kaggle v5, pushed 2026-10-09 11:25 IST. It ran for 5.9 h (21,272 s) against the ~6 h forecast,
on 2× T4. Raw logs are in `results/scale/`. Each run took 20,000 steps × 32 × 256 = 163.8M
tokens, one seed each. In the mixed runs that is 8.8 passes over 18.7M train tokens; the control
made 114 passes over its 1.44M.

| run | params | data | best val bpb | at step | final val−train gap | hours |
|---|---|---|---|---|---|---|
| **L** | 29.4M | Shakespeare + corpus | **1.4548** | 20000 | +0.347 | 4.1 |
| M | 17.3M | Shakespeare + corpus | 1.4692 | 20000 | +0.193 | 3.1 |
| S | 6.8M | Shakespeare + corpus | 1.5311 | 20000 | +0.013 | 1.8 |
| S, control | 6.8M | Shakespeare only | 1.6789 | 2000 | +3.634 | 1.6 |
| *ablation kavi ([journal](journal.md))* | *5.8M* | *Shakespeare only, BPE 4096* | *1.6269* | *4000* | | |

![validation bpb of the four runs](img/scale_curves.png)

Three questions, three answers.

1. **Does more data beat the ablation at the same size? Yes, by a lot.** S and its control
   have the same model, the same steps and the same vocabulary size; only the training text
   differs. The control peaks at 1.679 after 2000 steps, then memorises: by step 20,000 its
   train loss is 1.09 nats and its val loss 4.72, and its val bpb has climbed back to 2.049.
   S on the mixed corpus never turns up, and ends at **1.531**: −0.148 against the control's
   best, and −0.096 against the ablation's kavi, a slightly smaller model. Data was the
   bottleneck.
2. **Does size help once data stops being the limit? Yes, with sharply diminishing returns.**
   S → M (2.5× params) buys −0.062. M → L (1.7× params) buys only −0.014. The val−train gap
   says why. On this split, train loss starts *above* val (−0.07 to −0.11 at step 2000), because
   the training mix is harder to predict than Shakespeare alone. So a gap that ends at +0.35 for
   L means L has started memorising its 8.8 passes, where S (+0.01) has not. At 29M params the
   limit is data again. The next lever is more text or more dropout, not more width.
   Overall, L at **1.455** is 0.172 bpb better than the best model of the ablation.
3. **Do induction heads appear? Yes, in every run on the mixed corpus, and in none on
   Shakespeare alone.**

![induction probe during training](img/scale_induction.png)

| run | induction score at step 4000 | final score | final copy gain | best head |
|---|---|---|---|---|
| L | 0.145 | 0.264 (peak 0.281) | 1.00 nats | L6H0 |
| M | 0.153 | 0.223 (peak 0.249) | 1.26 nats | L6H1 |
| S | 0.077 | 0.125 (peak 0.137) | 0.51 nats | L5H5 |
| S, control | 0.018 | 0.018 | −0.26 nats | none |

The uniform baseline is about 1.4%. For comparison, the ablation's kavi scored 3.3%, with a 0.15
nat copy gain, at the end of its 4000 steps ([chapter 13](13-interpretability.md)). S on the
mixed corpus beats that at step 4000. S on Shakespeare only, the same model at the same step,
does not, and never does. So the missing ingredient in the ablation was **diverse data, not
training length**. A model that can memorise its training text doesn't need a general copying
rule. A model that can't memorise has to learn one, because repeated names and phrases within
a document are the cheapest thing left to predict.

**Not a sudden jump, at this resolution.** Olsson et al. describe a phase change. Here L's
score climbs steadily from 0.05 at step 1000 to 0.22 at step 5500, then levels off near 0.27
after step 10,000. Copy gain rises with it and settles near 1 nat. With an eval every 500
steps, a change compressed into fewer than 500 steps would look like this ramp too. What the
curves do rule out is a model that "suddenly" acquires the circuit late in training. All three
runs had most of their final induction score by step 7000, a third of the way through. The
strongest head always sits near the top of the stack (layer 6 of 8, layer 5 of 6; layers count
from 0). That is consistent with the two-layer circuit, which needs a previous-token head
somewhere below it; chapter 13 found one at layer 4 in the ablation's model. We didn't check
where the previous-token heads sit in these models.

**Caveats.** There is one seed per run. The ablation's seed spread was ≤ 0.0012 bpb, so the
data effect (0.148) and S → M (0.062) are far outside noise, while M → L (0.014) probably is too.
The probe uses only 8 sequences, so copy gain is noisy: M ends higher than L there even though
L has the stronger single head. The control's tokenizer is BPE 8192 trained on Shakespeare
alone, so it matches the mixed runs in vocabulary *size*, not merge for merge. The negative copy
gain in the control grows as it memorises. A plausible reading is that a memorising model
predicts a remembered continuation and is surprised when the random sequence repeats instead,
but we haven't tested that.

A sample from L (temperature 0.8, top-k 50, an unedited stretch of
`results/scale/scale-L-29M/samples.txt`):

```
THIRD GENTLEMAN.
’Tis very well.

FIRST GENTLEMAN.
I would a thousand times had never yet a look. But here he comes.

Enter Servant.

SERVANT.
Captain Macbeth, to whom is that?

FIRST GENTLEMAN.
Sir, that’s all one.
```

## Exercises

1. **Tune the guard.** Rebuild with `NGRAM = 6` and with `NGRAM = 12`, and compare
   `drop_shakespeare` in `data/extra_stats.json`. Print ten paragraphs that are dropped at 6 but
   kept at 8. Are they quotations or idioms?
2. **Find the other collaborations.** *Sir Thomas More*, *Edward III* and *Cardenio* (lost) all
   have Shakespeare attribution debates. Search the selection for them. Would `leak_mask` catch
   them if they were present? (Hint: only if Gutenberg #100 contains them.)
3. **Rebalance.** The KJV is 7.3% of the corpus. Write a `--cap` option that truncates any single
   text to at most 2% of the total, retrain the tokenizer, and compare val bpb after 2000 steps.
4. **Check the budget.** When the probe finishes, compute the achieved TFLOP/s for each size from
   the logged tok/s: $6 \cdot N \cdot \text{tok/s}$. Does efficiency rise with model width, as
   claimed above?
