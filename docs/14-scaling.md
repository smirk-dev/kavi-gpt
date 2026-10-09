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

## 7. Results

*Pending: the long run (Kaggle v5, pushed 2026-10-09 11:25 IST). This section and the [journal](journal.md) get the
numbers when they land.* The questions they answer:

1. **Does more data beat the ablation at the same size?** S has about the same size as the
   ablation's kavi (6.8M vs 5.8M; the difference is the larger vocabulary). If its val bpb
   comes in clearly under 1.627, data was the bottleneck.
2. **Does size help once data stops being the limit?** S vs M vs L, at equal steps.
3. **Do induction heads appear?** At 5.8M params and 4000 steps, `scripts/induction.py` found
   none ([chapter 13](13-interpretability.md)). Olsson et al. 2022 saw them form in a sudden
   "phase change" early in training, in models with at least two layers. That suggests our
   earlier models were too short-trained or too memorisation-bound. A longer run on more diverse
   text is the experiment. To see *when* the heads form, and not just whether they exist at the
   end, `scale_base.json` sets `train.induction_probe: true`. At every eval, `train.py` runs the
   probe from `kavi/probes.py` (the same code as `scripts/induction.py`) on one fixed batch of
   repeated sequences. It logs `ind_score` (the strongest head's attention on the copy target)
   and `copy_gain` (first-copy loss minus second-copy loss) to `log.jsonl`. A phase change
   would show as a sudden jump in both, from the ~1.4% uniform baseline toward tens of percent.

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
