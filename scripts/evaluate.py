"""Test a trained checkpoint thoroughly: a battery of measurements beyond the training log.

    python scripts/evaluate.py ckpt/scale-L-29M.npz [ckpt/...] [--out results/eval] [--quick]

The training log's val bpb is an *estimate* (40 random batches). This script measures, per
checkpoint:

  1. exact val bpb over every token of the val split, two ways: non-overlapping 256-token
     windows (early tokens get little context) and a sliding window (every token after the
     first window sees >= T/2 tokens of context);
  2. loss by position in the context window, i.e. how much the model uses its context;
  3. bpb by kind of line: speaker names, stage directions, dialogue;
  4. top-1 / top-5 accuracy and calibration (expected calibration error, ECE);
  5. bpb on held-out texts at growing distance from Shakespeare (data/heldout/, see HELDOUT);
  6. memorisation: greedy continuation of 64-token passages from the Shakespeare *train*
     split vs the unseen *val* split. A model that memorised reproduces train text verbatim;
  7. novelty: what fraction of a sample's word 8-grams occur in the training text, and the
     longest verbatim run, against the same numbers for real unseen Shakespeare;
  8. form: do sampled speaker names exist in the plays, are stage-direction brackets closed;
  9. KV cache: greedy decoding with and without the cache must give the same tokens;
 10. the per-head induction and previous-token map (kavi/probes.py), with 32 sequences.

Results go to <out>/<checkpoint name>.json; a markdown summary is printed. See docs/journal.md.
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kavi.checkpoint import load_model  # noqa: E402
from kavi.data import TokenData  # noqa: E402
from kavi.probes import induction, induction_batch  # noqa: E402

DATA = ROOT / "data"
HELD = DATA / "heldout"
# (key, Gutenberg id, label). None of these authors passes the corpus rule (first author born
# 1530-1625), so no model has seen them. Ordered by distance from Shakespeare.
HELDOUT = [
    ("dryden", 2062, "Dryden, All for Love (1677): a verse tragedy, the same story as Antony and Cleopatra"),
    ("austen", 1342, "Austen, Pride and Prejudice (1813): prose novel"),
    ("wells", 35, "Wells, The Time Machine (1895): prose novel"),
]
HELD_CHARS = 150_000          # per text, after skipping front matter
POS_BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 128), (128, 256)]
SPEAKER = re.compile(r"^[A-Z][A-Z0-9 ,.'’&-]*[A-Z]\.$")
STAGE_START = ("Enter ", "Exit", "Exeunt", "Re-enter", "SCENE", "ACT ", "Flourish", "Alarum")


# ---------------------------------------------------------------- scoring

def token_scores(model, toks, T, stride, batch=8):
    """Score every target toks[1:], each with as much left context as a window allows.

    Windows start every `stride` tokens. The first window scores all its targets; each later
    one scores only the targets the previous window didn't reach, so every token is scored
    exactly once and (after the first window) with at least T - stride tokens of context.
    Returns per-target arrays: nll (nats), ctx (tokens of context), p1 (top-1 probability),
    hit1, hit5 (target is the top-1 / in the top-5).
    """
    N = len(toks)
    out = {k: np.full(N - 1, np.nan) for k in ("nll", "ctx", "p1", "hit1", "hit5")}
    jobs, s = [], 0
    while True:
        L = min(T, N - 1 - s)
        lo = 0 if s == 0 else T - stride        # first target position (within window) to keep
        if L <= lo:
            break
        jobs.append((s, L, lo))
        if s + L >= N - 1:
            break
        s += stride
    by_len = {}
    for j in jobs:
        by_len.setdefault(j[1], []).append(j)
    for L, group in by_len.items():
        for i in range(0, len(group), batch):
            chunk = group[i:i + batch]
            x = np.stack([toks[s:s + L] for s, _, _ in chunk])
            y = np.stack([toks[s + 1:s + L + 1] for s, _, _ in chunk])
            logits, _ = model.forward(x)
            logits = np.asarray(logits, dtype=np.float64)
            m = logits.max(-1, keepdims=True)
            logz = m[..., 0] + np.log(np.exp(logits - m).sum(-1))
            tgt = np.take_along_axis(logits, y[..., None], -1)[..., 0]
            nll = logz - tgt
            p1 = np.exp(m[..., 0] - logz)
            rank = (logits > tgt[..., None]).sum(-1)          # how many beat the target
            for b, (s, _, lo) in enumerate(chunk):
                sl = slice(s + lo, s + L)                     # target index j-1 for toks[j]
                out["nll"][sl] = nll[b, lo:]
                out["ctx"][sl] = np.arange(lo, L)
                out["p1"][sl] = p1[b, lo:]
                out["hit1"][sl] = rank[b, lo:] == 0
                out["hit5"][sl] = rank[b, lo:] < 5
    return out


def bpb(nll, nbytes):
    return float(nll.sum() / math.log(2) / nbytes)


def ece(p1, hit, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p1, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(p1[sel].mean() - hit[sel].mean())
    return float(err)


# ---------------------------------------------------------------- line categories

def line_kinds(text):
    """A kind per line: speaker / stage / blank / dialogue. Stage directions in Gutenberg's
    Shakespeare are [_..._] (possibly over several lines) or start with Enter/Exit/SCENE..."""
    kinds, in_br = [], False
    for line in text.split("\n"):
        s = line.strip()
        if in_br or s.startswith("[_") or s.startswith(STAGE_START):
            kinds.append("stage")
            in_br = (in_br or "[_" in s) and "_]" not in s
        elif not s:
            kinds.append("blank")
        elif SPEAKER.match(s):
            kinds.append("speaker")
        else:
            kinds.append("dialogue")
    return kinds


def bpb_by_kind(toks, nll, vocab, text):
    """Attribute each target token's bits to the kind of line its first non-newline byte is in."""
    kinds = line_kinds(text)
    data = text.encode("utf-8")
    line_of = np.cumsum(np.frombuffer(data, dtype=np.uint8) == 10)   # line index per byte
    line_of = np.concatenate([[0], line_of[:-1]])
    tot = {}
    pos = len(vocab[toks[0]])
    for j in range(1, len(toks)):
        tb = vocab[toks[j]]
        k = len(tb) - len(tb.lstrip(b"\n"))
        kind = "newline" if k == len(tb) else kinds[line_of[min(pos + k, len(data) - 1)]]
        bits, nb = tot.get(kind, (0.0, 0))
        tot[kind] = (bits + nll[j - 1] / math.log(2), nb + len(tb))
        pos += len(tb)
    return {k: {"bpb": round(b / n, 4), "bytes": n} for k, (b, n) in sorted(tot.items())}


# ---------------------------------------------------------------- n-gram overlap

class NgramIndex:
    """Sorted hashes of every word 8-gram in a text, built in chunks to keep memory low."""

    def __init__(self, texts, n=8):
        self.n, self.ids = n, {}
        parts = []
        for text in texts:
            for i in range(0, len(text), 4_000_000):
                parts.append(self.encode(text[i:i + 4_000_000], grow=True))
        self.hashes = np.unique(self.hash(np.concatenate(parts)))

    def encode(self, text, grow=False):
        out = []
        for w in re.findall(r"[a-z]+", text.lower()):
            i = self.ids.get(w)
            if i is None:
                i = len(self.ids) + 1
                if grow:
                    self.ids[w] = i
            out.append(i)
        return np.array(out, dtype=np.uint64)

    def hash(self, ids):
        if len(ids) < self.n:
            return np.zeros(0, dtype=np.uint64)
        h = np.zeros(len(ids) - self.n + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for k in range(self.n):
                h = h * np.uint64(1_000_003) + ids[k:len(ids) - self.n + 1 + k]
        return h

    def overlap(self, text):
        """(fraction of the text's 8-grams found, longest verbatim run in words)."""
        h = self.hash(self.encode(text))
        if len(h) == 0:
            return 0.0, 0
        i = np.searchsorted(self.hashes, h).clip(0, len(self.hashes) - 1)
        hit = self.hashes[i] == h
        run = best = 0
        for x in hit:
            run = run + 1 if x else 0
            best = max(best, run)
        return float(hit.mean()), (best + self.n - 1) if best else 0


# ---------------------------------------------------------------- held-out texts

def heldout_texts():
    from build_corpus import strip_gutenberg
    HELD.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, gid, label in HELDOUT:
        raw = HELD / f"{gid}.txt"
        if not raw.exists():
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "kavi-gpt evaluation (learning project)"})
            raw.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            time.sleep(1.5)
        text = strip_gutenberg(raw.read_text(encoding="utf-8", errors="replace"))
        skip = min(20_000, len(text) // 10)                  # front matter, prefaces
        out[key] = (label, text[skip:skip + HELD_CHARS])
    docs = "".join((ROOT / "docs" / f).read_text(encoding="utf-8")
                   for f in ("00-overview.md", "09-training.md", "11-gpu.md")
                   if (ROOT / "docs" / f).exists())
    out["docs"] = ("This repo's docs (2026): modern technical English, Markdown, maths", docs[:HELD_CHARS])
    return out


# ---------------------------------------------------------------- generation tests

def passages(text, n, rng, chars=600):
    """n random (prompt_text, continuation_text) pairs cut at line starts."""
    starts = [m.start() + 1 for m in re.finditer("\n", text[:-2 * chars])]
    pick = rng.choice(len(starts), size=n, replace=False)
    out = []
    for k in pick:
        a = starts[k]
        # cut the prompt at a line end ~400 chars in, so the continuation starts a new line
        cut = text.find("\n", a + 400) + 1
        out.append((text[a:cut], text[cut:cut + 300]))
    return out


def memorisation(model, tok, text, rng, n=100, P=64, G=32):
    """Greedy-continue n passages; how many characters match the real continuation?"""
    pairs = passages(text, n, rng)
    ids = []
    for prompt, _ in pairs:
        t = tok.encode(prompt)
        ids.append(t[-P:] if len(t) >= P else [tok.encode("\n")[0]] * (P - len(t)) + t)
    gen = np.asarray(model.generate(np.array(ids), G, temperature=0))[:, P:]
    match = []
    for g, (_, ref) in zip(gen, pairs):
        out = tok.decode(g.tolist())
        k = 0
        while k < min(len(out), len(ref)) and out[k] == ref[k]:
            k += 1
        match.append(k)
    match = np.array(match)
    return {"mean_chars": round(float(match.mean()), 1), "median_chars": float(np.median(match)),
            "frac_ge_50": round(float((match >= 50).mean()), 3),
            "frac_ge_100": round(float((match >= 100).mean()), 3)}


def form_checks(samples, speakers):
    names, real, opened, closed = 0, 0, 0, 0
    for s in samples:
        for line in s.split("\n"):
            t = line.strip()
            if SPEAKER.match(t) and len(t) < 40:
                names += 1
                real += t in speakers
        opened += s.count("[_")
        closed += s.count("_]")
    return {"speaker_lines": names, "real_speaker_frac": round(real / max(names, 1), 3),
            "brackets_open": opened, "brackets_close": closed}


# ---------------------------------------------------------------- main

def evaluate(path, quick=False, index_cache={}):
    t0 = time.time()
    model, cfg = load_model(path)
    model.eval()
    T = cfg["model"]["block_size"]
    data = TokenData(ROOT / cfg["data"])
    tok = data.tokenizer()
    vocab = tok.vocab
    val = data.val[:20_001] if quick else data.val
    val_bytes = sum(len(vocab[t]) for t in val[1:])
    val_text = b"".join(vocab[t] for t in val).decode("utf-8", errors="replace")
    res = {"checkpoint": str(path), "params": model.num_params(), "data": cfg["data"],
           "logged_best_val_bpb": round(data.bpb(json.loads(str(np.load(path)["meta"]))["best_val"]), 4)}
    log = print

    # 1-4: the val split
    full = token_scores(model, val, T, stride=T)
    slide = token_scores(model, val, T, stride=T // 2)
    res["val"] = {
        "tokens": int(len(val) - 1), "bytes": int(val_bytes),
        "bpb_nonoverlap": round(bpb(full["nll"], val_bytes), 4),
        "bpb_sliding": round(bpb(slide["nll"], val_bytes), 4),
        "top1": round(float(slide["hit1"].mean()), 4), "top5": round(float(slide["hit5"].mean()), 4),
        "mean_p_top1": round(float(slide["p1"].mean()), 4),
        "ece": round(ece(slide["p1"], slide["hit1"]), 4),
    }
    res["loss_by_position"] = {f"{a}-{b - 1}": round(float(full["nll"][(full["ctx"] >= a) & (full["ctx"] < b)].mean()), 4)
                               for a, b in POS_BUCKETS}
    res["bpb_by_kind"] = bpb_by_kind(val, slide["nll"], vocab, val_text)
    log(f"[{Path(path).stem}] val done {time.time() - t0:.0f}s: {res['val']}")

    # 5: held-out texts
    res["heldout"] = {}
    if not quick:
        for key, (label, text) in heldout_texts().items():
            ids = np.array(tok.encode(text), dtype=np.int64)
            sc = token_scores(model, ids, T, stride=T // 2)
            nb = sum(len(vocab[t]) for t in ids[1:])
            res["heldout"][key] = {"label": label, "tokens": int(len(ids)),
                                   "bytes_per_token": round(nb / (len(ids) - 1), 3),
                                   "bpb": round(bpb(sc["nll"], nb), 4)}
        log(f"[{Path(path).stem}] held-out done {time.time() - t0:.0f}s: "
            + ", ".join(f"{k} {v['bpb']}" for k, v in res["heldout"].items()))

    # 6: memorisation
    rng = np.random.default_rng(7)
    shakes_train = (DATA / "train.txt").read_text(encoding="utf-8")
    n = 20 if quick else 100
    res["memorisation"] = {"train": memorisation(model, tok, shakes_train, rng, n=n),
                           "val": memorisation(model, tok, (DATA / "val.txt").read_text(encoding="utf-8"), rng, n=n)}
    log(f"[{Path(path).stem}] memorisation {res['memorisation']}")

    # 7-8: samples - novelty and form
    nl = tok.encode("\n")
    srng = np.random.default_rng(11)
    ns, length = (2, 100) if quick else (8, 400)
    gen = np.asarray(model.generate(np.array([nl] * ns), length, temperature=0.8, top_k=50, rng=srng))
    samples = [tok.decode(g[len(nl):].tolist()) for g in gen]
    mixed = "x" in Path(cfg["data"]).name
    key = "mixed" if mixed else "shakes"
    if key not in index_cache:
        texts = [shakes_train] + ([(DATA / "extra.txt").read_text(encoding="utf-8")] if mixed else [])
        index_cache[key] = NgramIndex(texts)
    idx = index_cache[key]
    ov = [idx.overlap(s) for s in samples]
    val_chunks = [val_text[i:i + len(samples[0])] for i in range(0, len(val_text), len(val_text) // ns)][:ns]
    vo = [idx.overlap(c) for c in val_chunks]
    speakers = {m.group(0) for m in re.finditer(r"(?m)^[A-Z][A-Z0-9 ,.'’&-]*[A-Z]\.$", shakes_train)}
    res["samples"] = {
        "n": ns, "tokens_each": length, "train_text": key,
        "ngram8_in_train": round(float(np.mean([o[0] for o in ov])), 4),
        "longest_copy_words": int(max(o[1] for o in ov)),
        "real_val_ngram8_in_train": round(float(np.mean([o[0] for o in vo])), 4),
        "real_val_longest_copy_words": int(max(o[1] for o in vo)),
        **form_checks(samples, speakers),
        "first_sample": samples[0][:600],
    }
    log(f"[{Path(path).stem}] samples {({k: v for k, v in res['samples'].items() if k != 'first_sample'})}")

    # 9: KV cache == no cache (greedy, within one context window)
    prompt = np.array([val[:64]])
    a = np.asarray(model.generate(prompt, 64, temperature=0, use_cache=True))
    b = np.asarray(model.generate(prompt, 64, temperature=0, use_cache=False))
    res["kv_cache_equal"] = bool((a == b).all())

    # 10: induction map
    pr = induction(model, induction_batch(data.val, min(50, T // 2), 32, np.random.default_rng(4321)))
    ind, prev = pr["ind"], pr["prev"]
    top = lambda m: [{"head": f"L{l}H{h}", "score": round(float(m[l, h]), 3)}
                     for l, h in zip(*np.unravel_index(np.argsort(-m, axis=None)[:3], m.shape))]
    res["induction"] = {"copy_gain": round(pr["loss_first"] - pr["loss_second"], 3),
                        "loss_first": round(pr["loss_first"], 3), "loss_second": round(pr["loss_second"], 3),
                        "top_induction": top(ind), "top_prev_token": top(prev),
                        "ind_by_layer": [round(float(x), 3) for x in ind.max(1)],
                        "prev_by_layer": [round(float(x), 3) for x in prev.max(1)]}
    res["seconds"] = round(time.time() - t0)
    log(f"[{Path(path).stem}] kv {res['kv_cache_equal']} induction {res['induction']} ({res['seconds']}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--quick", action="store_true", help="20k val tokens, no held-out texts")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for p in args.checkpoints:
        r = evaluate(p, quick=args.quick)
        (out / f"{Path(p).stem}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
