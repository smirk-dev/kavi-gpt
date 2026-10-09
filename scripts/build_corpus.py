"""Build a bigger *early-modern English* training corpus to go with Shakespeare.   PLAN phase 8

    python scripts/build_corpus.py select      # catalog -> configs/corpus_early_modern.json
    python scripts/build_corpus.py download    # -> data/raw/pg/<id>.txt  (polite, resumable)
    python scripts/build_corpus.py build       # -> data/extra.txt + stats

Why: the ablation models overfit (21 passes over 1.6M tokens). More text in the same idiom
(Marlowe, Jonson, Fletcher, Webster, Milton, Donne, the 1611 Bible...) should help the model at
*Shakespeare*, which stays the validation set, so bits-per-byte is directly comparable with
the 1.627 of docs/journal.md.

Selection (principled, not hand-picked): from Project Gutenberg's catalog, English texts in
Library of Congress class PR (English literature) whose *first* author was born 1530-1625,
plus a short list of extras (EXTRA_IDS, each with a reason). Every Shakespeare text is excluded.

Leakage guard: Shakespeare *is* the validation set, so a quotation or a collaboration in the
extra corpus would leak it. `build` drops every paragraph that shares any word 8-gram with
the complete works (train AND val), then drops exact-duplicate paragraphs across texts, and
skips a whole text when most of it has already been seen (another edition of the same play).
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "raw" / "pg_catalog.csv"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
SELECTION = ROOT / "configs" / "corpus_early_modern.json"
RAW = ROOT / "data" / "raw" / "pg"
OUT = ROOT / "data" / "extra.txt"
BORN = (1530, 1625)
EXTRA_IDS = {
    10: "King James Bible (1611): the same decade's English, ~4 MB",
}
START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG E(?:-?BOOK|TEXT)[^\n]*\n", re.I)
END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG E(?:-?BOOK|TEXT)", re.I)
NGRAM = 8


def select():
    if not CATALOG.exists():
        CATALOG.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CATALOG_URL, CATALOG)
    picked = []
    for r in csv.DictReader(open(CATALOG, encoding="utf-8")):
        tid = int(r["Text#"])
        if tid in EXTRA_IDS:
            picked.append({"id": tid, "title": r["Title"], "author": r["Authors"], "why": EXTRA_IDS[tid]})
            continue
        if r["Type"] != "Text" or r["Language"] != "en" or not r["LoCC"].startswith("PR"):
            continue
        if "shakespeare" in r["Authors"].lower() or tid == 100:
            continue                                        # validation set: never train on it
        first = r["Authors"].split(";")[0]
        m = re.search(r", (\d{4})\??-", first)
        if m and BORN[0] <= int(m.group(1)) <= BORN[1]:
            picked.append({"id": tid, "title": r["Title"], "author": first.strip()})
    picked.sort(key=lambda p: p["id"])
    SELECTION.write_text(json.dumps({"rule": f"en, LoCC PR*, first author born {BORN[0]}-{BORN[1]}, "
                                             "no Shakespeare; plus extras", "texts": picked},
                                    indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(picked)} texts -> {SELECTION.relative_to(ROOT)}")


def download(delay=1.5):
    texts = json.loads(SELECTION.read_text(encoding="utf-8"))["texts"]
    RAW.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(texts):
        dst = RAW / f"{t['id']}.txt"
        if dst.exists():
            continue
        url = f"https://www.gutenberg.org/cache/epub/{t['id']}/pg{t['id']}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "kavi-gpt corpus builder (learning project)"})
        for attempt in range(3):                                  # IncompleteRead happens; retry
            try:
                data = urllib.request.urlopen(req, timeout=60).read()
                dst.write_bytes(data)
                print(f"[{i + 1}/{len(texts)}] {t['id']}: {len(data) / 1e3:.0f} KB  {t['title'][:60]}", flush=True)
                break
            except Exception as e:                                # some ids have no plain-text file
                print(f"[{i + 1}/{len(texts)}] {t['id']}: attempt {attempt + 1} FAILED ({e})", flush=True)
                time.sleep(delay * 4 * (attempt + 1))
        time.sleep(delay)


def strip_gutenberg(raw):
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    s, e = START.search(text), END.search(text)
    if not s or not e:
        return None
    return text[s.end():e.start()].strip("\n") + "\n"


def words(s):
    return re.findall(r"[a-z]+", s.lower())


def ngrams(ws):
    return {hash(tuple(ws[i:i + NGRAM])) for i in range(len(ws) - NGRAM + 1)}


def build():
    shakes = (ROOT / "data" / "shakespeare.txt").read_text(encoding="utf-8")
    banned = ngrams(words(shakes))                                 # train AND val
    texts = json.loads(SELECTION.read_text(encoding="utf-8"))["texts"]
    seen, kept_texts, stats = set(), [], {"texts": 0, "skipped_dupe_text": 0, "no_markers": 0,
                                          "missing": 0, "paras": 0, "drop_shakespeare": 0,
                                          "drop_dupe": 0}
    for t in texts:
        p = RAW / f"{t['id']}.txt"
        if not p.exists():
            stats["missing"] += 1
            continue
        body = strip_gutenberg(p.read_text(encoding="utf-8", errors="replace"))
        if body is None:
            stats["no_markers"] += 1
            continue
        paras = [q for q in re.split(r"\n\s*\n", body) if q.strip()]
        keys = [" ".join(words(q)) for q in paras]
        if sum(k in seen for k in keys if k) > 0.5 * max(1, sum(1 for k in keys if k)):
            stats["skipped_dupe_text"] += 1                        # another edition of something we have
            continue
        out = []
        for q, k in zip(paras, keys):
            stats["paras"] += 1
            if k and k in seen:
                stats["drop_dupe"] += 1
                continue
            if ngrams(k.split()) & banned:
                stats["drop_shakespeare"] += 1
                continue
            seen.add(k)
            out.append(q.strip("\n"))
        if out:
            kept_texts.append("\n\n".join(out) + "\n")
            stats["texts"] += 1
    corpus = "\n\n\n".join(kept_texts)                              # blank lines between documents
    OUT.write_text(corpus, encoding="utf-8")
    stats["chars"] = len(corpus)
    stats["bytes"] = len(corpus.encode("utf-8"))
    stats["x_shakespeare_train"] = round(stats["bytes"] / len((ROOT / "data" / "train.txt")
                                                            .read_text(encoding="utf-8").encode("utf-8")), 2)
    print(json.dumps(stats, indent=1))
    (ROOT / "data" / "extra_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "download", "build", "all"])
    a = ap.parse_args()
    if a.step in ("select", "all"):
        select()
    if a.step in ("download", "all"):
        download()
    if a.step in ("build", "all"):
        build()


if __name__ == "__main__":
    sys.exit(main())
