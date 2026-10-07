"""Download Shakespeare's complete works and make a train/val split.

    python scripts/prepare_data.py

Source: Project Gutenberg eBook #100 (public domain). We strip Gutenberg's header and
licence, normalise line endings, then split into 100 contiguous shards (cut at line breaks);
every 10th shard is validation. Interleaving keeps every genre in both splits while the
validation text is still never seen in training.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def main():
    raw = DATA / "raw" / "pg100.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "kavi-gpt (educational)"})
        raw.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    text = raw.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    s, e = text.find(START), text.find(END)
    if s < 0 or e < 0:
        sys.exit("could not find Gutenberg START/END markers — has the file format changed?")
    text = text[text.index("\n", s) + 1:e].strip() + "\n"
    (DATA / "shakespeare.txt").write_text(text, encoding="utf-8")

    n_shards, bounds, pos = 100, [0], 0
    for i in range(1, n_shards):
        pos = text.find("\n", max(len(text) * i // n_shards, pos)) + 1
        bounds.append(pos)
    bounds.append(len(text))
    shards = [text[bounds[i]:bounds[i + 1]] for i in range(n_shards)]
    val = "".join(s for i, s in enumerate(shards) if i % 10 == 9)
    train = "".join(s for i, s in enumerate(shards) if i % 10 != 9)
    (DATA / "train.txt").write_text(train, encoding="utf-8")
    (DATA / "val.txt").write_text(val, encoding="utf-8")

    print(f"complete works: {len(text):,} chars, {text.count(chr(10)):,} lines, "
          f"{len(set(text))} distinct chars")
    print(f"train: {len(train):,} chars   val: {len(val):,} chars")


if __name__ == "__main__":
    main()
