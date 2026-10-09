"""scripts/evaluate.py: window scoring covers every token once and agrees with forward();
the n-gram index finds what is there and nothing else; line kinds parse Gutenberg layout."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import NgramIndex, line_kinds, token_scores  # noqa: E402
from kavi.model import GPT, GPTConfig  # noqa: E402


@pytest.fixture(scope="module")
def model():
    m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0), seed=1)
    m.eval()
    return m


@pytest.mark.parametrize("N,stride", [(17, 16), (40, 8), (41, 16), (100, 4), (5, 8)])
def test_every_target_scored_once(model, N, stride):
    toks = np.random.default_rng(N).integers(0, 50, N)
    out = token_scores(model, toks, 16, stride)
    assert not np.isnan(out["nll"]).any()
    # after the first window every token sees at least T - stride of context
    assert (out["ctx"][16:] >= 16 - stride).all()


def test_nonoverlapping_matches_forward(model):
    toks = np.random.default_rng(0).integers(0, 50, 3 * 16 + 1)
    out = token_scores(model, toks, 16, stride=16)
    _, loss = model.forward(toks[:-1].reshape(3, 16), toks[1:].reshape(3, 16))
    assert out["nll"].mean() == pytest.approx(float(loss), rel=1e-5)


def test_sliding_window_token_matches_its_window(model):
    toks = np.random.default_rng(1).integers(0, 50, 40)
    out = token_scores(model, toks, 16, stride=8)
    # target toks[30] is scored by the window starting at 16 (positions 8..15 of each later
    # window are new), at position 13 of that window
    logits, _ = model.forward(toks[16:32][None])
    lg = np.asarray(logits[0, 13], dtype=np.float64)
    nll = np.log(np.exp(lg - lg.max()).sum()) + lg.max() - lg[toks[30]]
    assert out["nll"][29] == pytest.approx(nll, rel=1e-6)
    assert out["ctx"][29] == 13


def test_ngram_index():
    train = "the quick brown fox jumps over the lazy dog and then the cat sat on the mat today"
    idx = NgramIndex([train])
    frac, run = idx.overlap("brown fox jumps over the lazy dog and then")
    assert frac == 1.0 and run == 9
    frac, run = idx.overlap("purple elephants dance quietly beneath silver moons every single night")
    assert frac == 0.0 and run == 0
    frac, run = idx.overlap("zebra the quick brown fox jumps over the lazy zebra")
    assert run == 8 and 0 < frac < 1


def test_line_kinds():
    text = "ROMEO.\nBut soft!\n\n[_Enter Juliet\nabove._]\nJULIET.\nAy me.\nEnter Nurse."
    assert line_kinds(text) == ["speaker", "dialogue", "blank", "stage", "stage",
                                "speaker", "dialogue", "stage"]
