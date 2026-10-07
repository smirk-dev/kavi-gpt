"""Inference-path equivalences: the KV cache must not change a single logit."""
import numpy as np
import pytest

from kavi import backend as B
from kavi.model import GPT, GPTConfig, sample


def make(preset):
    cfg = GPTConfig.preset(preset, vocab_size=17, n_embd=16, n_head=2, block_size=12, n_layer=2)
    model = GPT(cfg, seed=1)
    rng = np.random.default_rng(0)
    for _, p in model.named_params():
        p.data += B.asarray(rng.normal(scale=0.3, size=p.data.shape))
    return model.eval()


@pytest.mark.parametrize("preset", ["gpt2", "kavi"])
def test_kv_cache_matches_full_forward(preset):
    model = make(preset)
    ids = np.random.default_rng(2).integers(0, 17, size=(2, 12))
    full, _ = model.forward(ids)
    for blk in model.blocks:
        blk.attn.reset_cache(2)
    # prefill 5 tokens at once, then feed the rest one at a time
    step_logits = [model._forward_cached(ids[:, :5], 0)]
    for t in range(5, 12):
        step_logits.append(model._forward_cached(ids[:, t:t + 1], t))
    np.testing.assert_allclose(step_logits[0], full[:, 4], rtol=1e-9, atol=1e-12)
    for i, t in enumerate(range(5, 12), start=1):
        np.testing.assert_allclose(step_logits[i], full[:, t], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("preset", ["gpt2", "kavi"])
def test_greedy_generation_same_with_and_without_cache(preset):
    model = make(preset)
    prompt = np.array([[1, 2, 3]])
    a = model.generate(prompt, 9, temperature=0, use_cache=True)
    b = model.generate(prompt, 9, temperature=0, use_cache=False)
    np.testing.assert_array_equal(a, b)


def test_long_generation_runs_past_block_size():
    model = make("kavi")
    out = model.generate(np.array([[0]]), 40, temperature=1.0, rng=np.random.default_rng(0))
    assert out.shape == (1, 41)


def test_sampling_filters():
    logits = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    rng = np.random.default_rng(0)
    assert sample(logits, temperature=0)[0] == 0
    picks = {int(sample(logits, top_k=2, rng=rng)[0]) for _ in range(200)}
    assert picks == {0, 1}
    picks = {int(sample(logits, top_p=0.7, rng=rng)[0]) for _ in range(200)}
    assert picks == {0, 1}                      # 0.5 < 0.7, 0.5+0.3 >= 0.7 -> keep 2 tokens
    picks = {int(sample(logits, top_p=0.4, rng=rng)[0]) for _ in range(50)}
    assert picks == {0}
