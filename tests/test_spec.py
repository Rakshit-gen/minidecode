import numpy as np
import pytest

from minidecode.generate import generate_cached
from minidecode.kvcache import KVCache, step
from minidecode.model import Config, init
from minidecode.sampling import probs
from minidecode.spec import Stats, speculative

TC = Config(vocab=6, ctx=40, dim=16, layers=2, heads=2)
DC = Config(vocab=6, ctx=40, dim=8, layers=1, heads=2)
TARGET = (init(TC, seed=1), TC)
DRAFT = (init(DC, seed=2), DC)


def sharpen(params, by=40.0):
    """A random model is close to uniform; scale the output so greedy and
    sampling have real preferences to get right."""
    out = dict(params)
    out["lnf.g"] = params["lnf.g"] * by
    return out


T = (sharpen(TARGET[0]), TC)
D = (sharpen(DRAFT[0]), DC)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_output_is_exactly_the_targets(k):
    stats = Stats()
    got = speculative(T, D, [1, 2], 25, k=k, stats=stats)
    assert got == generate_cached(T[0], TC, [1, 2], 25, temperature=0)
    assert stats.tokens == 25 and 1 <= stats.tokens_per_call <= k + 1


def test_a_perfect_draft_is_always_accepted():
    stats = Stats()
    speculative(T, T, [3], 20, k=4, stats=stats)
    assert stats.acceptance == 1.0 and stats.target_calls == 4


def test_mismatched_vocabularies_and_overflow_are_refused():
    other = Config(vocab=7, ctx=40, dim=8, layers=1, heads=2)
    with pytest.raises(ValueError, match="same vocabulary"):
        speculative(T, (init(other), other), [1], 5)
    with pytest.raises(ValueError, match="fit in both contexts"):
        speculative(T, D, [1], 38, k=4)


def test_sampling_keeps_the_targets_distribution():
    t = (sharpen(TARGET[0], 3.0), TC)
    d = (sharpen(DRAFT[0], 3.0), DC)
    prompt = [1, 2]
    # Exact target distribution of the first two tokens.
    cache = KVCache(TC)
    p1 = probs(step(t[0], TC, np.array([prompt]), cache)[0, -1])
    p2 = np.zeros(6)
    for a in range(6):
        c = KVCache(TC)
        p2 += p1[a] * probs(step(t[0], TC, np.array([[*prompt, a]]), c)[0, -1])
    rng = np.random.default_rng(0)
    runs = np.array(
        [speculative(t, d, prompt, 2, k=3, temperature=1.0, rng=rng) for _ in range(3000)]
    )
    for i, p in enumerate((p1, p2)):
        freq = np.bincount(runs[:, i], minlength=6) / len(runs)
        assert 0.5 * np.abs(freq - p).sum() < 0.03, (i, freq, p)
