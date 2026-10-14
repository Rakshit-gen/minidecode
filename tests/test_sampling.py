import numpy as np

from minidecode.sampling import pick, probs


def test_greedy_top_k_and_temperature():
    logits = np.array([1.0, 3.0, 2.0, 0.0])
    rng = np.random.default_rng(0)
    assert pick(logits, rng, temperature=0) == 1
    p = probs(logits, top_k=2)
    assert p[0] == p[3] == 0 and np.isclose(p.sum(), 1)
    assert probs(logits, temperature=0.1)[1] > 0.99


def test_sampling_follows_the_distribution():
    logits = np.log(np.array([[0.1, 0.6, 0.3]]).repeat(20000, 0))
    draws = pick(logits, np.random.default_rng(1))
    freq = np.bincount(draws, minlength=3) / len(draws)
    assert np.allclose(freq, [0.1, 0.6, 0.3], atol=0.01)
