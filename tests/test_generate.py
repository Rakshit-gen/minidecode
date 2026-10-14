import numpy as np

from minidecode.generate import generate_cached, generate_naive
from minidecode.model import Config, init

CFG = Config(vocab=9, ctx=8, dim=16, heads=2)
P = init(CFG, seed=3)


def test_naive_generation_is_seeded_and_slides_past_the_context():
    a = generate_naive(P, CFG, [1, 2, 3], 20, np.random.default_rng(5))
    b = generate_naive(P, CFG, [1, 2, 3], 20, np.random.default_rng(5))
    assert a == b and len(a) == 20 and all(0 <= t < 9 for t in a)


def test_cached_matches_naive_within_the_context():
    for temperature in (0, 1.0):
        a = generate_naive(P, CFG, [1, 2], 6, np.random.default_rng(7), temperature)
        b = generate_cached(P, CFG, [1, 2], 6, np.random.default_rng(7), temperature)
        assert a == b


def test_cached_keeps_going_past_a_full_cache():
    out = generate_cached(P, CFG, [1, 2, 3], 40, np.random.default_rng(1))
    assert len(out) == 40


def test_stops_after_the_stop_token_count():
    free = generate_cached(P, CFG, [1], 60, np.random.default_rng(2))
    stop = free[5]
    want = [i for i, t in enumerate(free) if t == stop][:2]
    got = generate_cached(P, CFG, [1], 60, np.random.default_rng(2), stop=stop, stop_count=2)
    assert got == free[: want[1] + 1]
