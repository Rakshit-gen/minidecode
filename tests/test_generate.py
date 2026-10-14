import numpy as np

from minidecode.generate import generate_naive
from minidecode.model import Config, init

CFG = Config(vocab=9, ctx=8, dim=16, heads=2)
P = init(CFG, seed=3)


def test_naive_generation_is_seeded_and_slides_past_the_context():
    a = generate_naive(P, CFG, [1, 2, 3], 20, np.random.default_rng(5))
    b = generate_naive(P, CFG, [1, 2, 3], 20, np.random.default_rng(5))
    assert a == b and len(a) == 20 and all(0 <= t < 9 for t in a)
