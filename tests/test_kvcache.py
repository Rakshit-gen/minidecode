import numpy as np
import pytest

from minidecode.kvcache import KVCache, step
from minidecode.model import Config, forward, init

CFG = Config(vocab=9, ctx=12, dim=16, layers=2, heads=2)
P = init(CFG, seed=4)
IDS = np.random.default_rng(0).integers(0, 9, (2, 10))


def test_prefill_then_single_steps_match_the_full_forward():
    full, _ = forward(P, CFG, IDS)
    cache = KVCache(CFG, batch=2)
    got = [step(P, CFG, IDS[:, :4], cache)]
    got += [step(P, CFG, IDS[:, i : i + 1], cache) for i in range(4, 7)]
    got.append(step(P, CFG, IDS[:, 7:], cache))  # a block of 3, as a verify pass would
    assert np.allclose(np.concatenate(got, 1), full, atol=1e-5)
    assert cache.length == 10


def test_truncate_rewinds_and_overflow_is_refused():
    cache = KVCache(CFG)
    a = step(P, CFG, IDS[:1, :6], cache)
    cache.truncate(3)
    b = step(P, CFG, IDS[:1, 3:6], cache)
    assert np.allclose(a[:, 3:], b, atol=1e-5)
    with pytest.raises(ValueError, match="won't fit in 12"):
        step(P, CFG, IDS[:1, :7], cache)
    with pytest.raises(ValueError, match="can't truncate"):
        cache.truncate(7)
