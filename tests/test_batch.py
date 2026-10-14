import numpy as np
import pytest

from minidecode.batch import BatchCache, decode
from minidecode.generate import generate_cached
from minidecode.model import Config, init

CFG = Config(vocab=9, ctx=16, dim=16, layers=2, heads=2)
P = init(CFG, seed=5)


def greedy_batch(prompts, steps):
    cache = BatchCache(CFG, len(prompts))
    logits = np.stack([cache.prefill(P, i, p) for i, p in enumerate(prompts)])
    out = [[] for _ in prompts]
    for _ in range(steps):
        ids = logits.argmax(-1)
        for row, t in zip(out, ids, strict=True):
            row.append(int(t))
        logits = decode(P, CFG, ids, cache)
    return out


def test_ragged_batch_matches_one_at_a_time():
    prompts = [[1], [2, 3, 4, 5], [6, 7]]
    got = greedy_batch(prompts, 8)
    want = [generate_cached(P, CFG, p, 8, temperature=0) for p in prompts]
    assert got == want


def test_idle_slots_are_left_alone_and_full_slots_refused():
    cache = BatchCache(CFG, 2)
    cache.prefill(P, 0, [1, 2])
    cache.prefill(P, 1, [3])
    before = [k[1].copy() for k in cache.k]
    decode(P, CFG, np.array([4, 4]), cache, active=[True, False])
    assert list(cache.lengths) == [3, 1]
    assert all(np.array_equal(b, k[1]) for b, k in zip(before, cache.k, strict=True))
    cache.prefill(P, 0, list(range(1, 9)) * 2)
    with pytest.raises(ValueError, match="slot is full"):
        decode(P, CFG, np.array([1, 1]), cache)
