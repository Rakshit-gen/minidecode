import numpy as np
import pytest

from minidecode.kvcache import KVCache, step
from minidecode.model import Config, init
from minidecode.prefix import PrefixCache

CFG = Config(vocab=9, ctx=24, dim=16, layers=2, heads=2)
P = init(CFG, seed=7)
HEADER = [1, 2, 3, 4, 5, 6]


def plain(prompt):
    c = KVCache(CFG)
    return step(P, CFG, np.array([prompt]), c)[0, -1], c


def test_reused_prefix_gives_the_same_cache_and_logits():
    pc = PrefixCache(CFG)
    pc.prefill(P, HEADER, KVCache(CFG))
    for prompt in ([*HEADER, 7, 8], HEADER):
        cache = KVCache(CFG)
        logits = pc.prefill(P, prompt, cache)
        want, wc = plain(prompt)
        assert np.allclose(logits, want, atol=1e-5)
        assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(cache.k, wc.k, strict=True))
        assert cache.length == len(prompt)
    assert (pc.misses, pc.hits, pc.reused_tokens) == (1, 2, 12)


def test_oldest_entries_are_dropped_and_short_prompts_not_stored():
    pc = PrefixCache(CFG, max_entries=2)
    for first in (1, 2, 3):
        pc.prefill(P, [first, 0, 0, 0], KVCache(CFG))
    assert [k[0] for k in pc.entries] == [2, 3]
    pc.prefill(P, [5, 5], KVCache(CFG))
    assert len(pc.entries) == 2
    cache = KVCache(CFG)
    cache.length = 1
    with pytest.raises(ValueError, match="empty cache"):
        pc.prefill(P, [1], cache)
