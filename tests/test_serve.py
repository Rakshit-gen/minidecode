import pytest

from minidecode.generate import generate_cached
from minidecode.kvcache import KVCache
from minidecode.model import Config, init
from minidecode.prefix import PrefixCache
from minidecode.serve import Request, serve

CFG = Config(vocab=9, ctx=24, dim=16, layers=2, heads=2)
P = init(CFG, seed=6)


def requests(lengths):
    return [Request(i, [i % 9, 1], n) for i, n in enumerate(lengths)]


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_every_request_gets_what_it_would_get_alone(policy):
    reqs = requests([3, 12, 5, 9, 2, 7])
    run = serve(P, CFG, reqs, slots=3, policy=policy)
    for r in reqs:
        assert r.out == generate_cached(P, CFG, r.prompt, r.max_new, temperature=0)
    assert run.tokens == 38


def test_continuous_keeps_slots_busier_than_static():
    lengths = [2, 20, 2, 20, 2, 20, 2, 20]
    cont = serve(P, CFG, requests(lengths), slots=4, policy="continuous")
    stat = serve(P, CFG, requests(lengths), slots=4, policy="static")
    assert cont.steps < stat.steps
    assert cont.utilisation > stat.utilisation
    assert cont.mean_latency > 0


def test_requests_that_cannot_fit_are_refused():
    with pytest.raises(ValueError, match="more than the context"):
        serve(P, CFG, [Request(0, [1] * 10, 20)], slots=1)
    with pytest.raises(ValueError, match="unknown policy"):
        serve(P, CFG, [], slots=1, policy="fifo")


def test_a_prefix_cache_changes_nothing_but_the_work():
    header = [1, 2, 3, 4, 5]
    reqs = [Request(i, [*header, i % 9], 6) for i in range(6)]
    pc = PrefixCache(CFG)
    serve(P, CFG, reqs, slots=2, prefix=pc)
    for r in reqs:
        assert r.out == generate_cached(P, CFG, r.prompt, 6, temperature=0)
    assert pc.hits == 0  # every prompt differs in its last token...
    pc2 = PrefixCache(CFG)
    pc2.prefill(P, header, KVCache(CFG))  # primed with the header alone
    serve(P, CFG, [Request(i, [*header, i % 9], 6) for i in range(6)], slots=2, prefix=pc2)
    assert pc2.hits == 6 and pc2.reused_tokens == 30  # ...but all start with the header
