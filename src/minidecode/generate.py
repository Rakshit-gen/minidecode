"""Decoding loops."""

import numpy as np

from .kvcache import KVCache, step
from .model import Config, forward
from .sampling import pick


def generate_naive(params, cfg: Config, prompt, steps: int, rng=None, temperature=1.0, top_k=0):
    """The simplest decoder: run the whole context through the model for every
    new token. Correct, and quadratic in length; the baseline for the rest."""
    rng = rng or np.random.default_rng(0)
    ids = list(prompt)
    for _ in range(steps):
        window = np.array(ids[-cfg.ctx :])[None]
        logits, _ = forward(params, cfg, window)
        ids.append(int(pick(logits[0, -1], rng, temperature, top_k)))
    return ids[len(prompt) :]


def generate_cached(
    params,
    cfg: Config,
    prompt,
    steps: int,
    rng=None,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    stop: int | None = None,
    stop_count: int = 1,
):
    """Decode with a KV cache: the prompt once, then one token per step, for at
    most `steps` tokens, or until token `stop` has come out `stop_count` times.

    Positions are learned, so the cache can't simply slide. When it is full,
    the last half of the context is run through again as a fresh prompt, and
    decoding carries on from there."""
    rng = rng or np.random.default_rng(0)
    ids = list(prompt)
    cache = KVCache(cfg)
    logits = step(params, cfg, np.array(ids[-cfg.ctx :])[None], cache)[0, -1]
    seen = 0
    for n in range(steps):
        ids.append(int(pick(logits, rng, temperature, top_k, top_p)))
        seen += ids[-1] == stop
        if n + 1 == steps or seen == stop_count:
            break
        if cache.length == cfg.ctx:
            cache = KVCache(cfg)
            logits = step(params, cfg, np.array(ids[-(cfg.ctx // 2) :])[None], cache)[0, -1]
        else:
            logits = step(params, cfg, np.array([[ids[-1]]]), cache)[0, -1]
    return ids[len(prompt) :]
