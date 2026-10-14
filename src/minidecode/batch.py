"""Decoding many sequences at once.

Each row of a `BatchCache` is a slot with its own length, so sequences of
different lengths share one forward pass per step, and a finished row can be
refilled with a new request while the others keep going. That is what makes
continuous batching possible.
"""

import numpy as np

from .kvcache import KVCache, step
from .layers import gelu, heads, layernorm, merge, softmax
from .model import Config
from .sampling import pick


class BatchCache:
    def __init__(self, cfg: Config, slots: int, dtype=np.float32):
        self.cfg = cfg
        shape = (slots, cfg.heads, cfg.ctx, cfg.dim // cfg.heads)
        self.k = [np.zeros(shape, dtype) for _ in range(cfg.layers)]
        self.v = [np.zeros(shape, dtype) for _ in range(cfg.layers)]
        self.lengths = np.zeros(slots, dtype=np.int64)

    @property
    def slots(self) -> int:
        return len(self.lengths)

    def prefill(self, params, slot: int, prompt, prefix=None) -> np.ndarray:
        """Run a prompt into one slot, replacing whatever it held; returns the
        logits for the token after it. With a PrefixCache, a known start of
        the prompt is copied instead of recomputed."""
        one = KVCache(self.cfg)
        if prefix is not None:
            last = prefix.prefill(params, prompt, one)
        else:
            last = step(params, self.cfg, np.asarray(prompt)[None], one)[0, -1]
        n = one.length
        for i in range(self.cfg.layers):
            self.k[i][slot, :, :n] = one.k[i][0, :, :n]
            self.v[i][slot, :, :n] = one.v[i][0, :, :n]
        self.lengths[slot] = n
        return last


def decode(params, cfg: Config, ids: np.ndarray, cache: BatchCache, active=None) -> np.ndarray:
    """One new token per slot, each at its own position. `active` (bool per
    slot) leaves idle slots untouched. Returns logits (slots, V)."""
    B = cache.slots
    active = np.ones(B, bool) if active is None else np.asarray(active)
    pos = cache.lengths
    if (pos[active] >= cfg.ctx).any():
        raise ValueError("a slot is full; re-prefill it before decoding")
    pos = np.minimum(pos, cfg.ctx - 1)
    rows = np.arange(B)
    x = (params["tok"][ids] + params["pos"][pos])[:, None]  # (B, 1, D)
    mask = (np.arange(cfg.ctx)[None, :] <= pos[:, None])[:, None, None, :]  # (B,1,1,ctx)
    for i in range(cfg.layers):
        h, _ = layernorm(x, params[f"{i}.ln1.g"], params[f"{i}.ln1.b"])
        q, k, v = (heads(a, cfg.heads) for a in np.split(h @ params[f"{i}.attn.qkv"], 3, -1))
        old_k = cache.k[i][rows, :, pos].copy()
        old_v = cache.v[i][rows, :, pos].copy()
        cache.k[i][rows, :, pos] = k[:, :, 0]
        cache.v[i][rows, :, pos] = v[:, :, 0]
        scores = q @ cache.k[i].transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
        p = softmax(np.where(mask, scores, -np.inf))
        x = x + merge(p @ cache.v[i]) @ params[f"{i}.attn.out"]
        h, _ = layernorm(x, params[f"{i}.ln2.g"], params[f"{i}.ln2.b"])
        g, _ = gelu(h @ params[f"{i}.mlp.up"] + params[f"{i}.mlp.up_b"])
        x = x + g @ params[f"{i}.mlp.down"] + params[f"{i}.mlp.down_b"]
        idle = ~active  # put back what idle slots had at that position
        cache.k[i][rows[idle], :, pos[idle]] = old_k[idle]
        cache.v[i][rows[idle], :, pos[idle]] = old_v[idle]
    cache.lengths[active] += 1
    h, _ = layernorm(x[:, 0], params["lnf.g"], params["lnf.b"])
    return h @ params["tok"].T


def generate_batch(params, cfg: Config, prompts, steps: int, rng=None, temperature=1.0):
    """Static batching: start every prompt together and decode `steps` tokens
    for each. Stops early if a slot fills its context."""
    rng = rng or np.random.default_rng(0)
    cache = BatchCache(cfg, len(prompts))
    logits = np.stack([cache.prefill(params, i, p) for i, p in enumerate(prompts)])
    out = np.zeros((len(prompts), steps), dtype=np.int64)
    for n in range(steps):
        out[:, n] = pick(logits, rng, temperature)
        if n + 1 == steps or (cache.lengths >= cfg.ctx).any():
            return out[:, : n + 1]
        logits = decode(params, cfg, out[:, n], cache)
    return out
