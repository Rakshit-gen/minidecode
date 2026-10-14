"""Decoding with a key/value cache.

Without a cache, every new token reruns attention over the whole context. With
one, each layer keeps the keys and values it already computed, and a step only
runs the new tokens through the model. `step` takes any number of new tokens,
so the same code does the prompt (prefill), single tokens (decode) and blocks
of guesses (speculative decoding's verify pass).
"""

import numpy as np

from .layers import gelu, heads, layernorm, merge, softmax
from .model import Config


class KVCache:
    def __init__(self, cfg: Config, batch: int = 1, dtype=np.float32):
        shape = (batch, cfg.heads, cfg.ctx, cfg.dim // cfg.heads)
        self.k = [np.zeros(shape, dtype) for _ in range(cfg.layers)]
        self.v = [np.zeros(shape, dtype) for _ in range(cfg.layers)]
        self.length = 0  # positions filled

    def truncate(self, length: int) -> None:
        """Forget everything after `length` (rejected speculative tokens)."""
        if not 0 <= length <= self.length:
            raise ValueError(f"can't truncate a cache of {self.length} to {length}")
        self.length = length

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k + self.v)


def step(params: dict, cfg: Config, ids: np.ndarray, cache: KVCache) -> np.ndarray:
    """Run new tokens `ids` (B, t) through the model after what the cache holds;
    returns their logits (B, t, V)."""
    t = ids.shape[1]
    start, end = cache.length, cache.length + t
    if end > cfg.ctx:
        raise ValueError(f"cache holds {start} and {t} more won't fit in {cfg.ctx}")
    x = params["tok"][ids] + params["pos"][start:end]
    # New token i (at position start+i) sees every cached position and the
    # new ones up to itself.
    mask = np.arange(end)[None, :] <= (start + np.arange(t))[:, None]
    for i in range(cfg.layers):
        h, _ = layernorm(x, params[f"{i}.ln1.g"], params[f"{i}.ln1.b"])
        q, k, v = (heads(a, cfg.heads) for a in np.split(h @ params[f"{i}.attn.qkv"], 3, -1))
        cache.k[i][:, :, start:end] = k
        cache.v[i][:, :, start:end] = v
        # A float16 cache is widened back for the math, so only storage shrinks.
        keys = cache.k[i][:, :, :end].astype(x.dtype, copy=False)
        values = cache.v[i][:, :, :end].astype(x.dtype, copy=False)
        scores = q @ keys.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
        p = softmax(np.where(mask, scores, -np.inf))
        x = x + merge(p @ values) @ params[f"{i}.attn.out"]
        h, _ = layernorm(x, params[f"{i}.ln2.g"], params[f"{i}.ln2.b"])
        g, _ = gelu(h @ params[f"{i}.mlp.up"] + params[f"{i}.mlp.up_b"])
        x = x + g @ params[f"{i}.mlp.down"] + params[f"{i}.mlp.down_b"]
    cache.length = end
    h, _ = layernorm(x, params["lnf.g"], params["lnf.b"])
    return h @ params["tok"].T
