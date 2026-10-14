"""Pick the next token from logits."""

import numpy as np

from .layers import softmax


def probs(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0) -> np.ndarray:
    """The distribution sampling draws from, over the last axis. Temperature 0
    puts all the weight on the most likely token."""
    logits = logits.astype(np.float64)
    if temperature == 0:
        out = np.zeros_like(logits)
        np.put_along_axis(out, logits.argmax(-1)[..., None], 1.0, -1)
        return out
    if top_k:
        kth = np.sort(logits, -1)[..., -top_k][..., None]
        logits = np.where(logits < kth, -np.inf, logits)
    return softmax(logits / temperature)


def pick(logits, rng: np.random.Generator, temperature=1.0, top_k=0) -> np.ndarray:
    """One token per row of `logits` (..., V)."""
    p = probs(logits, temperature, top_k)
    if temperature == 0:
        return p.argmax(-1)
    flat = p.reshape(-1, p.shape[-1])
    u = rng.random((flat.shape[0], 1))
    # Inverse CDF; the clip guards against the cumsum ending a hair under 1.
    idx = (flat.cumsum(-1) < u).sum(-1).clip(max=p.shape[-1] - 1)
    return idx.reshape(p.shape[:-1])
