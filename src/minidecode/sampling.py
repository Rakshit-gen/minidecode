"""Pick the next token from logits."""

import numpy as np

from .layers import softmax


def probs(
    logits: np.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
) -> np.ndarray:
    """The distribution sampling draws from, over the last axis. Temperature 0
    puts all the weight on the most likely token. top_k keeps the k most likely
    tokens; top_p keeps the smallest set whose probability adds up to p."""
    logits = logits.astype(np.float64)
    if temperature == 0:
        out = np.zeros_like(logits)
        np.put_along_axis(out, logits.argmax(-1)[..., None], 1.0, -1)
        return out
    if top_k:
        kth = np.sort(logits, -1)[..., -top_k][..., None]
        logits = np.where(logits < kth, -np.inf, logits)
    p = softmax(logits / temperature)
    if top_p < 1.0:
        order = np.argsort(-p, -1)
        sorted_p = np.take_along_axis(p, order, -1)
        # Keep a token if the mass before it is under top_p (so the first always stays).
        keep_sorted = (sorted_p.cumsum(-1) - sorted_p) < top_p
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, order, keep_sorted, -1)
        p = np.where(keep, p, 0)
        p = p / p.sum(-1, keepdims=True)
    return p


def pick(logits, rng: np.random.Generator, temperature=1.0, top_k=0, top_p=1.0) -> np.ndarray:
    """One token per row of `logits` (..., V)."""
    p = probs(logits, temperature, top_k, top_p)
    if temperature == 0:
        return p.argmax(-1)
    flat = p.reshape(-1, p.shape[-1])
    u = rng.random((flat.shape[0], 1))
    # Inverse CDF; the clip guards against the cumsum ending a hair under 1.
    idx = (flat.cumsum(-1) < u).sum(-1).clip(max=p.shape[-1] - 1)
    return idx.reshape(p.shape[:-1])
