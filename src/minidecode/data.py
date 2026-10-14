"""Turn a token stream into training batches."""

import numpy as np


def split(ids: np.ndarray, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Train on the start of the stream, validate on the end, so validation
    text was never seen in training."""
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def batch(ids: np.ndarray, size: int, ctx: int, rng: np.random.Generator):
    """`size` random windows of `ctx` tokens, and the same windows shifted by
    one: the token to predict at each position."""
    if len(ids) <= ctx:
        raise ValueError(f"need more than {ctx} tokens, have {len(ids)}")
    starts = rng.integers(0, len(ids) - ctx, size)
    idx = starts[:, None] + np.arange(ctx)
    return ids[idx], ids[idx + 1]
