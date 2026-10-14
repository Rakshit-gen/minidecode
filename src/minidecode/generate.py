"""Decoding loops."""

import numpy as np

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
