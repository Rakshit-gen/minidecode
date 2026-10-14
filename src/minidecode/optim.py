"""AdamW and a learning-rate schedule."""

import math

import numpy as np


class AdamW:
    def __init__(self, params: dict, lr=3e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        b1, b2 = self.betas
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mhat = self.m[k] / (1 - b1**self.t)
            vhat = self.v[k] / (1 - b2**self.t)
            if params[k].ndim == 2:  # decay matrices, not norms or biases
                params[k] -= lr * self.wd * params[k]
            params[k] -= lr * mhat / (np.sqrt(vhat) + self.eps)


def schedule(step: int, total: int, peak: float, warmup: int = 50) -> float:
    """Linear warmup, then cosine decay to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))
