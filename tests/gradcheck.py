"""Compare a hand-written gradient with finite differences."""

import numpy as np


def numeric(f, x, eps=1e-6):
    """d f() / d x for a scalar f, by central differences, changing x in place."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        up = f()
        x[i] = old - eps
        down = f()
        x[i] = old
        grad[i] = (up - down) / (2 * eps)
    return grad


def close(a, b, tol=1e-6):
    return np.abs(a - b).max() / max(1e-12, np.abs(a).max() + np.abs(b).max()) < tol
