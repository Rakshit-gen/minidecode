"""The building blocks, each as a forward that returns what its backward
needs, and the backward by hand. Shapes: B batch, T time, D model width."""

import numpy as np

EPS = 1e-5


def layernorm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(-1, keepdims=True) + EPS)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_back(dy, cache):
    xhat, rstd, g = cache
    dxhat = dy * g
    dx = rstd * (
        dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True)
    )
    axes = tuple(range(dy.ndim - 1))
    return dx, (dy * xhat).sum(axes), dy.sum(axes)


K = np.sqrt(2 / np.pi)


def gelu(x):
    """The tanh approximation GPT-2 uses."""
    t = np.tanh(K * (x + 0.044715 * x**3))
    return 0.5 * x * (1 + t), (x, t)


def gelu_back(dy, cache):
    x, t = cache
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t**2) * K * (1 + 3 * 0.044715 * x**2))
