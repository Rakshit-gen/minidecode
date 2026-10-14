"""The building blocks, each as a forward that returns what its backward
needs, and the backward by hand. Shapes: B batch, T time, D model width."""

import numpy as np

EPS = 1e-5


def flat_t(a, b):
    """sum over batch and time of outer(a, b): the weight gradient of a linear
    layer. One 2D matmul, which BLAS does far faster than einsum."""
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


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
    t = np.tanh(K * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1 + t), (x, t)


def gelu_back(dy, cache):
    x, t = cache
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t**2) * K * (1 + 3 * 0.044715 * x * x))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def heads(x, n):
    """(B, T, D) -> (B, n, T, D/n)"""
    B, T, D = x.shape
    return x.reshape(B, T, n, D // n).transpose(0, 2, 1, 3)


def merge(x):
    """(B, n, T, d) -> (B, T, n*d)"""
    B, n, T, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, n * d)


def attention(x, wqkv, wo, n_heads):
    """Causal multi-head self-attention, projections included."""
    T = x.shape[1]
    q, k, v = (heads(a, n_heads) for a in np.split(x @ wqkv, 3, axis=-1))
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = q @ k.transpose(0, 1, 3, 2) * scale
    scores = np.where(np.tri(T, dtype=bool), scores, -np.inf)
    p = softmax(scores)
    o = merge(p @ v)
    return o @ wo, (x, wqkv, wo, q, k, v, p, o, scale, n_heads)


def attention_back(dy, cache):
    x, wqkv, wo, q, k, v, p, o, scale, n_heads = cache
    dwo = flat_t(o, dy)
    do = heads(dy @ wo.T, n_heads)
    dp = do @ v.transpose(0, 1, 3, 2)
    dv = p.transpose(0, 1, 3, 2) @ do
    ds = p * (dp - (dp * p).sum(-1, keepdims=True)) * scale  # masked entries have p = 0
    dq = ds @ k
    dk = ds.transpose(0, 1, 3, 2) @ q
    dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
    dwqkv = flat_t(x, dqkv)
    return dqkv @ wqkv.T, dwqkv, dwo


def cross_entropy(logits, targets):
    """Mean negative log likelihood of the targets, in nats."""
    shifted = logits - logits.max(-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    picked = np.take_along_axis(logp, targets[..., None], -1)[..., 0]
    return -picked.mean(), (logp, targets)


def cross_entropy_back(cache):
    logp, targets = cache
    d = np.exp(logp)
    np.put_along_axis(d, targets[..., None], np.take_along_axis(d, targets[..., None], -1) - 1, -1)
    return d / targets.size
