"""A GPT-style decoder: token and position embeddings, pre-norm blocks of
attention and an MLP, and an output layer tied to the token embeddings.

Weights are a flat dict of numpy arrays, named like "1.attn.qkv", which keeps
saving, quantizing and comparing them simple.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .layers import (
    attention,
    attention_back,
    cross_entropy,
    cross_entropy_back,
    flat_t,
    gelu,
    gelu_back,
    layernorm,
    layernorm_back,
)


@dataclass(frozen=True)
class Config:
    vocab: int
    ctx: int = 64  # longest sequence the position table covers
    dim: int = 64
    layers: int = 2
    heads: int = 4

    def __post_init__(self):
        if self.dim % self.heads:
            raise ValueError(f"dim {self.dim} doesn't split into {self.heads} heads")

    def to_dict(self) -> dict:
        return asdict(self)


def init(cfg: Config, seed: int = 0, dtype=np.float32) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    D = cfg.dim

    def normal(*shape, std=0.02):
        return (rng.normal(size=shape) * std).astype(dtype)

    # Residual projections start smaller so the stream doesn't grow with depth.
    small = 0.02 / np.sqrt(2 * cfg.layers)
    p = {"tok": normal(cfg.vocab, D), "pos": normal(cfg.ctx, D)}
    for i in range(cfg.layers):
        p |= {
            f"{i}.ln1.g": np.ones(D, dtype),
            f"{i}.ln1.b": np.zeros(D, dtype),
            f"{i}.attn.qkv": normal(D, 3 * D),
            f"{i}.attn.out": normal(D, D, std=small),
            f"{i}.ln2.g": np.ones(D, dtype),
            f"{i}.ln2.b": np.zeros(D, dtype),
            f"{i}.mlp.up": normal(D, 4 * D),
            f"{i}.mlp.up_b": np.zeros(4 * D, dtype),
            f"{i}.mlp.down": normal(4 * D, D, std=small),
            f"{i}.mlp.down_b": np.zeros(D, dtype),
        }
    return p | {"lnf.g": np.ones(D, dtype), "lnf.b": np.zeros(D, dtype)}


def count(params: dict) -> int:
    return sum(a.size for a in params.values())


def forward(params: dict, cfg: Config, ids: np.ndarray):
    """Logits for every position of `ids` (B, T), plus what backward needs."""
    T = ids.shape[1]
    if T > cfg.ctx:
        raise ValueError(f"sequence of {T} is longer than the context of {cfg.ctx}")
    x = params["tok"][ids] + params["pos"][:T]
    caches = []
    for i in range(cfg.layers):
        h, ln1 = layernorm(x, params[f"{i}.ln1.g"], params[f"{i}.ln1.b"])
        a, att = attention(h, params[f"{i}.attn.qkv"], params[f"{i}.attn.out"], cfg.heads)
        x = x + a
        h, ln2 = layernorm(x, params[f"{i}.ln2.g"], params[f"{i}.ln2.b"])
        u = h @ params[f"{i}.mlp.up"] + params[f"{i}.mlp.up_b"]
        g, gl = gelu(u)
        x = x + g @ params[f"{i}.mlp.down"] + params[f"{i}.mlp.down_b"]
        caches.append((ln1, att, ln2, h, gl, g))
    h, lnf = layernorm(x, params["lnf.g"], params["lnf.b"])
    logits = h @ params["tok"].T
    return logits, (ids, caches, lnf, h)


def loss_and_grads(params: dict, cfg: Config, x: np.ndarray, y: np.ndarray):
    """Mean cross-entropy of predicting y from x, and its gradient for every weight."""
    logits, (ids, caches, lnf, h) = forward(params, cfg, x)
    loss, ce = cross_entropy(logits, y)
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dlogits = cross_entropy_back(ce)
    grads["tok"] += flat_t(dlogits, h)  # the tied output layer
    dx, grads["lnf.g"], grads["lnf.b"] = layernorm_back(dlogits @ params["tok"], lnf)
    for i in reversed(range(cfg.layers)):
        ln1, att, ln2, h2, gl, g = caches[i]
        # MLP branch: x = x + gelu(h2 @ up + up_b) @ down + down_b
        grads[f"{i}.mlp.down"] = flat_t(g, dx)
        grads[f"{i}.mlp.down_b"] = dx.sum((0, 1))
        du = gelu_back(dx @ params[f"{i}.mlp.down"].T, gl)
        grads[f"{i}.mlp.up"] = flat_t(h2, du)
        grads[f"{i}.mlp.up_b"] = du.sum((0, 1))
        dh, grads[f"{i}.ln2.g"], grads[f"{i}.ln2.b"] = layernorm_back(
            du @ params[f"{i}.mlp.up"].T, ln2
        )
        dx = dx + dh
        # Attention branch: x = x + attention(layernorm(x))
        da, grads[f"{i}.attn.qkv"], grads[f"{i}.attn.out"] = attention_back(dx, att)
        dh, grads[f"{i}.ln1.g"], grads[f"{i}.ln1.b"] = layernorm_back(da, ln1)
        dx = dx + dh
    onehot = np.eye(cfg.vocab, dtype=dx.dtype)[ids.reshape(-1)]
    grads["tok"] += onehot.T @ dx.reshape(-1, cfg.dim)
    grads["pos"][: ids.shape[1]] = dx.sum(0)
    return loss, grads
