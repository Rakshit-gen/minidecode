"""A GPT-style decoder: token and position embeddings, pre-norm blocks of
attention and an MLP, and an output layer tied to the token embeddings.

Weights are a flat dict of numpy arrays, named like "1.attn.qkv", which keeps
saving, quantizing and comparing them simple.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .layers import attention, gelu, layernorm


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
