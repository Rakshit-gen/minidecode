import numpy as np
from gradcheck import close, numeric

from minidecode.layers import (
    attention,
    attention_back,
    cross_entropy,
    cross_entropy_back,
    gelu,
    gelu_back,
    layernorm,
    layernorm_back,
)

rng = np.random.default_rng(0)


def test_layernorm_normalises_and_its_gradient_checks():
    x, g, b = rng.normal(size=(2, 3, 5)), rng.normal(size=5), rng.normal(size=5)
    y, _ = layernorm(x, np.ones(5), np.zeros(5))
    assert np.allclose(y.mean(-1), 0) and np.allclose(y.std(-1), 1, atol=1e-3)
    w = rng.normal(size=(2, 3, 5))  # a random direction makes the loss scalar
    out, cache = layernorm(x, g, b)
    dx, dg, db = layernorm_back(w, cache)

    def loss():
        return (layernorm(x, g, b)[0] * w).sum()

    assert close(dx, numeric(loss, x)) and close(dg, numeric(loss, g))
    assert close(db, numeric(loss, b))


def test_gelu_gradient_checks():
    x = rng.normal(size=(4, 6)) * 2
    w = rng.normal(size=(4, 6))
    _, cache = gelu(x)
    assert close(gelu_back(w, cache), numeric(lambda: (gelu(x)[0] * w).sum(), x))
    assert abs(gelu(np.array(3.0))[0] - 2.9964) < 1e-4


def test_attention_is_causal_and_its_gradient_checks():
    D, H = 8, 2
    x = rng.normal(size=(2, 5, D))
    wqkv, wo = rng.normal(size=(D, 3 * D)) * 0.3, rng.normal(size=(D, D)) * 0.3
    y, cache = attention(x, wqkv, wo, H)
    changed = x.copy()
    changed[:, 3:] += 1.0  # the future changes...
    assert np.allclose(attention(changed, wqkv, wo, H)[0][:, :3], y[:, :3])  # ...the past doesn't

    w = rng.normal(size=y.shape)
    dx, dwqkv, dwo = attention_back(w, cache)

    def loss():
        return (attention(x, wqkv, wo, H)[0] * w).sum()

    assert close(dx, numeric(loss, x))
    assert close(dwqkv, numeric(loss, wqkv)) and close(dwo, numeric(loss, wo))


def test_cross_entropy_value_and_gradient():
    logits = rng.normal(size=(2, 3, 7))
    targets = rng.integers(0, 7, size=(2, 3))
    loss, cache = cross_entropy(logits, targets)
    uniform, _ = cross_entropy(np.zeros((1, 1, 7)), np.array([[2]]))
    assert np.isclose(uniform, np.log(7))
    g = numeric(lambda: cross_entropy(logits, targets)[0], logits)
    assert close(cross_entropy_back(cache), g)
