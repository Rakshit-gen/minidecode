import numpy as np
from gradcheck import close, numeric

from minidecode.layers import gelu, gelu_back, layernorm, layernorm_back

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
