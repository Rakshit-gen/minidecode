import numpy as np
import pytest
from gradcheck import close, numeric

from minidecode.layers import cross_entropy
from minidecode.model import Config, count, forward, init, loss_and_grads

CFG = Config(vocab=11, ctx=8, dim=16, layers=2, heads=4)


def test_shapes_and_size():
    p = init(CFG)
    logits, _ = forward(p, CFG, np.zeros((3, 5), dtype=np.int32))
    assert logits.shape == (3, 5, 11)
    # tok + pos + per layer (2 norms, qkv, out, mlp up and down with biases) + final norm
    per_layer = 4 * 16 + 16 * 48 + 16 * 16 + 16 * 64 + 64 + 64 * 16 + 16
    assert count(p) == 11 * 16 + 8 * 16 + 2 * per_layer + 2 * 16


def test_fresh_model_predicts_close_to_uniform():
    ids = np.arange(8)[None] % 11
    logits, _ = forward(init(CFG), CFG, ids)
    loss, _ = cross_entropy(logits, (ids + 1) % 11)
    assert abs(loss - np.log(11)) < 0.05


def test_bad_configs_and_inputs():
    with pytest.raises(ValueError, match="doesn't split into 3 heads"):
        Config(vocab=5, dim=16, heads=3)
    with pytest.raises(ValueError, match="longer than the context of 8"):
        forward(init(CFG), CFG, np.zeros((1, 9), dtype=np.int32))


def test_every_gradient_matches_finite_differences():
    cfg = Config(vocab=7, ctx=6, dim=8, layers=2, heads=2)
    p = init(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    for k in p:  # move norms and biases off their initial 1s and 0s
        p[k] += rng.normal(size=p[k].shape) * 0.1
    x, y = rng.integers(0, 7, (2, 5)), rng.integers(0, 7, (2, 5))
    _, grads = loss_and_grads(p, cfg, x, y)
    for name in p:
        want = numeric(lambda: loss_and_grads(p, cfg, x, y)[0], p[name])
        assert close(grads[name], want), name
