import numpy as np
import pytest

from minidecode.layers import cross_entropy
from minidecode.model import Config, count, forward, init

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
