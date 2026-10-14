import numpy as np
import pytest

from minidecode.data import batch, split


def test_targets_are_the_next_token():
    ids = np.arange(100)
    x, y = batch(ids, 8, 10, np.random.default_rng(0))
    assert x.shape == y.shape == (8, 10)
    assert (y == x + 1).all()


def test_split_keeps_order_and_short_streams_fail():
    train, val = split(np.arange(100), 0.2)
    assert train[-1] == 79 and val[0] == 80
    with pytest.raises(ValueError, match="need more than 10 tokens"):
        batch(np.arange(10), 1, 10, np.random.default_rng(0))
