import numpy as np

from minidecode.quant import dequantize, quantize_int8

rng = np.random.default_rng(0)


def test_int8_error_is_within_half_a_step():
    w = rng.normal(size=(64, 48)).astype(np.float32)
    t = quantize_int8(w)
    step = np.abs(w).max(0) / 127
    assert t.q.dtype == np.int8 and t.nbytes < w.nbytes / 3.5
    assert (np.abs(dequantize(t) - w) <= step * 0.5 + 1e-3 * np.abs(w).max(0)).all()


def test_all_zero_columns_survive():
    w = np.zeros((4, 3), np.float32)
    assert (dequantize(quantize_int8(w)) == 0).all()
