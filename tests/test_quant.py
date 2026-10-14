import numpy as np
import pytest

from minidecode.quant import dequantize, quantize_int4, quantize_int8

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


def test_int4_round_trips_every_code_and_packs_two_per_byte():
    w = rng.normal(size=(64, 8)).astype(np.float32)
    t = quantize_int4(w)
    assert t.q.shape == (32, 8) and t.nbytes < w.nbytes / 6
    back = dequantize(t)
    step = np.abs(w.reshape(2, 32, 8)).max(1) / 7
    assert (np.abs(back - w).reshape(2, 32, 8) <= step[:, None] * 0.5 + 1e-2).all()
    # With a scale of exactly 1, every code -7..7 must survive packing exactly.
    codes = np.resize(np.arange(-7, 8, dtype=np.float32), 32)[:, None].repeat(2, 1)
    assert (dequantize(quantize_int4(codes)) == codes).all()


def test_int4_needs_whole_groups():
    with pytest.raises(ValueError, match="groups of 32"):
        quantize_int4(np.zeros((40, 4), np.float32))
