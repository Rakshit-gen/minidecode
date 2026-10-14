"""Weight quantization.

int8: one scale per output column, weights rounded to -127..127.
int4: one scale per group of 32 input rows in each column, weights rounded
to -8..7 and packed two to a byte.
Only the matrices are quantized; norms and biases are a tiny share of the
model and stay float. Weights are dequantized when a model is loaded, so this
saves disk and download size, not compute: numpy has no fast int4 or int8
matmul.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class QTensor:
    bits: int
    q: np.ndarray  # int8 values, or uint8 holding two int4 values
    scale: np.ndarray  # float16
    shape: tuple[int, int]
    group: int = 0  # int4 only

    @property
    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def quantize_int8(w: np.ndarray) -> QTensor:
    scale = np.abs(w).max(0) / 127
    scale = np.where(scale == 0, 1, scale).astype(np.float16)
    q = np.clip(np.round(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return QTensor(8, q, scale, w.shape)


def dequantize(t: QTensor) -> np.ndarray:
    if t.bits == 8:
        return t.q.astype(np.float32) * t.scale.astype(np.float32)
    if t.bits == 4:
        return _dequantize_int4(t)
    raise ValueError(f"unsupported bits: {t.bits}")


def quantize_int4(w: np.ndarray, group: int = 32) -> QTensor:
    rows, cols = w.shape
    if rows % group or rows % 2:
        raise ValueError(f"{rows} rows don't split into groups of {group}")
    g = w.reshape(rows // group, group, cols)
    scale = np.abs(g).max(1) / 7  # (groups, cols)
    scale = np.where(scale == 0, 1, scale).astype(np.float16)
    q = np.clip(np.round(g / scale[:, None].astype(np.float32)), -8, 7).astype(np.int8)
    nibbles = (q.reshape(rows, cols) & 0x0F).astype(np.uint8)
    packed = nibbles[0::2] | (nibbles[1::2] << 4)  # row pairs share a byte
    return QTensor(4, packed, scale, w.shape, group)


def _dequantize_int4(t: QTensor) -> np.ndarray:
    rows, cols = t.shape
    low = (t.q & 0x0F).astype(np.int8)
    high = (t.q >> 4).astype(np.int8)
    q = np.empty((rows, cols), np.int8)
    q[0::2], q[1::2] = low, high
    q = np.where(q > 7, q - 16, q)  # back from 4-bit two's complement
    g = q.reshape(rows // t.group, t.group, cols).astype(np.float32)
    return (g * t.scale[:, None].astype(np.float32)).reshape(rows, cols)


def quantize_params(params: dict, bits: int) -> dict:
    """Quantize every matrix; leave norms and biases as they are. With bits=4,
    a matrix whose rows don't split into groups of 32 (the token table, whose
    rows are the vocabulary) gets int8 instead."""
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, not {bits}")
    out = {}
    for k, v in params.items():
        if v.ndim != 2:
            out[k] = v
        elif bits == 4 and v.shape[0] % 32 == 0:
            out[k] = quantize_int4(v)
        else:  # int8, or an int4 request for a matrix whose rows don't split into groups
            out[k] = quantize_int8(v)
    return out


def dequantize_params(params: dict) -> dict:
    return {k: dequantize(v) if isinstance(v, QTensor) else v for k, v in params.items()}


def size(params: dict) -> int:
    """Bytes the weights take as stored."""
    return sum(v.nbytes for v in params.values())
