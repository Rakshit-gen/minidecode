"""Weight quantization.

int8: one scale per output column, weights rounded to -127..127.
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
    raise ValueError(f"unsupported bits: {t.bits}")
