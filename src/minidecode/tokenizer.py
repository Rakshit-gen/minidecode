"""Characters as tokens. The log alphabet is about 50 characters, so a
character vocabulary keeps the embedding table tiny and needs no training."""

import numpy as np


class CharTokenizer:
    def __init__(self, chars: str):
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in the vocabulary")
        self.chars = chars
        self.index = {c: i for i, c in enumerate(chars)}

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        return cls("".join(sorted(set(text))))

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        try:
            return np.array([self.index[c] for c in text], dtype=np.int32)
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)
