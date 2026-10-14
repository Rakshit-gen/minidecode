"""Save and load a trained model as one .npz file: the weights, plus the
config and vocabulary as JSON. No pickle, so loading a file can't run code."""

import json
from pathlib import Path

import numpy as np

from .model import Config
from .tokenizer import CharTokenizer


def save(path, cfg: Config, params: dict, tok: CharTokenizer, meta: dict | None = None) -> None:
    header = {"config": cfg.to_dict(), "chars": tok.chars, "meta": meta or {}}
    np.savez(Path(path), __header__=np.frombuffer(json.dumps(header).encode(), np.uint8), **params)


def load(path) -> tuple[Config, dict, CharTokenizer, dict]:
    with np.load(Path(path), allow_pickle=False) as f:
        if "__header__" not in f.files:
            raise ValueError(f"{path} is not a minidecode checkpoint")
        header = json.loads(f["__header__"].tobytes())
        params = {k: f[k] for k in f.files if k != "__header__"}
    return Config(**header["config"]), params, CharTokenizer(header["chars"]), header["meta"]
