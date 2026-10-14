"""Save and load a trained model as one .npz file: the weights, plus the
config and vocabulary as JSON. No pickle, so loading a file can't run code.

Quantized matrices are stored as their packed values and scales and turned
back into float weights on load."""

import json
from pathlib import Path

import numpy as np

from .model import Config
from .quant import QTensor, dequantize_params
from .tokenizer import CharTokenizer


def save(path, cfg: Config, params: dict, tok: CharTokenizer, meta: dict | None = None) -> None:
    arrays, quant = {}, {}
    for k, v in params.items():
        if isinstance(v, QTensor):
            arrays[f"{k}.q"], arrays[f"{k}.scale"] = v.q, v.scale
            quant[k] = {"bits": v.bits, "shape": list(v.shape), "group": v.group}
        else:
            arrays[k] = v
    header = {"config": cfg.to_dict(), "chars": tok.chars, "meta": meta or {}, "quant": quant}
    np.savez(Path(path), __header__=np.frombuffer(json.dumps(header).encode(), np.uint8), **arrays)


def load(path) -> tuple[Config, dict, CharTokenizer, dict]:
    with np.load(Path(path), allow_pickle=False) as f:
        if "__header__" not in f.files:
            raise ValueError(f"{path} is not a minidecode checkpoint")
        header = json.loads(f["__header__"].tobytes())
        quant = header.get("quant", {})
        packed = {f"{k}.q" for k in quant} | {f"{k}.scale" for k in quant}
        params = {k: f[k] for k in f.files if k != "__header__" and k not in packed}
        for k, q in quant.items():
            shape = tuple(q["shape"])
            params[k] = QTensor(q["bits"], f[f"{k}.q"], f[f"{k}.scale"], shape, q["group"])
    meta = dict(header["meta"])
    if quant:
        meta["bits"] = min(q["bits"] for q in quant.values())
    cfg, tok = Config(**header["config"]), CharTokenizer(header["chars"])
    return cfg, dequantize_params(params), tok, meta
