import numpy as np
import pytest

from minidecode.checkpoint import load, save
from minidecode.model import Config, init
from minidecode.tokenizer import CharTokenizer


def test_round_trip(tmp_path):
    cfg = Config(vocab=4, ctx=8, dim=8, heads=2)
    params, tok = init(cfg), CharTokenizer("ab\n ")
    save(tmp_path / "m.npz", cfg, params, tok, {"steps": 3})
    cfg2, params2, tok2, meta = load(tmp_path / "m.npz")
    assert cfg2 == cfg and tok2.chars == tok.chars and meta == {"steps": 3}
    assert params2.keys() == params.keys()
    assert all(np.array_equal(params[k], params2[k]) for k in params)


def test_rejects_other_npz_files(tmp_path):
    np.savez(tmp_path / "x.npz", a=np.zeros(2))
    with pytest.raises(ValueError, match="not a minidecode checkpoint"):
        load(tmp_path / "x.npz")
