import numpy as np
import pytest

from minidecode.checkpoint import load, save
from minidecode.model import Config, init
from minidecode.quant import quantize_params
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


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_round_trip_is_smaller_and_close(tmp_path, bits):
    cfg = Config(vocab=4, ctx=32, dim=32, heads=2)
    params, tok = init(cfg), CharTokenizer("ab\n ")
    save(tmp_path / "f.npz", cfg, params, tok)
    save(tmp_path / "q.npz", cfg, quantize_params(params, bits), tok)
    assert (tmp_path / "q.npz").stat().st_size < (tmp_path / "f.npz").stat().st_size / 2.5
    _, back, _, meta = load(tmp_path / "q.npz")
    assert meta == {"bits": bits}
    assert back["0.attn.qkv"].dtype == np.float32
    assert np.abs(back["0.attn.qkv"] - params["0.attn.qkv"]).max() < 0.02
