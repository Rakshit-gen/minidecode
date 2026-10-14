import json

from minidecode.checkpoint import load
from minidecode.cli import main


def test_train_writes_a_checkpoint(tmp_path, capsys):
    out = tmp_path / "m.npz"
    args = ["train", "-o", str(out), "--lines", "800", "--steps", "20", "--dim", "16", "-q"]
    assert main([*args, "--ctx", "16", "--heads", "2"]) == 0
    assert "20 steps in" in capsys.readouterr().out
    cfg, _, _, meta = load(out)
    assert (cfg.dim, cfg.ctx) == (16, 16) and meta["steps"] == 20


def test_bad_settings_exit_2(tmp_path, capsys):
    assert main(["train", "-o", str(tmp_path / "m.npz"), "--dim", "10", "--heads", "4"]) == 2
    assert "doesn't split into 4 heads" in capsys.readouterr().err


def test_generate_and_eval(tmp_path, capsys):
    m = str(tmp_path / "m.npz")
    main(["train", "-o", m, "--lines", "800", "--steps", "5", "--dim", "16", "--heads", "2", "-q"])
    capsys.readouterr()
    assert main(["generate", m, "-n", "30", "-t", "0", "-p", "12:0"]) == 0
    assert len(capsys.readouterr().out.rstrip("\n")) <= 30
    assert main(["eval", m, "-n", "100", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {
        "bits_per_char",
        "valid_lines",
        "lines_checked",
    }
    assert main(["generate", m, "-p", "€"]) == 2


def test_bench_runs(tmp_path, capsys):
    m = str(tmp_path / "m.npz")
    main(["train", "-o", m, "--lines", "800", "--steps", "2", "--dim", "16", "--heads", "2", "-q"])
    capsys.readouterr()
    assert main(["bench", m, "-n", "10", "--batch", "1", "2"]) == 0
    out = capsys.readouterr().out
    assert "kv cache" in out and "batch of 2" in out


def test_quantize(tmp_path, capsys):
    m, q = str(tmp_path / "m.npz"), str(tmp_path / "q.npz")
    main(["train", "-o", m, "--lines", "800", "--steps", "2", "--dim", "32", "--heads", "2", "-q"])
    capsys.readouterr()
    assert main(["quantize", m, "-o", q, "--bits", "4", "-n", "100"]) == 0
    out = capsys.readouterr().out
    assert "float32  bits/char" in out and "int4     bits/char" in out
    assert main(["quantize", q, "-o", m + "2"]) == 2


def test_schedule(tmp_path, capsys):
    m = str(tmp_path / "m.npz")
    main(["train", "-o", m, "--lines", "800", "--steps", "2", "--dim", "16", "--heads", "2", "-q"])
    capsys.readouterr()
    assert main(["schedule", m, "--requests", "6", "--slots", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("6 requests") and out[2].startswith("static") and len(out) == 4


def test_spec(tmp_path, capsys):
    m, d = str(tmp_path / "m.npz"), str(tmp_path / "d.npz")
    base = ["--lines", "800", "--steps", "2", "--heads", "2", "-q"]
    main(["train", "-o", m, "--dim", "16", *base])
    main(["train", "-o", d, "--dim", "8", "--layers", "1", *base])
    capsys.readouterr()
    assert main(["spec", m, d, "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "speculative k=2" in out and "tokens per target pass" in out
