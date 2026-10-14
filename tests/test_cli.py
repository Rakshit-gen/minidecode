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
