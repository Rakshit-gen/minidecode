from minidecode.corpus import generate
from minidecode.quality import report
from minidecode.train import train


def test_training_improves_both_measures():
    text = generate(3000)
    small = dict(dim=32, ctx=32, heads=2, batch_size=8, every=1000)
    before = train(text, steps=1, **small)
    after = train(text, steps=150, **small)
    r0 = report(before.params, before.cfg, before.tok, chars=600)
    r1 = report(after.params, after.cfg, after.tok, chars=600)
    assert r1["bits_per_char"] < r0["bits_per_char"] - 1
    assert r1["valid_lines"] >= r0["valid_lines"]


def test_report_pools_seeds():
    from minidecode.corpus import generate
    from minidecode.model import Config, init
    from minidecode.tokenizer import CharTokenizer

    tok = CharTokenizer.fit(generate(2000))
    cfg = Config(vocab=tok.size, ctx=16, dim=8, layers=1, heads=2)
    p = init(cfg, seed=0)
    one = [report(p, cfg, tok, chars=200, seed=s) for s in (3, 4)]
    both = report(p, cfg, tok, chars=200, seed=3, seeds=2)
    assert both["lines_checked"] == sum(r["lines_checked"] for r in one)
    lo, hi = both["valid_lines_range"]
    assert lo <= both["valid_lines"] <= hi
