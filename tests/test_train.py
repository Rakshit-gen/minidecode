from minidecode.corpus import generate
from minidecode.train import train


def test_a_short_run_learns_something():
    logs = []
    t = train(generate(3000), steps=60, batch_size=8, every=30, log=logs.append, dim=32, ctx=32)
    first_val, last_val = t.history[0][2], t.history[-1][2]
    assert [s for s, _, _ in t.history] == [30, 60]
    assert last_val < first_val < 3.9  # uniform over ~50 characters is 3.9 nats
    assert logs[-1].startswith("step    60  train ")
