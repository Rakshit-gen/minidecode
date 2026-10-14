import numpy as np

from minidecode.optim import AdamW, schedule


def test_adamw_finds_a_minimum():
    p = {"w": np.array([[3.0, -2.0]]), "b": np.array([5.0])}
    opt = AdamW(p, lr=0.1, weight_decay=0.0)
    for _ in range(500):
        opt.step(p, {"w": 2 * (p["w"] - 1), "b": 2 * p["b"]})
    assert np.allclose(p["w"], 1, atol=1e-2) and np.allclose(p["b"], 0, atol=1e-2)


def test_schedule_warms_up_then_decays():
    lrs = [schedule(s, 1000, 1.0, warmup=100) for s in range(1000)]
    assert lrs[0] == 0.01 and lrs[99] == 1.0
    assert lrs[500] < lrs[200] and abs(lrs[-1] - 0.1) < 1e-3
