"""How good a model (or a cheaper way of running it) is.

- bits per character on log text it never trained on (lower is better)
- the share of lines it writes that are valid log lines (higher is better)
"""

import numpy as np

from .corpus import generate, valid_rate
from .generate import generate_cached
from .layers import cross_entropy
from .model import forward

HELD_OUT_SEED = 10_001  # training uses small seeds; this text is never trained on


def bits_per_char(params, cfg, tok, lines: int = 400) -> float:
    ids = tok.encode(generate(lines, seed=HELD_OUT_SEED))
    n = (len(ids) - 1) // cfg.ctx
    x = ids[: n * cfg.ctx].reshape(n, cfg.ctx)
    y = ids[1 : n * cfg.ctx + 1].reshape(n, cfg.ctx)
    total = 0.0
    for i in range(0, n, 64):  # chunks keep memory flat
        logits, _ = forward(params, cfg, x[i : i + 64])
        total += cross_entropy(logits, y[i : i + 64])[0] * len(x[i : i + 64])
    return float(total / n / np.log(2))


def sample(params, cfg, tok, chars: int = 2000, seed: int = 0, temperature: float = 1.0) -> str:
    rng = np.random.default_rng(seed)
    out = generate_cached(params, cfg, tok.encode("\n"), chars, rng, temperature)
    return tok.decode(out)


def report(params, cfg, tok, chars: int = 2000, seed: int = 0, seeds: int = 1) -> dict:
    """Valid lines are pooled over `seeds` samples (seed, seed+1, ...): one
    sample of a few dozen lines moves by several points from seed to seed."""
    valid = lines = 0
    rates = []
    for s in range(seed, seed + seeds):
        text = sample(params, cfg, tok, chars, s)
        n = text.count("\n")
        rates.append(valid_rate(text))
        valid += round(rates[-1] * n)
        lines += n
    return {
        "bits_per_char": round(bits_per_char(params, cfg, tok), 4),
        "valid_lines": round(valid / lines, 4) if lines else 0.0,
        "lines_checked": lines,
        "valid_lines_range": [round(min(rates), 4), round(max(rates), 4)],
    }
