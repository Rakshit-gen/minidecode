"""Train a model on text."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .data import batch, split
from .layers import cross_entropy
from .model import Config, forward, init, loss_and_grads
from .optim import AdamW, schedule
from .tokenizer import CharTokenizer


@dataclass
class Trained:
    cfg: Config
    params: dict
    tok: CharTokenizer
    history: list[tuple[int, float, float]] = field(default_factory=list)  # step, train, val
    seconds: float = 0.0


def evaluate(params, cfg: Config, ids: np.ndarray, windows: int = 32) -> float:
    """Mean loss in nats over fixed, evenly spaced windows of `ids`."""
    starts = np.linspace(0, len(ids) - cfg.ctx - 1, windows).astype(int)
    idx = starts[:, None] + np.arange(cfg.ctx)
    logits, _ = forward(params, cfg, ids[idx])
    return float(cross_entropy(logits, ids[idx + 1])[0])


def train(
    text: str,
    steps: int = 600,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
    every: int = 100,
    **model,
) -> Trained:
    tok = CharTokenizer.fit(text)
    cfg = Config(vocab=tok.size, **model)
    train_ids, val_ids = split(tok.encode(text))
    params = init(cfg, seed)
    opt = AdamW(params, lr=lr)
    rng = np.random.default_rng(seed)
    out = Trained(cfg, params, tok)
    start = time.perf_counter()
    for step in range(steps):
        x, y = batch(train_ids, batch_size, cfg.ctx, rng)
        loss, grads = loss_and_grads(params, cfg, x, y)
        opt.step(params, grads, schedule(step, steps, lr))
        if (step + 1) % every == 0 or step + 1 == steps:
            val = evaluate(params, cfg, val_ids)
            out.history.append((step + 1, float(loss), val))
            if log:
                log(f"step {step + 1:5}  train {loss:.3f}  val {val:.3f}")
    out.seconds = time.perf_counter() - start
    return out
