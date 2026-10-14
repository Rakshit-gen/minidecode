"""Scheduling requests onto batch slots.

Static batching fills every slot, decodes until the longest request in the
batch is done, then starts the next batch: short requests sit in finished
slots waiting for long ones. Continuous batching refills a slot as soon as
its request finishes, so the batch stays full.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .batch import BatchCache, decode
from .model import Config
from .sampling import pick


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new: int
    out: list[int] = field(default_factory=list)
    started: float = 0.0  # when its prefill ran
    first_token: float = 0.0
    finished: float = 0.0


@dataclass
class Run:
    requests: list[Request]
    start: float
    seconds: float
    steps: int  # batched decode calls
    busy: int  # slot-steps that produced a token someone wanted
    offered: int  # slot-steps available

    @property
    def tokens(self) -> int:
        return sum(len(r.out) for r in self.requests)

    @property
    def utilisation(self) -> float:
        return self.busy / self.offered if self.offered else 0.0

    @property
    def mean_latency(self) -> float:
        """Seconds from the start of the run until a request finished, averaged."""
        return float(np.mean([r.finished - self.start for r in self.requests]))


def serve(
    params,
    cfg: Config,
    requests: list[Request],
    slots: int,
    policy: str = "continuous",
    temperature: float = 0.0,
    seed: int = 0,
    prefix=None,
) -> Run:
    """Run every request to completion. All requests are queued at the start."""
    if policy not in ("continuous", "static"):
        raise ValueError(f"unknown policy {policy!r}")
    for r in requests:
        if len(r.prompt) + r.max_new > cfg.ctx:
            raise ValueError(f"request {r.id} needs more than the context of {cfg.ctx}")
    rng = np.random.default_rng(seed)
    waiting = deque(requests)
    cache = BatchCache(cfg, slots)
    owner: list[Request | None] = [None] * slots
    logits = np.zeros((slots, cfg.vocab), np.float32)
    start = time.perf_counter()
    steps = useful = total = 0
    while waiting or any(owner):
        free = [i for i, r in enumerate(owner) if r is None]
        if waiting and free and (policy == "continuous" or len(free) == slots):
            for i in free:
                if not waiting:
                    break
                r = waiting.popleft()
                r.started = time.perf_counter()
                logits[i] = cache.prefill(params, i, r.prompt, prefix)
                owner[i] = r
        active = np.array([r is not None for r in owner])
        ids = pick(logits, rng, temperature)
        now = time.perf_counter()
        for i, r in enumerate(owner):
            if r is None:
                continue
            if not r.out:
                r.first_token = now
            r.out.append(int(ids[i]))
            if len(r.out) == r.max_new:
                r.finished = now
                owner[i] = None
        still = np.array([r is not None for r in owner])
        useful += int(active.sum())
        total += slots
        if still.any():
            logits = decode(params, cfg, ids, cache, active=still)
            steps += 1
    return Run(requests, start, time.perf_counter() - start, steps, useful, total)
