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
    temperature: float | None = None  # None: the engine's
    stop: int | None = None  # finish early after this token...
    stop_count: int = 1  # ...has come out this many times
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
    def mean_first_token(self) -> float:
        """Seconds from the start of the run to a request's first token, averaged."""
        return float(np.mean([r.first_token - self.start for r in self.requests]))

    @property
    def mean_latency(self) -> float:
        """Seconds from the start of the run until a request finished, averaged."""
        return float(np.mean([r.finished - self.start for r in self.requests]))


class Engine:
    """The scheduling loop, one step at a time, so requests can be added while
    it runs. `serve` drives it over a fixed list; the HTTP server drives it
    from a queue."""

    def __init__(
        self,
        params,
        cfg: Config,
        slots: int,
        policy="continuous",
        temperature=0.0,
        seed=0,
        prefix=None,
    ):
        if policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {policy!r}")
        self.params, self.cfg, self.policy, self.prefix = params, cfg, policy, prefix
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.waiting: deque[Request] = deque()
        self.cache = BatchCache(cfg, slots)
        self.owner: list[Request | None] = [None] * slots
        self.logits = np.zeros((slots, cfg.vocab), np.float32)
        self.steps = self.busy_slots = self.offered = 0

    def submit(self, r: Request) -> None:
        if len(r.prompt) + r.max_new > self.cfg.ctx:
            raise ValueError(f"request {r.id} needs more than the context of {self.cfg.ctx}")
        if not r.prompt or r.max_new < 1:
            raise ValueError(f"request {r.id} needs a prompt and max_new of at least 1")
        self.waiting.append(r)

    @property
    def idle(self) -> bool:
        return not self.waiting and not any(self.owner)

    def _admit(self) -> None:
        free = [i for i, r in enumerate(self.owner) if r is None]
        if not (self.waiting and free):
            return
        if self.policy == "static" and len(free) < len(self.owner):
            return  # static: wait until the whole batch is done
        for i in free:
            if not self.waiting:
                break
            r = self.waiting.popleft()
            r.started = time.perf_counter()
            self.logits[i] = self.cache.prefill(self.params, i, r.prompt, self.prefix)
            self.owner[i] = r

    def step(self) -> list[Request]:
        """Admit what fits, emit one token per busy slot, and return the
        requests that finished."""
        self._admit()
        ids = np.zeros(len(self.owner), dtype=np.int64)
        done = []
        now = time.perf_counter()
        for i, r in enumerate(self.owner):
            if r is None:
                continue
            t = self.temperature if r.temperature is None else r.temperature
            ids[i] = pick(self.logits[i], self.rng, t)
            if not r.out:
                r.first_token = now
            r.out.append(int(ids[i]))
            self.busy_slots += 1
            if len(r.out) == r.max_new or (
                r.stop is not None and r.out.count(r.stop) >= r.stop_count
            ):
                r.finished = now
                self.owner[i] = None
                done.append(r)
        self.offered += len(self.owner)
        still = np.array([r is not None for r in self.owner])
        if still.any():
            self.logits = decode(self.params, self.cfg, ids, self.cache, active=still)
            self.steps += 1
        return done


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
    engine = Engine(params, cfg, slots, policy, temperature, seed, prefix)
    for r in requests:
        engine.submit(r)
    start = time.perf_counter()
    while not engine.idle:
        engine.step()
    seconds = time.perf_counter() - start
    return Run(requests, start, seconds, engine.steps, engine.busy_slots, engine.offered)
