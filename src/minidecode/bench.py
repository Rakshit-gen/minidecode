"""Timing, the same way for everything: one warm-up run, then the best of a
few, reported as tokens per second."""

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Timing:
    name: str
    tokens: int
    seconds: float  # best run

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.seconds


def best_of(fn: Callable[[], object], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def timed(name: str, tokens: int, fn: Callable[[], object], repeat: int = 3) -> Timing:
    fn()  # warm up caches and allocations
    return Timing(name, tokens, best_of(fn, repeat))


def table(rows: list[Timing], base: Timing | None = None) -> str:
    base = base or rows[0]
    # Speedup is throughput against the first row, so rows that produce
    # different numbers of tokens (batch sizes) compare fairly.
    out = [f"{'':24} {'tokens/s':>10} {'time':>9} {'speedup':>8}"]
    for r in rows:
        out.append(
            f"{r.name:24} {r.tokens_per_s:10.0f} {r.seconds * 1000:7.1f}ms"
            f" {r.tokens_per_s / base.tokens_per_s:7.2f}x"
        )
    return "\n".join(out) + "\n"
