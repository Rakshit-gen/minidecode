"""Speculative decoding.

A small draft model guesses k tokens one at a time. The target model then
checks all k in a single forward pass and keeps the longest prefix it agrees
with, plus one token of its own. Every target pass yields at least one token
and at most k + 1, and the output is exactly what the target alone would
produce:

- greedy (temperature 0): a guess is kept if it is the target's argmax.
- sampling: a guess g is kept with probability min(1, p(g) / q(g)); on a
  rejection the replacement is drawn from max(0, p - q), renormalised
  (Leviathan et al., 2023). That keeps the target's distribution exactly.
"""

from dataclasses import dataclass

import numpy as np

from .kvcache import KVCache, step
from .sampling import probs


@dataclass
class Stats:
    proposed: int = 0
    accepted: int = 0
    target_calls: int = 0
    tokens: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_call(self) -> float:
        return self.tokens / self.target_calls if self.target_calls else 0.0


def _draw(p: np.ndarray, rng) -> int:
    return int(min(np.searchsorted(p.cumsum(), rng.random(), side="right"), len(p) - 1))


def speculative(
    target,
    draft,
    prompt,
    steps: int,
    k: int = 4,
    temperature: float = 0.0,
    rng=None,
    stats: Stats | None = None,
) -> list[int]:
    """`target` and `draft` are (params, cfg) pairs sharing a vocabulary."""
    (tp, tc), (dp, dc) = target, draft
    if tc.vocab != dc.vocab:
        raise ValueError("draft and target need the same vocabulary")
    if len(prompt) + steps + k + 1 > min(tc.ctx, dc.ctx):
        raise ValueError("prompt, steps and k must fit in both contexts")
    rng = rng or np.random.default_rng(0)
    stats = stats if stats is not None else Stats()
    ids = list(prompt)
    tcache, dcache = KVCache(tc), KVCache(dc)
    # Both caches hold everything but the last token, which is fed next.
    if len(ids) > 1:
        step(tp, tc, np.array([ids[:-1]]), tcache)
        step(dp, dc, np.array([ids[:-1]]), dcache)
    out: list[int] = []
    while len(out) < steps:
        base = tcache.length  # position of the last token
        guesses, qs = [], []
        token = ids[-1]
        for _ in range(k):
            q = probs(step(dp, dc, np.array([[token]]), dcache)[0, -1], temperature)
            token = int(q.argmax()) if temperature == 0 else _draw(q, rng)
            guesses.append(token)
            qs.append(q)
        logits = step(tp, tc, np.array([[ids[-1], *guesses]]), tcache)[0]
        ps = probs(logits, temperature)  # (k + 1, V)
        stats.target_calls += 1
        stats.proposed += k
        kept = []
        for g, q, p in zip(guesses, qs, ps, strict=False):
            if temperature == 0:
                ok = g == int(p.argmax())
            else:
                ok = rng.random() < min(1.0, p[g] / q[g])
            if not ok:
                break
            kept.append(g)
        n = len(kept)
        if n < k and temperature != 0:
            rest = np.maximum(ps[n] - qs[n], 0)
            extra = _draw(rest / rest.sum(), rng) if rest.sum() > 0 else _draw(ps[n], rng)
        else:
            extra = int(ps[n].argmax()) if temperature == 0 else _draw(ps[n], rng)
        stats.accepted += n
        new = [*kept, extra][: steps - len(out)]
        out += new
        stats.tokens += len(new)
        ids += new
        # The target saw last + k guesses; keep last + the accepted ones.
        tcache.truncate(base + 1 + n)
        # The draft saw last + the first k - 1 guesses.
        dcache.truncate(min(dcache.length, base + 1 + n))
        if n == k:
            step(dp, dc, np.array([[guesses[-1]]]), dcache)
    return out
