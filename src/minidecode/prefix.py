"""Reuse the KV entries of prompt prefixes that were seen before.

Requests often start the same way (a system prompt, a fixed header). The keys
and values for those positions are the same every time, so a prefill can copy
them instead of recomputing them and only run the new part of the prompt.

Besides each whole prompt, the cache keeps every prefix whose length is a
multiple of `block`, the way paged KV caches share blocks. Two prompts that
share a header of at least one block then share its entries, without anyone
having to register the header first.
"""

from collections import OrderedDict

import numpy as np

from .kvcache import KVCache, step
from .model import Config


class PrefixCache:
    def __init__(self, cfg: Config, max_entries: int = 64, min_length: int = 4, block: int = 8):
        self.cfg = cfg
        self.block = block
        self.max_entries = max_entries
        self.min_length = min_length  # shorter prefixes aren't worth a lookup
        self.entries: OrderedDict[tuple, tuple[list, list, np.ndarray]] = OrderedDict()
        self.hits = self.misses = self.reused_tokens = 0

    def _longest(self, prompt: tuple):
        best = None
        for key in self.entries:
            if len(key) <= len(prompt) and prompt[: len(key)] == key:
                if best is None or len(key) > len(best):
                    best = key
        return best

    def prefill(self, params, prompt, cache: KVCache) -> np.ndarray:
        """Fill an empty `cache` with `prompt`, reusing the longest cached
        prefix; returns the logits after the prompt. The prompt and its block
        prefixes are then remembered for next time."""
        if cache.length:
            raise ValueError("prefill needs an empty cache")
        prompt = tuple(int(t) for t in prompt)
        key = self._longest(prompt)
        if key is None:
            self.misses += 1
        else:
            k, v, logits = self.entries[key]
            self._copy_in(cache, k, v, len(key))
            self.entries.move_to_end(key)
            self.hits += 1
            if len(key) == len(prompt):
                if logits is not None:  # stored as a whole prompt: logits and all
                    self.reused_tokens += len(key)
                    return logits
                cache.length -= 1  # a block entry has no logits; recompute the last token
            self.reused_tokens += cache.length
        logits = step(params, self.cfg, np.array([prompt[cache.length :]]), cache)[0, -1]
        n = len(prompt)
        for length in range(self.block, n, self.block):
            self._store(prompt[:length], cache, None)
        if n >= self.min_length:
            self._store(prompt, cache, logits)
        return logits

    def _store(self, key: tuple, cache: KVCache, logits) -> None:
        if key in self.entries:
            if logits is not None:  # upgrade a block entry with its logits
                k, v, _ = self.entries[key]
                self.entries[key] = (k, v, logits)
            self.entries.move_to_end(key)
            return
        n = len(key)
        k = [a[:, :, :n].copy() for a in cache.k]
        v = [a[:, :, :n].copy() for a in cache.v]
        self.entries[key] = (k, v, logits)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    @staticmethod
    def _copy_in(cache: KVCache, k, v, n: int) -> None:
        for i in range(len(cache.k)):
            cache.k[i][:, :, :n] = k[i]
            cache.v[i][:, :, :n] = v[i]
        cache.length = n
