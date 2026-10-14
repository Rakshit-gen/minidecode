# How minidecode works

Everything is numpy. Nothing is downloaded: the model is trained from scratch
on synthetic access-log lines, in about 14 seconds on a laptop CPU.

## The model

A decoder-only transformer with character tokens.

- `model.py`: token and position embeddings, then `layers` blocks of
  pre-layernorm causal self-attention and a GELU MLP, then a final layernorm.
  The output projection is the token embedding transposed (tied weights).
- `layers.py`: each piece has a forward and a hand-written backward. The tests
  check every backward against finite differences.
- `train.py`: AdamW with a warmup and cosine decay, random windows of the
  corpus as batches. The default model has 105,280 parameters.

## The corpus

`corpus.py` writes lines like `12:08:37 PUT /api/orders/4996 200 32ms`. The
clock restarts every 200 lines so training and validation see the same hours.
`problem(line)` says what is wrong with a line (a bad time, an unknown
endpoint, a status that doesn't fit the method), which gives a second quality
measure next to bits per character: the share of generated lines that are
valid.

## Generation

`generate_naive` reruns the whole context for every new token. That is the
baseline everything else is timed against.

`generate_cached` keeps a KV cache (`kvcache.py`). `step()` takes any number
of new tokens: the whole prompt at once (prefill), one token (decode), or k
guessed tokens (speculative verify). Only the new tokens are computed; their
keys and values are written into the cache and attention reads the rest from
it. When the context is full, the last half is prefilled again and decoding
carries on. A float16 cache halves the memory and is widened back to float32
for the math.

`sampling.py` handles temperature, top-k and top-p.

## Quantization

`quant.py` stores weight matrices as integers plus scales:

- int8: one float16 scale per output column, `w = q * scale`.
- int4: groups of 32 rows share a scale per column, codes from -7 to 7, two
  codes packed into each byte. A matrix whose rows don't split into groups of
  32 falls back to int8: that is the token table, whose 41 rows are the
  vocabulary.

Every matrix is quantized, embeddings included. Layernorm parameters and
biases stay float32. The checkpoint keeps
the packed form on disk and `load` dequantizes once, so quantization here
saves disk and download size, not compute: the matmuls still run in float32.

## Batching

`batch.py` gives every slot of a batch its own cache length, so sequences at
different positions decode in one call. `decode()` takes an `active` mask and
leaves idle slots alone.

`serve.py` puts requests onto slots. Static batching waits for the whole batch
to finish before admitting more. Continuous batching admits a new request as
soon as a slot frees up. `Engine` is that loop one step at a time, so new
requests can join while it runs.

`prefix.py` remembers the KV entries of prompts it has seen, both the whole
prompt and every prefix whose length is a multiple of 8. A new prompt copies
the longest stored prefix and only prefills the rest.

## Speculative decoding

`spec.py`: a small draft model proposes k tokens, the target checks them all
in one `step()`, keeps the agreed prefix plus one token of its own, and both
caches are truncated back to what was kept. At temperature 0 the output is
exactly the target's greedy output (tested). When sampling, guesses are
accepted with probability min(1, p/q) and a rejection resamples from
max(0, p - q), which keeps the target's distribution (tested by comparing
token frequencies over many runs).

It only pays off when a draft step is much cheaper than a target step. With
the default 105k model as the target, the 15k draft is only about twice as
fast per step, because Python overhead dominates both, and there is no gain.
With an 803k target it is about 5 times as fast and speculation wins.

## The server

`server.py` is `http.server.ThreadingHTTPServer`. Each request thread puts a
`Request` on the engine and waits on an event. One worker thread owns the
engine and steps it whenever there is work, so concurrent HTTP requests are
decoded in the same batch.
