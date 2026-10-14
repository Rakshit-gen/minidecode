# minidecode

A small transformer inference engine in numpy. It trains its own tiny model in
seconds on a CPU, then runs it with the tricks real inference servers use: a
KV cache, int8 and int4 weights, continuous batching, prefix caching and
speculative decoding. Each one is measured against the plain version.

No model downloads and no GPU. The only dependency is numpy; the whole
environment with dev tools is about 33 MB.

## Quickstart

```
uv tool install git+https://github.com/Rakshit-gen/minidecode
minidecode train -o model.npz            # about 14 s on a laptop
minidecode generate model.npz --lines 3
minidecode bench model.npz
minidecode serve model.npz               # POST /generate on port 8000
```

The model learns to write web access-log lines:

```
12:08:37 PUT /api/orders/4996 200 32ms
13:09:06 GET /static/app.js 404 9ms
15:49:34 GET / 200 6ms
```

Those three came from the server, one request each, sent at the same time.

## What it does, with numbers

All numbers below are from one Apple M4, Python 3.13, on the default model
(2 layers, dim 64, 105,280 parameters, trained in 13.7 s). They are best of 3
after a warm-up run. Rerun any of them with the command shown.

### KV cache: `minidecode bench model.npz`

```
200 tokens, context 48     tokens/s      time  speedup
naive (full rerun)             1450   138.0ms    1.00x
kv cache                       3521    56.8ms    2.43x
kv cache, float16              3503    57.1ms    2.42x
```

The float16 cache halves the memory (48 KiB to 24 KiB per sequence) at the
same speed.

### Batching: same command

```
47 tokens per sequence     tokens/s      time  speedup
batch of 1                     3237    14.5ms    1.00x
batch of 4                    10203    18.4ms    3.15x
batch of 16                   24506    30.7ms    7.57x
batch of 64                   37276    80.7ms   11.52x
```

### Quantization: `minidecode quantize model.npz -o q4.npz --bits 4`

| Weights | File size | Bits per char (held out) | Valid lines, 5 seeds |
|---|---|---|---|
| float32 | 418 KiB | 0.986 | 78.4% (74.1 to 83.3) |
| int8 | 119 KiB | 0.986 | 79.6% (73.7 to 83.3) |
| int4 | 74 KiB | 0.991 | 76.2% (69.5 to 81.4) |

Valid lines is the share of generated lines that are well formed log lines,
from `minidecode eval MODEL --seeds 5` (about 300 lines). The spread between
seeds is wider than the gap between the formats, so the honest reading is:
int8 costs nothing measurable here, int4 costs 0.005 bits per char. Weights
are dequantized on load, so this saves disk, not compute.

### Continuous batching: `minidecode schedule model.npz`

64 requests of mixed lengths on 8 slots:

```
policy                     time  tokens/s  steps   busy  first token     done
static                    130ms      6015    243    39%       53.7ms     59ms
continuous                 80ms      9794    123    79%       26.4ms     34ms
```

With a shared prompt header (`--header "12:00:00 GET /api/orders"`), the
prefix cache reuses the header's KV entries:

```
continuous                 64ms      6562     64    81%       29.2ms     34ms
continuous + prefix        39ms     10766     64    81%       18.6ms     21ms
                       reused 1672 prompt tokens in 63 hits
```

### Speculative decoding: `minidecode spec TARGET DRAFT`

The draft is 1 layer, dim 32 (15,488 parameters). Greedy, 40 tokens:

| Target | Target alone | Best speculative | Accepted |
|---|---|---|---|
| default, 105k params | 3842 tokens/s | 3862 (k=2), 1.01x | 87% |
| 4 layers, dim 128, 803k params | 1549 tokens/s | 2076 (k=2), 1.34x | 74% |

On the small target there is no gain: the draft is only about twice as fast
per step because Python overhead dominates both. The bigger target makes the
draft relatively cheap and speculation pays. At temperature 0 the output is
exactly what the target writes alone; with sampling it uses rejection
sampling so the target's distribution is kept (both are tested).

### HTTP server: `scripts/load.py`

`minidecode serve model.npz` with 8 slots, 200 requests of one line each:

```
 clients    req/s  tokens/s      p50      p95
       1     67.6      2169     16ms     18ms
       8    222.1      7336     30ms     64ms
      32    302.0     10224     93ms    168ms
```

Requests that arrive while others are decoding join the running batch.

## The server

```
curl -X POST localhost:8000/generate \
  -d '{"prompt": "12:0", "max_new": 40, "temperature": 0.8, "lines": 1}'
{"text": "7:00 PUT /api/users/1166 204 36ms\n", "tokens": 34, "seconds": 0.0189}
```

`max_new` defaults to 32, `temperature` to the server's `-t`, and `lines`
stops after that many newlines. A prompt with characters the model never saw,
or one that doesn't fit the 48 character context with `max_new`, gets a 400.
`GET /health` returns the model shape and how many requests were served.

## Commands

```
minidecode train    [-o FILE] [--steps N] [--dim D] [--layers L] [--heads H] [--ctx C]
minidecode generate MODEL [-p PROMPT] [-n CHARS] [--lines N] [-t TEMP] [-k TOPK] [--top-p P]
minidecode eval     MODEL [--seeds N] [--json]
minidecode info     MODEL
minidecode bench    MODEL [-n TOKENS] [--batch 1 4 16 64] [--json]
minidecode quantize MODEL -o FILE --bits 8|4
minidecode schedule MODEL [--requests N] [--slots S] [--header TEXT]
minidecode spec     TARGET DRAFT [-k 2 4 6] [-t TEMP]
minidecode serve    MODEL [--port 8000] [--slots 8] [-t TEMP]
```

The two extra models in the speculative table were trained with
`train -o big.npz --dim 128 --layers 4` (51.4 s) and
`train -o draft.npz --dim 32 --layers 1 --heads 2` (3.5 s).

How each part works: [docs/how-it-works.md](docs/how-it-works.md).

## Measured vs Claimed

| Claim | Value | How measured | Date |
|---|---|---|---|
| Training time | 13.7 s, val loss 0.669 nats/char | `minidecode train`, Apple M4 | 2026-10-14 |
| KV cache speedup | 2.43x over full rerun | `minidecode bench`, 200 tokens, best of 3 | 2026-10-14 |
| Batch of 64 | 11.5x the tokens/s of batch 1 | same | 2026-10-14 |
| int4 quality cost | 0.986 to 0.991 bits/char, 418 to 74 KiB | `quantize --bits 4`, `eval --seeds 5` | 2026-10-14 |
| Continuous batching | 9794 vs 6015 tokens/s static | `minidecode schedule` | 2026-10-14 |
| Speculative decoding | 1.34x on the 803k target, 1.01x on the 105k one | `minidecode spec`, greedy, 40 tokens | 2026-10-14 |
| Server | 10224 tokens/s at 32 clients, 2169 at 1 | `scripts/load.py`, 8 slots | 2026-10-14 |
| Tests | 73, gradients checked by finite differences | `uv run pytest` | 2026-10-14 |

## Status

Works: everything above.

Not done: quantized matmuls (weights are dequantized on load, so int4 saves
size, not time), paged KV memory (the prefix cache copies entries rather than
sharing pages), streaming responses, and any tokenizer beyond characters. The
model only knows the log format it was trained on.
