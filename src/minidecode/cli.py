"""Command line entry point."""

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .batch import generate_batch
from .bench import table, timed
from .checkpoint import load, save
from .corpus import generate
from .generate import generate_cached, generate_naive
from .kvcache import KVCache
from .model import count
from .prefix import PrefixCache
from .quality import report
from .quant import quantize_params
from .serve import Request, serve
from .spec import Stats, speculative
from .train import train


def cmd_train(args) -> int:
    text = generate(args.lines, seed=args.seed)
    log = None if args.quiet else (lambda s: print(s, file=sys.stderr))
    t = train(
        text,
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        log=log,
        ctx=args.ctx,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
    )
    step, train_loss, val = t.history[-1]
    meta = {"steps": step, "val_loss": val, "seconds": round(t.seconds, 2), "lines": args.lines}
    save(args.output, t.cfg, t.params, t.tok, meta)
    print(
        f"{count(t.params):,} parameters, {step} steps in {t.seconds:.1f} s,"
        f" val loss {val:.3f} nats/char -> {args.output}"
    )
    return 0


def cmd_generate(args) -> int:
    cfg, params, tok, _ = load(args.model)
    rng = np.random.default_rng(args.seed)
    prompt = tok.encode(args.prompt.encode().decode("unicode_escape"))
    newline = int(tok.encode("\n")[0]) if "\n" in tok.chars else None
    out = generate_cached(
        params,
        cfg,
        prompt,
        args.chars,
        rng,
        args.temperature,
        args.top_k,
        args.top_p,
        stop=newline if args.lines else None,
        stop_count=args.lines or 1,
    )
    print(tok.decode(out), end="" if args.chars and tok.decode(out[-1:]) == "\n" else "\n")
    return 0


def cmd_eval(args) -> int:
    cfg, params, tok, meta = load(args.model)
    r = report(params, cfg, tok, chars=args.chars, seed=args.seed)
    if args.json:
        print(json.dumps(r))
    else:
        print(f"bits per char {r['bits_per_char']:.3f}  (held-out logs)")
        print(f"valid lines   {r['valid_lines']:.1%}  ({r['lines_checked']} sampled lines)")
    return 0


def cmd_info(args) -> int:
    cfg, params, tok, meta = load(args.model)
    kv = KVCache(cfg).nbytes()
    print(f"{args.model}  ({os.path.getsize(args.model) / 1024:.0f} KiB on disk)")
    print(f"weights      {meta.get('bits', 32)}-bit, {count(params):,} parameters")
    print(f"shape        {cfg.layers} layers, dim {cfg.dim}, {cfg.heads} heads, context {cfg.ctx}")
    print(f"vocabulary   {cfg.vocab} characters")
    print(f"KV cache     {kv / 1024:.0f} KiB per sequence at full context")
    if "steps" in meta:
        print(
            f"trained      {meta['steps']} steps on {meta.get('lines', '?')} lines in"
            f" {meta.get('seconds', 0):.1f} s, val loss {meta['val_loss']:.3f}"
        )
    return 0


def cmd_bench(args) -> int:
    cfg, params, tok, _ = load(args.model)
    prompt = tok.encode("\n")
    n = args.tokens
    rows = [
        timed("naive (full rerun)", n, lambda: generate_naive(params, cfg, prompt, n)),
        timed("kv cache", n, lambda: generate_cached(params, cfg, prompt, n)),
        timed(
            "kv cache, float16",
            n,
            lambda: generate_cached(params, cfg, prompt, n, kv_dtype=np.float16),
        ),
    ]
    print(f"{n} tokens, context {cfg.ctx}, best of 3")
    print(table(rows), end="")
    for dtype in (np.float32, np.float16):
        size = KVCache(cfg, dtype=dtype).nbytes()
        print(f"kv cache memory per sequence, {np.dtype(dtype).name}: {size / 1024:.0f} KiB")
    steps = cfg.ctx - 1  # one prompt token, then fill the context
    batch_rows = [
        timed(
            f"batch of {b}",
            b * steps,
            lambda b=b: generate_batch(params, cfg, [prompt] * b, steps),
        )
        for b in args.batch
    ]
    print(f"\nbatched decoding, {steps} tokens per sequence")
    print(table(batch_rows), end="")
    return 0


def cmd_quantize(args) -> int:
    cfg, params, tok, meta = load(args.model)
    if "bits" in meta:
        raise ValueError(f"{args.model} is already quantized")
    save(args.output, cfg, quantize_params(params, args.bits), tok, meta)
    before, after = os.path.getsize(args.model), os.path.getsize(args.output)
    print(f"{args.model}: {before / 1024:.0f} KiB -> {args.output}: {after / 1024:.0f} KiB")
    if not args.no_eval:
        _, qparams, _, _ = load(args.output)
        for name, p in (("float32", params), (f"int{args.bits}", qparams)):
            r = report(p, cfg, tok, chars=args.chars)
            print(
                f"{name:8} bits/char {r['bits_per_char']:.4f}  valid lines {r['valid_lines']:.1%}"
            )
    return 0


def workload(n: int, longest: int, seed: int) -> list[tuple[list[int], int]]:
    """Mostly short requests with some long ones, like chat traffic."""
    rng = np.random.default_rng(seed)
    short = rng.integers(4, max(5, longest // 4), n)
    long = rng.integers(longest // 2, longest + 1, n)
    return [(i, int(long[i] if rng.random() < 0.25 else short[i])) for i in range(n)]


def cmd_schedule(args) -> int:
    cfg, params, tok, _ = load(args.model)
    header = args.header.encode().decode("unicode_escape")

    def prompt(i: int) -> list[int]:
        # With a header, prompts share it and differ after it.
        return list(tok.encode(f"{header}\n{i % 24:02}:" if header else "\n"))

    longest = max(len(prompt(i)) for i in range(args.requests))
    if cfg.ctx - longest < 8:
        raise ValueError(
            f"prompts of {longest} characters leave under 8 to generate in a context of"
            f" {cfg.ctx}; use a shorter --header"
        )
    jobs = workload(args.requests, cfg.ctx - longest, args.seed)
    print(f"{len(jobs)} requests, {sum(n for _, n in jobs)} tokens, {args.slots} slots")
    print(
        f"{'policy':22} {'time':>8} {'tokens/s':>9} {'steps':>6} {'busy':>6}"
        f" {'first token':>12} {'done':>8}"
    )
    runs = [("static", "static", None), ("continuous", "continuous", None)]
    if header:
        runs.append(("continuous + prefix", "continuous", PrefixCache(cfg)))
    for name, policy, prefix in runs:
        reqs = [Request(i, prompt(i), n) for i, n in jobs]
        serve(params, cfg, [Request(0, prompt(0), 2)], args.slots, policy)  # warm up
        r = serve(params, cfg, reqs, args.slots, policy, prefix=prefix)
        print(
            f"{name:22} {r.seconds * 1000:6.0f}ms {r.tokens / r.seconds:9.0f} {r.steps:6}"
            f" {r.utilisation:6.0%} {r.mean_first_token * 1000:10.1f}ms"
            f" {r.mean_latency * 1000:6.0f}ms"
        )
        if prefix is not None:
            print(f"{'':22} reused {prefix.reused_tokens} prompt tokens in {prefix.hits} hits")
    return 0


def cmd_serve(args) -> int:
    from .server import make_server

    cfg, params, tok, _ = load(args.model)
    httpd = make_server(
        params, cfg, tok, args.host, args.port, slots=args.slots, temperature=args.temperature
    )
    host, port = httpd.server_address[:2]
    print(f"serving {args.model} on http://{host}:{port} with {args.slots} slots", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        httpd.worker.stop()
    return 0


def cmd_spec(args) -> int:
    tcfg, tparams, ttok, _ = load(args.model)
    dcfg, dparams, dtok, _ = load(args.draft)
    if ttok.chars != dtok.chars:
        raise ValueError("draft and target were trained with different vocabularies")
    prompt = list(ttok.encode("\n"))
    n = min(tcfg.ctx, dcfg.ctx) - len(prompt) - max(args.k) - 1
    rows = [timed("target alone", n, lambda: generate_cached(tparams, tcfg, prompt, n, None, 0))]
    rows.append(timed("draft alone", n, lambda: generate_cached(dparams, dcfg, prompt, n, None, 0)))
    notes = []
    for k in args.k:
        stats = Stats()

        def run(k=k, stats=stats):
            speculative(
                (tparams, tcfg),
                (dparams, dcfg),
                prompt,
                n,
                k,
                args.temperature,
                np.random.default_rng(0),
                stats,
            )

        rows.append(timed(f"speculative k={k}", n, run))
        notes.append(
            f"k={k}: {stats.acceptance:.0%} of guesses accepted,"
            f" {stats.tokens_per_call:.2f} tokens per target pass"
        )
    print(f"{n} tokens, temperature {args.temperature:g}, best of 3")
    print(table(rows), end="")
    print("\n".join(notes))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="minidecode", description="A small transformer inference engine in numpy."
    )
    p.add_argument("--version", action="version", version=f"minidecode {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("train", help="train a model on synthetic access logs")
    s.add_argument("-o", "--output", default="model.npz", help="checkpoint to write")
    s.add_argument("--lines", type=int, default=20000, help="log lines to train on")
    s.add_argument("--steps", type=int, default=800)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--ctx", type=int, default=48, help="context length in characters")
    s.add_argument("--dim", type=int, default=64)
    s.add_argument("--layers", type=int, default=2)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("generate", help="write text with a trained model")
    s.add_argument("model", help="a checkpoint from train")
    s.add_argument("-p", "--prompt", default="\\n", help="text to continue (escapes allowed)")
    s.add_argument("-n", "--chars", type=int, default=400, help="most characters to write")
    s.add_argument("-l", "--lines", type=int, default=0, help="stop after this many lines")
    s.add_argument("-t", "--temperature", type=float, default=1.0)
    s.add_argument("-k", "--top-k", type=int, default=0)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("eval", help="bits per char and valid line rate")
    s.add_argument("model")
    s.add_argument("-n", "--chars", type=int, default=2000, help="characters to sample")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("info", help="describe a checkpoint")
    s.add_argument("model")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("bench", help="time the decoders against each other")
    s.add_argument("model")
    s.add_argument("-n", "--tokens", type=int, default=200)
    s.add_argument(
        "--batch", type=int, nargs="+", default=[1, 4, 16, 64], help="batch sizes to time"
    )
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("quantize", help="write an int8 or int4 copy of a model")
    s.add_argument("model")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--bits", type=int, choices=[8, 4], default=8)
    s.add_argument("-n", "--chars", type=int, default=2000, help="characters to sample")
    s.add_argument("--no-eval", action="store_true", help="skip the quality comparison")
    s.set_defaults(func=cmd_quantize)

    s = sub.add_parser("schedule", help="compare static and continuous batching")
    s.add_argument("model")
    s.add_argument("--requests", type=int, default=64)
    s.add_argument("--slots", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--header", default="", help="a prompt start every request shares")
    s.set_defaults(func=cmd_schedule)

    s = sub.add_parser("serve", help="answer POST /generate over HTTP")
    s.add_argument("model")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--slots", type=int, default=8)
    s.add_argument("-t", "--temperature", type=float, default=0.8)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("spec", help="time speculative decoding with a draft model")
    s.add_argument("model", help="the target model")
    s.add_argument("draft", help="a smaller model trained on the same text")
    s.add_argument("-k", type=int, nargs="+", default=[2, 4, 6], help="guesses per pass")
    s.add_argument("-t", "--temperature", type=float, default=0.0)
    s.set_defaults(func=cmd_spec)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as e:
        print(f"minidecode: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
