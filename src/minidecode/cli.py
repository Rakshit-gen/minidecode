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
from .model import count
from .quality import report
from .quant import quantize_params
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
    out = generate_cached(params, cfg, prompt, args.chars, rng, args.temperature, args.top_k)
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


def cmd_bench(args) -> int:
    cfg, params, tok, _ = load(args.model)
    prompt = tok.encode("\n")
    n = args.tokens
    rows = [
        timed("naive (full rerun)", n, lambda: generate_naive(params, cfg, prompt, n)),
        timed("kv cache", n, lambda: generate_cached(params, cfg, prompt, n)),
    ]
    print(f"{n} tokens, context {cfg.ctx}, best of 3")
    print(table(rows), end="")
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
    s.add_argument("-n", "--chars", type=int, default=400)
    s.add_argument("-t", "--temperature", type=float, default=1.0)
    s.add_argument("-k", "--top-k", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("eval", help="bits per char and valid line rate")
    s.add_argument("model")
    s.add_argument("-n", "--chars", type=int, default=2000, help="characters to sample")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_eval)

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
