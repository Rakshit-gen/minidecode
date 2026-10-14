"""Command line entry point."""

import argparse
import json
import sys

import numpy as np

from . import __version__
from .checkpoint import load, save
from .corpus import generate
from .generate import generate_cached
from .model import count
from .quality import report
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
