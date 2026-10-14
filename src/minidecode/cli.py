"""Command line entry point."""

import argparse
import sys

from . import __version__
from .checkpoint import save
from .corpus import generate
from .model import count


def cmd_train(args) -> int:
    from .train import train

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
