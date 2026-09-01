"""CLI entry point: ``python -m frankenrl.run --config configs/x.yaml --seed 0``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from frankenrl.config import load_config
from frankenrl.train import train


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="frankenrl", description=__doc__)
    p.add_argument("--config", required=True, help="path to a YAML run config")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--label", default=None, help="override config label (output subdir)")
    p.add_argument("--out-dir", default=None, help="override output root")
    p.add_argument(
        "-o", "--set", action="append", default=[], metavar="key.path=value",
        help="override any config field, repeatable (YAML-parsed value)",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.label is not None:
        cfg = replace(cfg, label=args.label)
    if args.out_dir is not None:
        cfg = replace(cfg, out_dir=args.out_dir)

    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
