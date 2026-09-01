"""Local multi-seed / multi-config sweep (sequential; for a laptop or one GPU).

    python scripts/sweep.py --configs configs/sac_pendulum.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from frankenrl.config import load_config
from frankenrl.train import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out-dir", default="runs")
    p.add_argument("-o", "--set", action="append", default=[])
    args = p.parse_args()

    for config in args.configs:
        for seed in args.seeds:
            base = load_config(config, args.set)
            cfg = replace(base, seed=seed, out_dir=args.out_dir, label=f"{base.label}_s{seed}")
            print(f"\n=== {cfg.label} ===")
            train(cfg)


if __name__ == "__main__":
    main()
