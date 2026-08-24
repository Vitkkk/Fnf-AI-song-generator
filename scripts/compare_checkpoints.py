#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Generate the same latent seed from every saved epoch checkpoint.")
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--output", default="generated/checkpoint_comparison")
    p.add_argument("--prompt", default="fnf")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    ckpts = sorted(Path(args.checkpoints).glob("epoch_*.pt"))
    if not ckpts:
        raise SystemExit("No epoch_*.pt checkpoints found")
    for ckpt in ckpts:
        epoch_name = ckpt.stem
        cmd = [
            "python", "scripts/generate.py",
            "--checkpoint", str(ckpt),
            "--output", str(Path(args.output) / epoch_name),
            "--prompt", args.prompt,
            "--seed", str(args.seed),
            "--temperature", str(args.temperature),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
