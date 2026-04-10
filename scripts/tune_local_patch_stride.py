#!/usr/bin/env python3
"""Utility: compare val IoU for patch stride 4 vs 8 after short training (optional Week 9 tuning)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fno-checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for stride in (4, 8):
        out = repo / "outputs" / "week9_local_tune" / f"ps{stride}"
        cmd = [
            sys.executable,
            str(repo / "scripts" / "train_local_rejector.py"),
            "--fno-checkpoint",
            str(args.fno_checkpoint),
            "--patch-stride",
            str(stride),
            "--epochs",
            str(args.epochs),
            "--output-dir",
            str(out),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(repo))


if __name__ == "__main__":
    main()
