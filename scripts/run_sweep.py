#!/usr/bin/env python3
"""Run the learning-rate sweep, one arm per process.

    python scripts/run_sweep.py full_budget
    python scripts/run_sweep.py half_budget --dry-run

The grid lives in ``conf/sweep.yaml``. Each run is a separate process on
purpose: TensorFlow does not release GPU memory between models inside one
process, and on a 4 GB card the second run would die on allocation.

Never run two of these at once. There is one card, and two jobs on it fail in
ways that look like bad hyperparameters rather than contention.

If a run dies at its evaluation step -- an OOM after training finished, which
happens intermittently because the optimizer slots are still resident -- its
``best.keras`` is on disk and this script re-scores it in a fresh process
instead of retraining. That recovery is what ``--no-rescue`` turns off.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitcic import config  # noqa: E402


def run_name(arm: str, lr: float, subset) -> str:
    name = arm if arm == "zeroshot" else f"{arm}-lr{lr:g}"
    return name + (f"-tr{subset}" if subset else "")


def main() -> None:
    sweeps = config.load("sweep")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep", choices=sorted(sweeps))
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--no-rescue", action="store_true",
                    help="do not re-score a run that died after training")
    args = ap.parse_args()

    spec = sweeps[args.sweep]
    subset = spec.get("train_subset")
    runs_root = config.resolve("runs_root")

    jobs: list[tuple[str, float]] = []
    if not subset:
        jobs.append(("zeroshot", 0.0))
    for arm in ("finetune", "scratch"):
        jobs += [(arm, lr) for lr in spec.get(arm, [])]

    print(f"sweep {args.sweep}: {len(jobs)} runs, train_subset={subset}")
    for arm, lr in jobs:
        name = run_name(arm, lr, subset)
        out = runs_root / name
        if (out / "metrics.json").exists():
            print(f"skip  {name} (metrics.json present)")
            continue

        cmd = [sys.executable, "-m", "pitcic.train.finetune", "--arm", arm]
        if arm != "zeroshot":
            cmd += ["--lr", f"{lr:g}"]
        if subset:
            cmd += ["--train-subset", str(subset)]

        print(f"\n=== {name}\n{' '.join(cmd)}")
        if args.dry_run:
            continue
        started = time.time()
        rc = subprocess.run(cmd).returncode
        print(f"--- {name} rc={rc} in {time.time() - started:.0f}s")

        if rc != 0 and not args.no_rescue and (out / "best.keras").exists():
            print(f"    retrying evaluation of {name} in a fresh process")
            rescue = cmd + ["--eval-only", str(out / "best.keras")]
            rc = subprocess.run(rescue).returncode
            print(f"--- {name} rescue rc={rc}")
        if rc != 0:
            print(f"    {name} FAILED; continuing with the rest of the sweep")


if __name__ == "__main__":
    main()
