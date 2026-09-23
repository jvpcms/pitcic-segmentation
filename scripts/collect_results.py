#!/usr/bin/env python3
"""Fold every run's metrics.json into the two CSV tables under results/.

    python scripts/collect_results.py

``results/sweep.csv`` is one row per run at the reported 5% threshold, and
``results/per_class_iou.csv`` is the same runs by class. Both are committed:
they are small, they are what the report's tables are built from, and they let
someone check the numbers without a GPU.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitcic import config  # noqa: E402
from pitcic.taxonomy import CLASS_NAMES  # noqa: E402

REPORTED = "unk05"


def main() -> None:
    runs_root = config.resolve("runs_root")
    out_dir = config.REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    rows, per_class = [], []
    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        rec = json.loads(metrics_path.read_text())
        sc = rec.get("protocol", {}).get(REPORTED)
        if sc is None:
            print(f"skip {metrics_path.parent.name}: no {REPORTED} block")
            continue
        rows.append({
            "run": rec["name"],
            "arm": rec["arm"],
            "lr": rec["lr"] if rec["arm"] != "zeroshot" else "",
            "train_tiles": rec["n_train_tiles"],
            "train_subset": rec.get("train_subset") or "",
            "epochs_run": rec.get("epochs_run", ""),
            "best_epoch": rec.get("best_epoch", ""),
            "pixel_accuracy": round(sc["pixel_accuracy"], 4),
            "miou": round(sc["miou"], 4),
            "subtiles_kept": sc["subtiles_kept"],
            "subtiles_total": sc["subtiles_total"],
        })
        row = {"run": rec["name"], "arm": rec["arm"]}
        for name in CLASS_NAMES:
            v = sc["iou"][name]
            row[name] = "" if v is None else round(v, 4)
        per_class.append(row)

    if not rows:
        raise SystemExit(f"no runs with metrics under {runs_root}")

    def write(path: Path, records: list[dict]) -> None:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        print(f"wrote {path} ({len(records)} runs)")

    write(out_dir / "sweep.csv", rows)
    write(out_dir / "per_class_iou.csv", per_class)


if __name__ == "__main__":
    main()
