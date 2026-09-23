#!/usr/bin/env python3
"""Fetch the trained model weights from this repository's GitHub release.

    python scripts/fetch/checkpoints.py            # both
    python scripts/fetch/checkpoints.py --only base

Two checkpoints are released, and they are the only two the report quotes:

``deepglobe_unet_base.keras``
    U-Net / EfficientNetB0 trained on DeepGlobe at 2 m/px. The starting point
    of every fine-tuning arm, and the zero-shot baseline on its own.

``finetune_lr3e-4_best.keras``
    The best fine-tuning arm, lr 3e-4, 82,0% mIoU under the protocol. Load this
    one to reproduce the reported segmentation without training anything.

Nothing else is released. The other sweep runs exist only as metrics, which are
committed under ``results/`` -- thirteen more checkpoints would be 1,5 GB of
weights nobody would load twice.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from _common import download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pitcic import config  # noqa: E402

REPO = "jvpcms/pitcic-segmentation"
TAG = "v1.0.0"
WANTED = {"base": "deepglobe_unet_base.keras", "finetuned": "finetune_lr3e-4_best.keras"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(WANTED), help="fetch one of them")
    ap.add_argument("--tag", default=TAG, help=f"release tag (default {TAG})")
    args = ap.parse_args()

    keys = [args.only] if args.only else sorted(WANTED)
    todo = {k: config.resolve(f"checkpoints.{k}") for k in keys}
    if all(p.exists() for p in todo.values()):
        for k, p in todo.items():
            print(f"have  {k}: {p}")
        return

    url = f"https://api.github.com/repos/{REPO}/releases/tags/{args.tag}"
    print(f"release {REPO}@{args.tag}")
    try:
        with urllib.request.urlopen(url) as resp:
            rel = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"cannot read the release ({exc}).\n"
            f"  check it exists: https://github.com/{REPO}/releases/tag/{args.tag}"
        )
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}

    for key, dest in todo.items():
        name = WANTED[key]
        if name not in assets:
            raise SystemExit(
                f"{name} is not on release {args.tag}; it carries {sorted(assets)}"
            )
        download(assets[name], dest)


if __name__ == "__main__":
    main()
