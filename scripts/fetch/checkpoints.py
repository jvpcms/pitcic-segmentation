#!/usr/bin/env python3
"""Fetch the trained model weights from this repository's GitHub release.

    python scripts/fetch/checkpoints.py

One checkpoint is released, and it is the one every headline number belongs to:

``finetune_lr3e-4_best.keras``
    The best fine-tuning arm, lr 3e-4, 82,0% mIoU under the protocol. Load it to
    reproduce the reported segmentation without training anything.

The DeepGlobe base model is **not** released. DeepGlobe is distributed under
DigitalGlobe's Internal Use License Agreement, which grants a licence for the
licensee's internal use and prohibits distributing the products or derivatives
to third parties. Weights trained on that imagery are not clearly outside that
prohibition, so they are not redistributed here. Rebuild the base model with
``make base``, which downloads DeepGlobe from its source under the terms you
accept there. Reproducing the fine-tuning arms needs that step; reproducing the
reported evaluation does not.

The other sweep runs are not released either, for a different reason: their
metrics are committed under ``results/`` and thirteen more checkpoints would be
1,5 GB of weights nobody would load twice.
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
ASSET = "finetune_lr3e-4_best.keras"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=TAG, help=f"release tag (default {TAG})")
    args = ap.parse_args()

    dest = config.resolve("checkpoints.finetuned")
    if dest.exists():
        print(f"have  {dest}")
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
    if ASSET not in assets:
        raise SystemExit(
            f"{ASSET} is not on release {args.tag}; it carries {sorted(assets)}"
        )
    download(assets[ASSET], dest)


if __name__ == "__main__":
    main()
