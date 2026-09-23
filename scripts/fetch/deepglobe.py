#!/usr/bin/env python3
"""Fetch the DeepGlobe Land Cover Classification dataset from Kaggle.

    python scripts/fetch/deepglobe.py

DeepGlobe is distributed under a licence that does not permit redistribution,
so this repository cannot mirror it and neither can the released artifacts. You
download it yourself, from the source, under the terms you accept there.

Needs Kaggle API credentials: create a token at kaggle.com/settings and put it
at ~/.kaggle/kaggle.json (chmod 600), or export KAGGLE_USERNAME and KAGGLE_KEY.
You must also have accepted the competition rules on the dataset page once,
in a browser, or the API returns 403.

What lands on disk is the raw release: 803 training images of 2448x2448 with
their RGB masks. ``make deepglobe-tiles`` turns that into the 256 px training
set actually used, resampled to 2 m/px to match CBERS-4A WPM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pitcic import config  # noqa: E402

SLUG = "balraj98/deepglobe-land-cover-classification-dataset"


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> None:
    root = config.resolve("deepglobe.root")
    if (root / "train").exists():
        print(f"have  {root} -- delete it to re-fetch")
        return
    if not have_credentials():
        raise SystemExit(
            "no Kaggle credentials found.\n"
            "  put a token at ~/.kaggle/kaggle.json, or export "
            "KAGGLE_USERNAME and KAGGLE_KEY.\n"
            f"  dataset page: https://www.kaggle.com/datasets/{SLUG}"
        )

    cache = config.resolve("data_root") / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"kaggle datasets download -d {SLUG}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(cache)],
            check=True,
        )
    except FileNotFoundError:
        raise SystemExit("the kaggle CLI is not installed: pip install kaggle")

    archives = sorted(cache.glob("*deepglobe*.zip"))
    if not archives:
        raise SystemExit(f"the download left no zip in {cache}")
    root.mkdir(parents=True, exist_ok=True)
    print(f"unpack {archives[-1].name} -> {root}")
    with zipfile.ZipFile(archives[-1]) as zf:
        zf.extractall(root)
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
