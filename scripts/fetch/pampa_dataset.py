#!/usr/bin/env python3
"""Fetch the hand-annotated CBERS-4A dataset from Zenodo.

    python scripts/fetch/pampa_dataset.py

Resolves the version DOI recorded in ``conf/paths.yaml`` through the Zenodo API,
downloads the archive, checks it against the checksum Zenodo publishes for it,
and unpacks it under ``pampa.root``. Then runs the dataset's own
``scripts/verify.py --skip-reproduce``, which checks that every manifest row has
its files and that the masks use only palette colours, so a partial unpack is
caught here rather than three hours into a training run.

``--full-verify`` drops ``--skip-reproduce`` and additionally re-derives each
PNG from its GeoTIFF and the manifest stretch parameters. That needs the ``geo``
extra installed and takes several minutes; it is how you confirm the imagery
itself is intact, not merely present.

The version DOI is pinned on purpose: it is the exact data every number in the
report was computed from. The concept DOI in the same config always points at
the newest release and is what to cite, not what to train on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from _common import download, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pitcic import config  # noqa: E402

API = "https://zenodo.org/api/records/"


def record_id(doi: str) -> str:
    """10.5281/zenodo.22922626 -> 22922626."""
    tail = doi.rsplit(".", 1)[-1]
    if not tail.isdigit():
        raise SystemExit(f"cannot read a Zenodo record id out of {doi!r}")
    return tail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--full-verify",
        action="store_true",
        help="also re-derive every PNG from its GeoTIFF (needs the geo extra)",
    )
    args = ap.parse_args()

    paths = config.paths()
    doi = paths["pampa"]["doi"]
    root = config.resolve("pampa.root")
    if (root / "manifest.csv").exists():
        print(f"have  {root} -- delete it to re-fetch")
        return

    url = API + record_id(doi)
    print(f"doi   {doi}")
    with urllib.request.urlopen(url) as resp:
        rec = json.load(resp)
    print(f"title {rec['metadata']['title']}")

    files = rec.get("files", [])
    if len(files) != 1:
        names = [f["key"] for f in files]
        raise SystemExit(f"expected one archive on the record, found {names}")
    entry = files[0]
    algo, _, digest = entry["checksum"].partition(":")

    cache = config.resolve("data_root") / "_downloads"
    archive = download(entry["links"]["self"], cache / entry["key"])
    if algo == "md5":
        import hashlib
        h = hashlib.md5()
        with archive.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        got = h.hexdigest()
    elif algo == "sha256":
        got = sha256(archive)
    else:
        raise SystemExit(f"unhandled checksum algorithm {algo!r}")
    if got != digest:
        raise SystemExit(f"{algo} mismatch: expected {digest}, got {got}")
    print(f"{algo} ok")

    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"unpack -> {root}")
    with zipfile.ZipFile(archive) as zf:
        # GitHub-sourced Zenodo archives wrap everything in one top directory.
        tops = {Path(n).parts[0] for n in zf.namelist()}
        if len(tops) == 1:
            zf.extractall(root.parent / "_unpack")
            (root.parent / "_unpack" / tops.pop()).rename(root)
            (root.parent / "_unpack").rmdir()
        else:
            zf.extractall(root)

    verify = root / "scripts" / "verify.py"
    if not verify.exists():
        print("note: no verify.py in the archive, skipping the integrity check")
        return
    cmd = [sys.executable, str(verify)]
    if not args.full_verify:
        cmd.append("--skip-reproduce")
    print("verify " + " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
