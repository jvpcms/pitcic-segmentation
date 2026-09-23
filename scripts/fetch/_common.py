"""Shared download helpers for the fetch scripts.

Every fetch script is idempotent: it checks what is already on disk before
touching the network, and it verifies what it wrote. Re-running one is always
safe and usually free.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download ``url`` to ``dest``, skipping the transfer if it is already there.

    Writes to a ``.part`` file and renames only after the checksum passes, so an
    interrupted run never leaves a truncated file that a later run would accept.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_sha256 is None or sha256(dest) == expected_sha256:
            print(f"have  {dest}")
            return dest
        print(f"stale {dest}, re-downloading")

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get   {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}  {done >> 20} MiB", end="", flush=True)
        print()

    if expected_sha256 is not None:
        got = sha256(tmp)
        if got != expected_sha256:
            tmp.unlink()
            raise SystemExit(
                f"checksum mismatch for {url}\n  expected {expected_sha256}\n  got      {got}"
            )
    tmp.rename(dest)
    print(f"wrote {dest}")
    return dest
