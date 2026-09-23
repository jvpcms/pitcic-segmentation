"""Path and hyperparameter resolution.

Every path in this project comes from ``conf/paths.yaml`` and every path there
points inside ``data/`` or ``artifacts/``, which are gitignored. Nothing is
hardcoded to a machine, and nothing that a script can download is committed.

Override any scalar with an environment variable named ``PITCIC_`` plus the
dotted key in upper snake case, e.g. ``PITCIC_DATA_ROOT`` or
``PITCIC_DEEPGLOBE_ROOT``. That is how you point a run at a scratch disk
without editing tracked files.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"


def _env_override(key_path: tuple[str, ...], value: Any) -> Any:
    name = "PITCIC_" + "_".join(key_path).upper()
    return os.environ.get(name, value)


def _walk(node: Any, prefix: tuple[str, ...] = ()) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, prefix + (k,)) for k, v in node.items()}
    return _env_override(prefix, node)


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load ``conf/<name>.yaml`` with environment overrides applied."""
    path = CONF_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config {path}")
    with path.open(encoding="utf-8") as fh:
        return _walk(yaml.safe_load(fh))


def paths() -> dict:
    return load("paths")


def resolve(dotted: str) -> Path:
    """``resolve("pampa.root")`` -> absolute Path, relative to the repo root."""
    node: Any = paths()
    for part in dotted.split("."):
        node = node[part]
    p = Path(node)
    return p if p.is_absolute() else REPO_ROOT / p


def require(dotted: str, hint: str = "") -> Path:
    """Like ``resolve`` but fails with an actionable message if absent.

    Missing data is the normal state of a fresh clone, so the error names the
    fetch script rather than surfacing a FileNotFoundError from deep inside a
    loader.
    """
    p = resolve(dotted)
    if not p.exists():
        msg = f"{dotted} not found at {p}"
        if hint:
            msg += f"\n  {hint}"
        raise FileNotFoundError(msg)
    return p
