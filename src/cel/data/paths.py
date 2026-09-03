"""Filesystem path resolution for raw and cached datasets.

Paths are derived from the repository root by default, so nothing here is
hard-coded to a user-specific location (OUTLINE.md workflow rule). Override the
data root with the ``CASCADE_DATA_DIR`` environment variable, or pass explicit
paths to the loaders.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_DIR = "CASCADE_DATA_DIR"


def repo_root() -> Path:
    """Return the repository root.

    This module lives at ``src/cel/data/paths.py``, so the root is four
    parents up.
    """
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Return the data root (``$CASCADE_DATA_DIR`` or ``<repo>/data``)."""
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "data"


def raw_dir() -> Path:
    """Return ``<data>/raw`` (downloaded source files, gitignored)."""
    return data_dir() / "raw"


def cache_dir() -> Path:
    """Return ``<data>/cache`` (parsed artifacts, gitignored)."""
    return data_dir() / "cache"
