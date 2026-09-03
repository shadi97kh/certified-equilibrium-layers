#!/usr/bin/env python3
"""Fail if synthetic-data generation appears on any AAAI reported-number path.

Per the hard data policy in OUTLINE.md, no synthetic data may live in
``src/cel/data/`` or ``experiments/aaai/`` (the loaders and the experiments
that feed reported AAAI numbers). This script greps those paths for
synthetic-generation patterns and exits non-zero if any appear.

Scanned paths:
- ``src/cel/data/`` (the real-data loaders)
- ``experiments/aaai/`` (the real-data experiments and the aggregator)

Flagged patterns (case-insensitive):
- random generation used to build features / labels / edges: ``np.random``,
  ``default_rng``, ``standard_normal``, ``rng.uniform``, ``rng.integers``,
  ``rng.choice``, ``rng.random``,
- the words ``synthetic``, ``plant``, ``proxy``, ``toy``.

A line is exempt only if it carries the marker ``# real-data-policy: allow``
(for the rare legitimate mention, e.g. an error message that names the policy).
The honest disclaimer "does not fabricate baseline numbers" is intentionally NOT
flagged: only data-construction patterns are.
Synthetic CERTIFICATE-SOUNDNESS verification lives under ``experiments/soundness/``
and is intentionally NOT scanned: it verifies the math on random graphs and is
never a reported biological number.

Usage: ``python scripts/assert_no_synthetic.py`` (exit 0 clean, 1 on any hit).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN_DIRS = [REPO / "src" / "cel" / "data", REPO / "experiments" / "aaai"]

PATTERNS = [
    r"np\.random",
    r"default_rng",
    r"standard_normal",
    r"rng\.uniform",
    r"rng\.integers",
    r"rng\.choice",
    r"rng\.random",
    r"\bsynthetic\b",
    r"\bplant\b",
    r"\bproxy\b",
    r"\btoy\b",
]
_RE = re.compile("|".join(PATTERNS), re.IGNORECASE)
ALLOW_MARKER = "# real-data-policy: allow"


def scan_file(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        if _RE.search(line):
            hits.append((lineno, line.strip()))
    return hits


def _report(all_hits: list[tuple[Path, int, str]]) -> int:
    if all_hits:
        print("HARD DATA POLICY VIOLATION: synthetic-generation patterns on a reported-number path")
        for path, lineno, line in all_hits:
            print(f"  {path}:{lineno}: {line}")
        print(
            f"\n{len(all_hits)} violation(s). Move synthetic verification to experiments/soundness/"
        )
        print(
            "or load real data and RAISE FileNotFoundError / NotImplementedError when it is absent."
        )
        return 1
    return 0


def _scan_default() -> int:
    """Scan the fixed reported-number code paths (src/cel/data, experiments/aaai)."""
    all_hits: list[tuple[Path, int, str]] = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in scan_file(path):
                all_hits.append((path.relative_to(REPO), lineno, line))
    rc = _report(all_hits)
    if rc == 0:
        print("OK: no synthetic-generation patterns in src/cel/data/ or experiments/aaai/")
    return rc


def _scan_paths(paths: list[str]) -> int:
    """Validate explicit paths: each MUST exist; .py files are scanned for synthetic
    -generation patterns; non-.py files are asserted to be real (present and non-empty).
    A missing path or an empty data file is a failure (exit 1)."""
    all_hits: list[tuple[Path, int, str]] = []
    n_py = n_data = 0
    for raw in paths:
        target = Path(raw)
        if not target.exists():
            print(f"PATH NOT FOUND: {raw} (no synthetic check possible; real data is absent)")
            return 1
        if target.is_file():
            py_files = [target] if target.suffix == ".py" else []
            data_files = [] if target.suffix == ".py" else [target]
        else:
            py_files = [p for p in target.rglob("*.py") if "__pycache__" not in p.parts]
            data_files = [p for p in target.rglob("*") if p.is_file() and p.suffix != ".py"]
        for p in py_files:
            n_py += 1
            rel = p.resolve().relative_to(REPO) if REPO in p.resolve().parents else p
            for lineno, line in scan_file(p):
                all_hits.append((rel, lineno, line))
        for d in data_files:
            if d.stat().st_size == 0:
                print(f"EMPTY DATA FILE: {d} (REAL DATA ONLY: a real file must be non-empty)")
                return 1
            n_data += 1
    rc = _report(all_hits)
    if rc == 0:
        print(
            f"OK: {n_data} real data file(s) present and non-empty, {n_py} .py file(s) scanned; "
            "no synthetic-generation patterns."
        )
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _scan_paths(argv) if argv else _scan_default()


if __name__ == "__main__":
    sys.exit(main())
