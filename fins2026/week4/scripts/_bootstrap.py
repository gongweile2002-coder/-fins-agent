# ruff: noqa
"""Small helper for Week 4 scripts that need repo-root imports."""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root from the current script path."""

    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "tools" / "workflow_lib.py"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find the fins-agent repo root.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
