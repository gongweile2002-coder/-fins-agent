# ruff: noqa
"""Week 4 portfolio app entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "fintools").is_dir()),
    Path(__file__).resolve().parents[3],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fins2026.week4.app.app_views import main  # noqa: E402

if __name__ == "__main__":
    main()
