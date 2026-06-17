# ruff: noqa
"""Australia-first macro forecast app entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "fintools").is_dir()),
    Path(__file__).resolve().parents[3],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fins2026.week3.app.app_config import *  # noqa: F403,E402
from fins2026.week3.app.app_data import *  # noqa: F403,E402
from fins2026.week3.app.app_insights import *  # noqa: F403,E402
from fins2026.week3.app.app_views import (  # noqa: E402,F401
    main,
    render_equation,
    render_forecast_controls,
)

if __name__ == "__main__":
    main()
