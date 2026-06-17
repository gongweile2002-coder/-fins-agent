# ruff: noqa
"""Build the Week 2 FT-style validation figure proof pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUTPUT = "fins2026/week2/results/figures/style_gallery"
DOCX_NAME = "validation_figures_ft.docx"


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root from this script location."""

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

from tools import workflow_lib  # noqa: E402


def resolve_output_docx(output: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Return the expected FT Word proof-pack path."""

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path / DOCX_NAME


def build_validation_gallery(output: str | Path = DEFAULT_OUTPUT) -> Path:
    """Generate the FT validation gallery and return the Word document path."""

    status, lines = workflow_lib.build_figure_examples(
        REPO_ROOT,
        REPO_ROOT,
        output=str(output),
        docx=True,
        style="ft",
    )
    for line in lines:
        print(line)
    if status != 0:
        raise SystemExit(status)

    docx_path = resolve_output_docx(output)
    print(f"Word proof pack: {docx_path}")
    return docx_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build the Week 2 FT-style validation figure proof pack.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Repo-relative or absolute output folder. Defaults to "
            "fins2026/week2/results/figures/style_gallery."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the script."""

    args = parse_args(argv)
    build_validation_gallery(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
