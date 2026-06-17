# ruff: noqa
"""Build one Word-ready FT-style figure from a validation dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

DEFAULT_OUTPUT = Path("fins2026/week2/results/figures/style_gallery")


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

from fintools.datasets import load_validation_dataset  # noqa: E402
from fintools.figures import (  # noqa: E402
    FigureContext,
    WordFigureEntry,
    cumulative_returns_plot,
    export_word_figure,
    insert_figures_docx,
)


def sample_label(index) -> str:
    """Return a compact sample-period label."""

    return f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"


def resolve_output_dir(output: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute output directory."""

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def build_single_word_figure(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Path]:
    """Create one FT-style figure and a one-figure Word document."""

    output_dir = resolve_output_dir(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ff3 = load_validation_dataset("ff3_monthly")
    context = FigureContext(
        title="Growth Of One Dollar In The Market Factor",
        note=(
            "FT-style example using monthly Fama/French market excess returns. "
            "Gray bands denote NBER recessions."
        ),
        source=ff3.source,
        sample=sample_label(ff3.data.index),
        units="Growth of one dollar, log scale",
    )
    fig, _ = cumulative_returns_plot(
        ff3.data,
        "Mkt-RF",
        returns_are_percent=True,
        log_scale=True,
        title=context.title,
        profile="word_a4",
        style="ft",
    )
    bundle = export_word_figure(
        fig,
        output_dir,
        "week2_single_market_growth",
        context=context,
        spec="full_width",
    )
    plt.close(fig)

    docx_path = insert_figures_docx(
        [WordFigureEntry(bundle["png"], context=context)],
        output_dir / "week2_single_market_growth.docx",
        title="Week 2 Single Figure Example",
    )

    return {**bundle, "docx": docx_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build one Word-ready FT-style validation figure.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Repo-relative or absolute output folder. Defaults to "
            "fins2026/week2/results/figures/style_gallery."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the script."""

    args = parse_args(argv)
    paths = build_single_word_figure(args.output)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
