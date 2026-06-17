# ruff: noqa
"""Pull official Australian macro data for the Week 2 Australia figure pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("fins2026/week2/results/data")
FIXTURE_LONG_TABLE = Path("fins2026/week2/data/australia_macro_stage1_long.csv")


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
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [REPO_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fins2026.week2.code.australia_macro_panel import (  # noqa: E402
    build_feature_panel,
    build_monthly_native_panel,
    build_observable_panel,
    build_quarterly_native_panel,
    build_reference_panel,
    build_stage1_long_table,
    download_live_rba_bundles,
    load_fixture_long_table,
)
from fins2026.week2.code.australia_macro_specs import (  # noqa: E402
    CLASSROOM_INFORMATION_SET_MONTH_END,
    CLASSROOM_REFERENCE_ENDPOINT,
)


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute path."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def load_live_stage1_long_table() -> pd.DataFrame:
    """Download live RBA tables and build the Stage 1 long table."""

    bundles = download_live_rba_bundles()
    return build_stage1_long_table(bundles)


def build_output_bundle(stage1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the Stage 2 derived panels from the Stage 1 long table."""

    reference_panel = build_reference_panel(stage1)
    observable_panel = build_observable_panel(stage1)
    feature_panel = build_feature_panel(reference_panel)
    return {
        "stage1_long": stage1,
        "monthly_native_panel": build_monthly_native_panel(stage1),
        "quarterly_native_panel": build_quarterly_native_panel(stage1),
        "reference_panel": reference_panel,
        "observable_panel": observable_panel,
        "feature_panel": feature_panel,
    }


def write_frame_csv_and_parquet(frame: pd.DataFrame, target: Path) -> dict[str, Path]:
    """Write one dataframe to CSV and Parquet with a shared stem."""

    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path = target.with_suffix(".csv")
    parquet_path = target.with_suffix(".parquet")
    csv_frame = frame.reset_index() if frame.index.name else frame.copy()
    csv_frame.to_csv(csv_path, index=False)
    if frame.index.name:
        frame.to_parquet(parquet_path)
    else:
        frame.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "parquet": parquet_path}


def write_australia_macro_bundle(
    bundle: dict[str, pd.DataFrame],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write the Stage 1 and Stage 2 Australia macro outputs under results/data."""

    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    stems = {
        "stage1_long": "australia_macro_stage1_long",
        "monthly_native_panel": "australia_macro_monthly_native_panel",
        "quarterly_native_panel": "australia_macro_quarterly_native_panel",
        "reference_panel": "australia_macro_reference_panel",
        "observable_panel": "australia_macro_observable_panel",
        "feature_panel": "australia_macro_feature_panel",
    }
    for key, stem in stems.items():
        paths = write_frame_csv_and_parquet(bundle[key], output_root / stem)
        written[f"{key}_csv"] = paths["csv"]
        written[f"{key}_parquet"] = paths["parquet"]
    return written


def write_fixture_long_table(stage1: pd.DataFrame, path: str | Path = FIXTURE_LONG_TABLE) -> Path:
    """Refresh the committed Stage 1 fixture path from a live pull."""

    fixture_path = resolve_repo_path(path)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(fixture_path, index=False)
    return fixture_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Pull official Australian macro data for Week 2 figures.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Repo-relative or absolute output folder under results/data.",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use the committed Stage 1 Australia fixture instead of live RBA downloads.",
    )
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="Refresh the committed Stage 1 Australia fixture from the live pull.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Australia macro pull and panel build."""

    args = parse_args(argv)
    try:
        stage1 = (
            load_fixture_long_table(resolve_repo_path(FIXTURE_LONG_TABLE))
            if args.use_fixture
            else load_live_stage1_long_table()
        )
    except Exception as exc:
        print(f"Could not load Australia macro data: {exc}")
        print("Try again online, or rerun with --use-fixture for the offline class path.")
        return 1

    if not args.use_fixture and args.write_fixture:
        fixture_path = write_fixture_long_table(stage1)
        print(f"Updated fixture: {fixture_path}")

    bundle = build_output_bundle(stage1)
    written = write_australia_macro_bundle(bundle, args.output_dir)
    print(
        "Built Week 2 Australia macro outputs with the frozen classroom timing "
        f"contract: reference endpoint {CLASSROOM_REFERENCE_ENDPOINT:%Y-%m-%d}, "
        f"information set month-end {CLASSROOM_INFORMATION_SET_MONTH_END:%Y-%m-%d}."
    )
    for label, path in written.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
