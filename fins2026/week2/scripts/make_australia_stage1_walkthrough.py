# ruff: noqa
"""Walk through Week 2 Australia Data Factory Floor Station 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT = Path("fins2026/week2/results/data/australia_macro_stage1_long.csv")
FIXTURE_LONG_TABLE = Path("fins2026/week2/data/australia_macro_stage1_long.csv")
WALKTHROUGH_SERIES = ["GCPIAGYP", "GGDPCVGDP", "GLFSURSA", "FXRTWI"]


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
    RBA_CSV_URLS,
    RBA_HISTORICAL_XLS_URLS,
    build_stage1_long_table,
    download_live_rba_bundles,
    load_fixture_long_table,
)
from fins2026.week2.code.australia_macro_specs import (  # noqa: E402
    AUSTRALIA_MACRO_SPECS,
    CLASSROOM_REFERENCE_ENDPOINT,
    CLASSROOM_SAMPLE_START,
    required_source_tables,
)


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute path."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def source_url(table_code: str) -> str:
    """Return the official source URL for one RBA table code."""

    if table_code in RBA_CSV_URLS:
        return RBA_CSV_URLS[table_code]
    return RBA_HISTORICAL_XLS_URLS[table_code]


def stage1_output_paths(path: str | Path) -> tuple[Path, Path]:
    """Return the CSV and Parquet output paths for the walkthrough."""

    csv_path = resolve_repo_path(path)
    return csv_path, csv_path.with_suffix(".parquet")


def write_stage1_outputs(
    stage1: pd.DataFrame,
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Path]:
    """Write the Stage 1 long table to CSV and Parquet."""

    csv_path, parquet_path = stage1_output_paths(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(csv_path, index=False)
    stage1.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "parquet": parquet_path}


def print_step(title: str) -> None:
    """Print one walkthrough step heading."""

    print()
    print(title)
    print("-" * len(title))


def print_source_table_plan() -> None:
    """Print the official source tables used in the Station 1 pull."""

    print_step("Step 1: Pull Official Source Tables")
    print("Week 2 Station 1 uses these RBA tables:")
    for table_code in required_source_tables():
        print(f"- {table_code}: {source_url(table_code)}")


def print_series_contracts() -> None:
    """Print the key series contracts students should inspect before merging."""

    print_step("Step 2: Check Series Contracts")
    for series_id in WALKTHROUGH_SERIES:
        spec = AUSTRALIA_MACRO_SPECS[series_id]
        print(
            "- "
            f"{series_id} | {spec.display_name} | table {spec.source_table} | "
            f"{spec.native_frequency} | reference rule {spec.reference_date_rule} | "
            f"observable lag {spec.observable_lag_months} month(s)"
        )


def print_download_summary(bundles: dict[str, object]) -> None:
    """Print a compact summary of downloaded RBA bundles."""

    print_step("Step 3: Download And Parse The Tables")
    for table_code in required_source_tables():
        bundle = bundles[table_code]
        publication = (
            f"{bundle.publication_date:%Y-%m-%d}"
            if not pd.isna(bundle.publication_date)
            else "unknown"
        )
        print(
            "- "
            f"{table_code}: {bundle.frame.shape[0]:,} dated rows, "
            f"{bundle.metadata.shape[0]:,} mapped series, "
            f"publication {publication}"
        )


def print_stage1_summary(stage1: pd.DataFrame) -> None:
    """Print the typed Stage 1 structure and timing fields."""

    print_step("Step 4: Build The Typed Stage 1 Long Table")
    print(
        "Station 1 clips the classroom sample to "
        f"{CLASSROOM_SAMPLE_START:%Y-%m-%d} through "
        f"{CLASSROOM_REFERENCE_ENDPOINT:%Y-%m-%d}."
    )
    print(f"- rows: {len(stage1):,}")
    print(f"- columns: {', '.join(stage1.columns)}")
    print("- source tables:")
    for table_code, count in stage1["source_table"].value_counts().sort_index().items():
        print(f"  - {table_code}: {count:,} rows")
    print("- native frequencies:")
    for frequency, count in stage1["native_frequency"].value_counts().sort_index().items():
        print(f"  - {frequency}: {count:,} rows")


def print_assertion_report(stage1: pd.DataFrame) -> None:
    """Print and enforce the Week 2 Stage 1 assertion contract."""

    print_step("Step 5: Run The Week 2 Stage 1 Assertions")
    expected_series = set(AUSTRALIA_MACRO_SPECS)
    actual_series = set(stage1["series_id"].dropna().unique())
    report = {
        "typed_dates": pd.api.types.is_datetime64_any_dtype(stage1["reference_date"])
        and pd.api.types.is_datetime64_any_dtype(stage1["release_date"])
        and pd.api.types.is_datetime64_any_dtype(stage1["observable_month_end"]),
        "typed_values": pd.api.types.is_numeric_dtype(stage1["value"]),
        "all_series_present": expected_series == actual_series,
        "ordered_dates": bool((stage1["reference_date"] <= stage1["release_date"]).all()),
        "unique_pairs": not bool(stage1.duplicated(["series_id", "reference_date"]).any()),
    }
    messages = {
        "typed_dates": "typed date fields",
        "typed_values": "numeric value column",
        "all_series_present": "all required series appear in the long table",
        "ordered_dates": "reference_date never exceeds release_date",
        "unique_pairs": "no duplicate (series_id, reference_date) pairs",
    }
    for key, label in messages.items():
        outcome = report[key]
        print(f"- {label}: {'PASS' if outcome else 'FAIL'}")
        assert outcome, label


def print_timing_preview(stage1: pd.DataFrame) -> None:
    """Print a small timing preview for representative series."""

    print_step("Step 6: Inspect Timing Fields")
    for series_id in WALKTHROUGH_SERIES:
        subset = stage1.loc[
            stage1["series_id"] == series_id,
            [
                "series_id",
                "display_name",
                "raw_date",
                "reference_date",
                "release_date",
                "observable_month_end",
                "value",
            ],
        ].head(3)
        print(subset.to_string(index=False))
        print()


def print_write_summary(paths: dict[str, Path]) -> None:
    """Print the saved output paths."""

    print_step("Step 7: Save The Stage 1 Output")
    for label, path in paths.items():
        print(f"- {label}: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Walk through the Week 2 Australia Stage 1 Data Factory Floor pipeline.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Repo-relative or absolute CSV path for the Stage 1 long table. "
            "A matching Parquet file is also written."
        ),
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help=(
            "Use the committed Stage 1 fixture for an offline walkthrough instead "
            "of downloading live RBA tables."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the student-facing Station 1 walkthrough."""

    args = parse_args(argv)
    print("Week 2 Australia Stage 1 Data Factory Floor Walkthrough")
    print("This script shows how to pull, type, time-stamp, and save the long table.")

    print_source_table_plan()
    print_series_contracts()

    if args.use_fixture:
        print_step("Step 3: Load The Offline Fixture")
        stage1 = load_fixture_long_table(resolve_repo_path(FIXTURE_LONG_TABLE))
        print(f"- fixture: {resolve_repo_path(FIXTURE_LONG_TABLE)}")
        print("- mode: offline walkthrough of the frozen Stage 1 output")
    else:
        bundles = download_live_rba_bundles()
        print_download_summary(bundles)
        stage1 = build_stage1_long_table(bundles)

    print_stage1_summary(stage1)
    print_assertion_report(stage1)
    print_timing_preview(stage1)
    paths = write_stage1_outputs(stage1, args.output)
    print_write_summary(paths)
    print()
    print("Stage 1 complete. The next step is to build reference, observable, and feature panels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
