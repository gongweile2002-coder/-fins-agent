# ruff: noqa
"""Summarize the data sources used in Week 3."""

from __future__ import annotations

from pathlib import Path

from fins2026.week3.code import AUSTRALIA_FORECAST_SPECS

WEEK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_ROOT / "data"
RESULTS_DATA_DIR = WEEK_ROOT / "results" / "data"


def visible_files(directory: Path) -> list[Path]:
    """Return non-placeholder files inside a directory tree."""

    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )


def describe_directory(label: str, directory: Path) -> list[str]:
    """Return a short inventory for one week data directory."""

    files = visible_files(directory)
    lines = [f"{label}: {directory.relative_to(WEEK_ROOT).as_posix()}"]
    if not files:
        lines.append("- no files yet")
        return lines
    for path in files:
        rel = path.relative_to(WEEK_ROOT).as_posix()
        lines.append(f"- {rel} ({path.stat().st_size} bytes)")
    return lines


def describe_week_data() -> str:
    """Return a plain-text summary of source and generated datasets."""

    monthly = [
        spec.label for spec in AUSTRALIA_FORECAST_SPECS.values() if spec.frequency == "monthly"
    ]
    quarterly = [
        spec.label
        for spec in AUSTRALIA_FORECAST_SPECS.values()
        if spec.frequency == "quarterly"
    ]
    lines = ["Week 3 data inventory", ""]
    lines.extend(describe_directory("Source data", DATA_DIR))
    lines.append("")
    lines.append("Week 3 app-backed sources:")
    lines.append("- Australia primary data comes from the Week 2 RBA Stage 1 timing pipeline.")
    lines.append(
        "- The app and benchmark scripts use the observable Australia month-end panel "
        "as the forecasting source of truth."
    )
    lines.append(
        "- U.S. secondary data comes from the Week 2 FRED month-end panel and is "
        "used for context plus selected exogenous features."
    )
    lines.append(
        "- Fixture mode uses the committed Australia Stage 1 file plus frozen U.S. "
        "validation datasets."
    )
    lines.append(
        "- Live mode rebuilds Australia inputs from RBA tables and U.S. inputs from "
        "FRED before falling back if needed."
    )
    lines.append("")
    lines.append("Week 3 forecast targets:")
    lines.append(f"- Monthly: {', '.join(monthly)}")
    lines.append(f"- Quarterly: {', '.join(quarterly)}")
    lines.append("")
    lines.extend(describe_directory("Generated data", RESULTS_DATA_DIR))
    return "\n".join(lines)


def main() -> None:
    print(describe_week_data())


if __name__ == "__main__":
    main()
