# ruff: noqa
"""Summarize the current data files for this week."""

from __future__ import annotations

from pathlib import Path

WEEK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_ROOT / 'data'
RESULTS_DATA_DIR = WEEK_ROOT / 'results' / 'data'


def visible_files(directory: Path) -> list[Path]:
    """Return non-placeholder files inside a directory tree."""

    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob('*')
        if path.is_file() and path.name != '.gitkeep'
    )


def describe_directory(label: str, directory: Path) -> list[str]:
    """Return a short inventory for one week data directory."""

    files = visible_files(directory)
    lines = [f'{label}: {directory.relative_to(WEEK_ROOT).as_posix()}']
    if not files:
        lines.append('- no files yet')
        return lines
    for path in files:
        rel = path.relative_to(WEEK_ROOT).as_posix()
        lines.append(f'- {rel} ({path.stat().st_size} bytes)')
    return lines


def describe_week_data() -> str:
    """Return a plain-text summary of source and generated datasets."""

    lines = ['Week data inventory', '']
    lines.extend(describe_directory('Source data', DATA_DIR))
    lines.append('')
    lines.extend(describe_directory('Generated data', RESULTS_DATA_DIR))
    return '\n'.join(lines)


def main() -> None:
    print(describe_week_data())


if __name__ == '__main__':
    main()

