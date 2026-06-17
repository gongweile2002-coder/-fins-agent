# ruff: noqa
"""Week 1 scaffold and dataset smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

WEEK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_ROOT / "data"
GUIDANCE_DIR = WEEK_ROOT / "guidance"
WORKSHOP_CSV = DATA_DIR / "week1_workshop_panel.csv"
WORKSHOP_PARQUET = DATA_DIR / "week1_workshop_panel.parquet"
ASSIGNMENT_TSV = DATA_DIR / "week1_assignment_data.txt"


def test_week_scaffold_smoke() -> None:
    for relative in [
        'README.md',
        'WORKSHOP.md',
        'DATA_GUIDE.md',
        'PRACTICE_GUIDE.md',
        'SUBMISSION_CHECKLIST.md',
        'AGENTS.md',
        'CLAUDE.md',
        'GEMINI.md',
        'QWEN.md',
        'prompts/README.md',
        'prompts/assistant_starter.md',
        'prompts/workshop_walkthrough.md',
        'prompts/practice1_coach.md',
        'guidance/week-context.md',
        'guidance/data-context.md',
        'guidance/output-context.md',
        'scripts/run_week.py',
        'scripts/describe_data.py',
        'scripts/06_coke_pepsi_practice.py',
        'data/README.md',
        'data/week1_workshop_panel.csv',
        'data/week1_workshop_panel.parquet',
        'data/week1_assignment_data.txt',
        'scratch/README.md',
    ]:
        assert (WEEK_ROOT / relative).exists(), relative


def test_week1_prompt_loaders_include_output_context() -> None:
    for relative in [
        'AGENTS.md',
        'CLAUDE.md',
        'GEMINI.md',
        'QWEN.md',
        'prompts/assistant_starter.md',
    ]:
        text = (WEEK_ROOT / relative).read_text(encoding='utf-8')
        assert 'guidance/output-context.md' in text, relative


def test_week1_generated_week_context_mentions_prompt_files() -> None:
    text = (GUIDANCE_DIR / 'week-context.md').read_text(encoding='utf-8')
    assert '## Prompt Files' in text
    assert 'prompts/assistant_starter.md' in text
    assert 'prompts/workshop_walkthrough.md' in text
    assert 'prompts/practice1_coach.md' in text


def test_week1_public_text_is_student_facing() -> None:
    internal_dir = WEEK_ROOT / ('_' + 'solutions')
    banned_markers = ['_' + 'solutions', 'maint' + 'ainer']

    assert not internal_dir.exists()

    for path in WEEK_ROOT.rglob('*'):
        if not path.is_file():
            continue
        if '__pycache__' in path.parts:
            continue
        if path.suffix.lower() not in {'.md', '.py'}:
            continue
        text = path.read_text(encoding='utf-8')
        lowered = text.lower()
        for marker in banned_markers:
            assert marker not in lowered, path.relative_to(WEEK_ROOT).as_posix()


def test_week1_workshop_dataset_facts() -> None:
    raw = pd.read_csv(WORKSHOP_CSV, parse_dates=['Date'], dayfirst=True)
    assert raw.shape == (26159, 6)
    assert int(raw.duplicated(['Date', 'Ticker']).sum()) == 3

    clean = pd.read_parquet(WORKSHOP_PARQUET)
    assert clean.shape == (26156, 6)
    assert int(clean.duplicated(['Date', 'Ticker']).sum()) == 0
    assert clean.groupby('Ticker').size().to_dict() == {
        'AAPL': 6539,
        'MSFT': 6539,
        'NVDA': 6539,
        'ORCL': 6539,
    }


def test_week1_practice_dataset_facts() -> None:
    panel = pd.read_csv(ASSIGNMENT_TSV, sep='\t')
    assert panel.shape == (13078, 10)

    panel['Date'] = pd.to_datetime(panel['DlyCalDt'], format='%Y%m%d')
    assert int(panel.duplicated(['Date', 'Ticker']).sum()) == 0
    assert int(panel.isna().sum().sum()) == 0
    assert panel.groupby('Ticker').size().to_dict() == {'KO': 6539, 'PEP': 6539}

    cross_section = (
        panel.loc[panel['Date'] == pd.Timestamp('2020-03-16'), ['Ticker', 'DlyRet']]
        .sort_values('Ticker')
        .reset_index(drop=True)
    )
    assert len(cross_section) == 2
    returns = dict(zip(cross_section['Ticker'], cross_section['DlyRet'], strict=True))
    assert returns['KO'] == pytest.approx(-0.066227, abs=1e-6)
    assert returns['PEP'] == pytest.approx(-0.112672, abs=1e-6)

    wide = panel.pivot(index='Date', columns='Ticker', values='DlyRet').dropna()
    growth = (1 + wide).cumprod().iloc[-1]
    assert growth['KO'] == pytest.approx(4.93, abs=0.03)
    assert growth['PEP'] == pytest.approx(7.90, abs=0.03)
    assert (growth['PEP'] / growth['KO']) == pytest.approx(1.60, abs=0.03)

