# ruff: noqa
"""Tests for the Week 2 beginner plotting on-ramp."""

from __future__ import annotations

from pathlib import Path

from fins2026.week2.code.beginner_plotting import (
    DEFAULT_TICKER,
    export_ft_time_series,
    export_simple_time_series,
    make_synthetic_price_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMP_ROOT = REPO_ROOT / ".tmp-beginner-plotting-tests"


def test_synthetic_price_frame_shape_and_order() -> None:
    frame = make_synthetic_price_frame()

    assert list(frame.columns) == ["date", "ticker", "price"]
    assert len(frame) == 50
    assert frame["date"].is_monotonic_increasing
    assert set(frame["ticker"]) == {DEFAULT_TICKER}
    assert (frame["price"] > 0).all()


def test_simple_export_writes_png_and_pdf() -> None:
    frame = make_synthetic_price_frame()
    output_dir = TEMP_ROOT / "simple"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = export_simple_time_series(frame, output_dir)

    for suffix in ["png", "pdf"]:
        assert suffix in paths
        assert paths[suffix].exists()
        assert paths[suffix].stat().st_size > 0


def test_ft_export_writes_caption_bundle() -> None:
    frame = make_synthetic_price_frame()
    output_dir = TEMP_ROOT / "ft"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = export_ft_time_series(frame, output_dir)

    for suffix in ["png", "pdf", "caption"]:
        assert suffix in paths
        assert paths[suffix].exists()
        assert paths[suffix].stat().st_size > 0
    assert "Synthetic Price Series" in paths["caption"].read_text(encoding="utf-8")
