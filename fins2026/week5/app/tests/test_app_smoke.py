# ruff: noqa
"""Smoke test for the Week 5 client-facing crypto fund app."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
VIEWS = {
    "Overview": "Published fund lineup",
    "Fund Details": "Current live holdings",
    "Invest": "Illustrative allocation",
    "Portfolio Design": "Portfolio design comparison",
    "Data & Downloads": "Data & downloads",
    "Methodology": "Methodology",
}


def test_week5_streamlit_app_smoke(monkeypatch) -> None:
    if platform.system() == "Windows" and not os.environ.get("RUN_STREAMLIT_APPTEST_ON_WINDOWS"):
        pytest.skip(
            "Streamlit AppTest can leave locked temp files on native Windows; "
            "run in Linux CI or set RUN_STREAMLIT_APPTEST_ON_WINDOWS=1."
        )

    temp_root = ROOT / ".tmp-streamlit-week5-app-test"
    temp_root.mkdir(exist_ok=True)
    monkeypatch.setenv("TMP", str(temp_root))
    monkeypatch.setenv("TEMP", str(temp_root))
    monkeypatch.setenv("WEEK5_APP_FORCE_FIXTURE", "1")
    tempfile.tempdir = str(temp_root)

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    for view, expected_text in VIEWS.items():
        at = AppTest.from_file(app_path, default_timeout=80)
        at.query_params["view"] = view
        at.query_params["sample"] = "3Y"
        at.query_params["fund"] = "mean_variance_tangency_long_only"
        at.query_params["amount"] = "25000"
        at.query_params["design_freq"] = "monthly"
        at.query_params["design_window"] = "365"
        at.query_params["design_rule"] = "expanding"
        at.run()
        assert not at.exception, f"{view} tab raised: {at.exception}"
        rendered_text = "\n".join(
            str(element.value)
            for collection in [
                at.title,
                at.subheader,
                at.caption,
                at.markdown,
                at.info,
                at.warning,
                at.button,
                getattr(at, "download_button", []),
            ]
            for element in collection
        )
        assert "Digital Asset Fund Explorer" in rendered_text
        assert expected_text in rendered_text
        assert "Fallback snapshot through prices" in rendered_text
        assert "portfolio_key" not in rendered_text
        assert "constraint_mode" not in rendered_text
    shutil.rmtree(temp_root, ignore_errors=True)


def test_week5_streamlit_app_design_view_weekly(monkeypatch) -> None:
    if platform.system() == "Windows" and not os.environ.get("RUN_STREAMLIT_APPTEST_ON_WINDOWS"):
        pytest.skip(
            "Streamlit AppTest can leave locked temp files on native Windows; "
            "run in Linux CI or set RUN_STREAMLIT_APPTEST_ON_WINDOWS=1."
        )

    temp_root = ROOT / ".tmp-streamlit-week5-app-test"
    temp_root.mkdir(exist_ok=True)
    monkeypatch.setenv("TMP", str(temp_root))
    monkeypatch.setenv("TEMP", str(temp_root))
    monkeypatch.setenv("WEEK5_APP_FORCE_FIXTURE", "1")
    tempfile.tempdir = str(temp_root)

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app_path, default_timeout=80)
    at.query_params["view"] = "Portfolio Design"
    at.query_params["sample"] = "1Y"
    at.query_params["fund"] = "minimum_variance_long_only"
    at.query_params["amount"] = "10000"
    at.query_params["design_freq"] = "weekly"
    at.query_params["design_window"] = "180"
    at.query_params["design_rule"] = "rolling"
    at.run()
    assert not at.exception, at.exception
    rendered_text = "\n".join(
        str(element.value)
        for collection in [
            at.title,
            at.subheader,
            at.caption,
            at.markdown,
            at.info,
            at.warning,
            at.button,
            getattr(at, "download_button", []),
        ]
        for element in collection
    )
    assert "Portfolio design comparison" in rendered_text
    assert "These settings are a design comparison only." in rendered_text
    shutil.rmtree(temp_root, ignore_errors=True)
