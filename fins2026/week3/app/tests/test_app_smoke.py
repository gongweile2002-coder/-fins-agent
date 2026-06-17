# ruff: noqa
"""Smoke test for the Australia macro forecast app."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
VIEWS = {
    "Overview": "Overview",
    "Australia Snapshot": "Australia Snapshot",
    "Forecasts": "Forecasts",
    "Model Comparison": "Model Comparison",
    "Backtests": "Backtests",
    "U.S. Context": "U.S. Context",
    "Data": "Data",
    "Methodology": "Methodology",
}


def test_week3_streamlit_app_smoke(monkeypatch) -> None:
    if platform.system() == "Windows" and not os.environ.get("RUN_STREAMLIT_APPTEST_ON_WINDOWS"):
        pytest.skip(
            "Streamlit AppTest can leave locked temp files on native Windows; "
            "run in Linux CI or set RUN_STREAMLIT_APPTEST_ON_WINDOWS=1."
        )

    temp_root = ROOT / ".tmp-streamlit-app-test"
    temp_root.mkdir(exist_ok=True)
    monkeypatch.setenv("TMP", str(temp_root))
    monkeypatch.setenv("TEMP", str(temp_root))
    tempfile.tempdir = str(temp_root)

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    for view, expected_text in VIEWS.items():
        at = AppTest.from_file(app_path, default_timeout=30)
        at.query_params["view"] = view
        at.query_params["sample"] = "20Y"
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
        assert expected_text in rendered_text
        assert "Fixture snapshot through Australia monthly" in rendered_text
    shutil.rmtree(temp_root, ignore_errors=True)


def test_week3_streamlit_app_smoke_nondefault_model(monkeypatch) -> None:
    if platform.system() == "Windows" and not os.environ.get("RUN_STREAMLIT_APPTEST_ON_WINDOWS"):
        pytest.skip(
            "Streamlit AppTest can leave locked temp files on native Windows; "
            "run in Linux CI or set RUN_STREAMLIT_APPTEST_ON_WINDOWS=1."
        )

    temp_root = ROOT / ".tmp-streamlit-app-test"
    temp_root.mkdir(exist_ok=True)
    monkeypatch.setenv("TMP", str(temp_root))
    monkeypatch.setenv("TEMP", str(temp_root))
    tempfile.tempdir = str(temp_root)

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.query_params["view"] = "Forecasts"
    at.query_params["sample"] = "20Y"
    at.query_params["forecast"] = "Cash rate target"
    at.query_params["model"] = "armax"
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
    assert "Forecasts" in rendered_text
    assert "1 step only" in rendered_text
    assert "ARMA + exog" in rendered_text
    shutil.rmtree(temp_root, ignore_errors=True)
