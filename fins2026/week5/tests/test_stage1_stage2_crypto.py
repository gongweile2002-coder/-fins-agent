# ruff: noqa
"""Offline tests for the Week 5 crypto helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fins2026.week5.code.crypto_api_yahoo import (
    build_yahoo_crypto_coverage_summary,
    load_yahoo_crypto_tickers_from_file,
    normalize_yahoo_crypto_chart_payload,
)
from fins2026.week5.code.risk_free_rate_french import parse_french_daily_rfr
from fins2026.week5.code.stage2_crypto_figures import (
    _build_cross_sectional_dispersion_frame,
    _build_daily_volume_share_wide,
    _build_headline_dollar_volume_wide,
    _build_headline_drawdown_wide,
    _build_relative_to_btc_wide,
    _build_rolling_corr_to_btc_wide,
    _build_trailing_dollar_volume_frame,
    make_stage2_figure_pack,
)
from fins2026.week5.code.stage2_crypto_returns import (
    assert_return_parity,
    build_adjusted_close_wide,
    build_feature_long_panel,
    compute_long_returns,
    compute_wide_returns,
    merge_daily_rfr,
    summarize_stage2_metrics,
)

HEADLINE_TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]


def _synthetic_figure_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic Stage 1 and Stage 2 panels for figure tests."""

    dates = pd.date_range("2024-01-01", periods=220, freq="D")
    stage1_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(HEADLINE_TICKERS):
        price = 100.0 + 30.0 * ticker_index
        volume = 500_000 + 100_000 * ticker_index
        for date_index, date in enumerate(dates):
            market_component = ((date_index % 11) - 5) * 0.004
            idiosyncratic_component = ((date_index + 2 * ticker_index) % 7 - 3) * 0.0015
            ret = 0.0006 * (ticker_index + 1) + 0.75 * market_component + idiosyncratic_component
            if date_index == 0:
                ret = np.nan
            if pd.notna(ret):
                price *= 1.0 + float(ret)
            stage1_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adjClose": price,
                    "volume": volume + 5_000 * date_index,
                }
            )
            feature_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "ret": ret,
                    "excess_ret": ret - 0.0001 if pd.notna(ret) else np.nan,
                }
            )

    return pd.DataFrame(stage1_rows), pd.DataFrame(feature_rows)


def test_load_yahoo_crypto_tickers_from_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    ticker_file = tmp_path / "crypto_tickers.txt"
    ticker_file.write_text("# comment\nBTC-USD\n\neth-usd\n", encoding="utf-8")
    assert load_yahoo_crypto_tickers_from_file(ticker_file) == ("BTC-USD", "ETH-USD")


def test_normalize_yahoo_crypto_chart_payload_drops_all_null_placeholder_rows() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": "BTC-USD",
                        "exchangeName": "CCC",
                        "instrumentType": "CRYPTOCURRENCY",
                        "firstTradeDate": 1546300800,
                        "exchangeTimezoneName": "UTC",
                        "dataGranularity": "1d",
                    },
                    "timestamp": [1704067200, 1704153600, 1704240000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [42000.0, None, 43000.0],
                                "high": [42500.0, None, 43500.0],
                                "low": [41500.0, None, 42500.0],
                                "close": [42300.0, None, 43200.0],
                                "volume": [1000, None, 1100],
                            }
                        ],
                        "adjclose": [{"adjclose": [42300.0, None, 43200.0]}],
                    },
                }
            ],
            "error": None,
        }
    }

    frame, metadata = normalize_yahoo_crypto_chart_payload("BTC-USD", payload)

    assert len(frame) == 2
    assert frame["date"].tolist() == [
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-01-03").date(),
    ]
    assert metadata["currency"] == "USD"
    assert metadata["instrumentType"] == "CRYPTOCURRENCY"


def test_build_yahoo_crypto_coverage_summary_uses_requested_calendar_start() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["BTC-USD", "BTC-USD", "ETH-USD", "ETH-USD"],
            "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-02", "2019-01-03"]),
        }
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["BTC-USD", "ETH-USD"],
            "currency": ["USD", "USD"],
            "exchangeName": ["CCC", "CCC"],
            "instrumentType": ["CRYPTOCURRENCY", "CRYPTOCURRENCY"],
            "firstTradeDate": [
                pd.Timestamp("2019-01-01").date(),
                pd.Timestamp("2019-01-02").date(),
            ],
            "exchangeTimezoneName": ["UTC", "UTC"],
            "dataGranularity": ["1d", "1d"],
        }
    )

    coverage = build_yahoo_crypto_coverage_summary(
        panel,
        metadata,
        requested_start_date="2019-01-01",
    )

    assert coverage.loc[coverage["ticker"] == "BTC-USD", "covers_requested_start"].iloc[0]
    assert not coverage.loc[coverage["ticker"] == "ETH-USD", "covers_requested_start"].iloc[0]
    assert (
        coverage.loc[coverage["ticker"] == "ETH-USD", "effective_start_date"].iloc[0]
        == pd.Timestamp("2019-01-01").date()
    )


def test_parse_french_daily_rfr_keeps_only_daily_rows_and_scales_to_decimal() -> None:
    csv_text = "\n".join(
        [
            "This file was created by CMPT_ME_BEME_RETS_DAILY using the 202605 CRSP database.",
            ",Mkt-RF,SMB,HML,RF",
            "19260701,    0.09,   -0.25,   -0.27,    0.01",
            "19260702,    0.45,   -0.33,   -0.06,    0.01",
            "",
            " Annual Factors: January-December ",
            "1927,   29.44,   -2.20,   -4.58,    3.12",
        ]
    )

    frame = parse_french_daily_rfr(csv_text)

    assert list(frame.columns) == ["date", "rfr"]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["1926-07-01", "1926-07-02"]
    assert frame["rfr"].tolist() == pytest.approx([0.0001, 0.0001])


def test_merge_daily_rfr_forward_fills_weekends_and_tail_dates(tmp_path: Path) -> None:
    rfr_path = tmp_path / "french_daily_rfr.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"]),
            "rfr": [0.00009, 0.0001, 0.0002],
        }
    ).to_parquet(rfr_path, index=False)

    frame = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * 6,
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09"]
            ),
            "ret": [0.00, 0.01, 0.02, -0.01, 0.03, 0.01],
        }
    )

    merged = merge_daily_rfr(frame, rfr_path=rfr_path)

    assert merged["rfr"].tolist() == pytest.approx(
        [0.00009, 0.0001, 0.0001, 0.0001, 0.0002, 0.0002]
    )


def test_stage2_wide_and_long_returns_match_on_daily_crypto_panel() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * 4 + ["ETH-USD"] * 4,
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                ]
            ),
            "adjClose": [100.0, 102.0, 101.0, 104.03, 200.0, 206.0, 203.94, 208.0188],
        }
    )
    wide_prices = build_adjusted_close_wide(panel, price_column="adjClose")
    wide_returns = compute_wide_returns(wide_prices)
    long_returns = compute_long_returns(panel, price_column="adjClose")

    max_abs_diff = assert_return_parity(long_returns, wide_returns)

    assert max_abs_diff <= 1e-12
    assert wide_returns["BTC-USD"].iloc[1] == pytest.approx(0.02)
    assert long_returns.loc[
        long_returns["ticker"] == "ETH-USD",
        "ret",
    ].iloc[1] == pytest.approx(0.03)

def test_build_feature_long_panel_adds_expected_columns_and_uses_daily_window(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2024-01-01", periods=190, freq="D")
    base_pattern = np.array([0.01, -0.02, 0.005, -0.003, 0.004, -0.006, 0.002])
    returns = np.resize(base_pattern, len(dates)).astype(float)
    returns[0] = np.nan
    returns[150] = 0.25
    returns[151] = -0.18

    long_returns = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * len(dates),
            "date": dates,
            "ret": returns,
        }
    )
    rfr_dates = pd.bdate_range(dates.min(), dates.max())
    rfr_path = tmp_path / "french_daily_rfr.parquet"
    pd.DataFrame({"date": rfr_dates, "rfr": np.full(len(rfr_dates), 0.0001)}).to_parquet(
        rfr_path,
        index=False,
    )

    featured = build_feature_long_panel(long_returns, rfr_path=rfr_path)

    expected_columns = {
        "abs_ret",
        "rfr",
        "excess_ret",
        "is_large_move_10pct",
        "is_large_move_20pct",
        "rolling_6m_avg_ret",
        "rolling_6m_vol",
        "rolling_6m_var_95",
        "rolling_6m_sharpe",
        "rolling_6m_sortino",
    }
    assert expected_columns.issubset(featured.columns)
    assert featured.loc[featured["date"] == dates[150], "is_large_move_10pct"].item()
    assert featured.loc[featured["date"] == dates[150], "is_large_move_20pct"].item()
    last_row = featured.iloc[-1]
    assert pd.notna(last_row["rolling_6m_avg_ret"])
    assert pd.notna(last_row["rolling_6m_vol"])
    assert pd.notna(last_row["rolling_6m_var_95"])
    assert pd.notna(last_row["rolling_6m_sharpe"])
    assert pd.notna(last_row["rolling_6m_sortino"])


def test_summarize_stage2_metrics_includes_sharpe_sortino_and_tail_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    rows: list[dict[str, object]] = []
    tickers = ["BTC-USD", "ETH-USD", "XRP-USD"]
    for ticker_index, ticker in enumerate(tickers):
        for date_index, date in enumerate(dates):
            ret = 0.002 * (ticker_index + 1) + ((date_index % 7) - 3) * 0.004
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "ret": ret,
                    "excess_ret": ret - 0.0001,
                }
            )
    summary = summarize_stage2_metrics(pd.DataFrame(rows))

    expected_columns = {
        "full_sample_sharpe",
        "full_sample_sortino",
        "skewness",
        "excess_kurtosis",
        "tail_share_abs_z_gt_2",
        "tail_share_abs_z_gt_3",
        "ann_return_pct",
        "ann_volatility_pct",
    }
    assert expected_columns.issubset(summary.columns)
    assert summary["full_sample_sharpe"].notna().all()
    assert summary["full_sample_sortino"].notna().all()


def test_derived_figure_inputs_have_expected_bounds() -> None:
    stage1_panel, feature_panel = _synthetic_figure_inputs()

    drawdown = _build_headline_drawdown_wide(feature_panel)
    relative = _build_relative_to_btc_wide(feature_panel)
    rolling_corr = _build_rolling_corr_to_btc_wide(feature_panel)
    dispersion = _build_cross_sectional_dispersion_frame(feature_panel)
    headline_volume = _build_headline_dollar_volume_wide(stage1_panel)
    volume_share = _build_daily_volume_share_wide(stage1_panel)
    dollar_volume = _build_trailing_dollar_volume_frame(stage1_panel)

    assert (drawdown.to_numpy() <= 1e-12).all()
    assert relative.iloc[0].to_numpy() == pytest.approx(np.ones(len(relative.columns)))
    clean_corr = rolling_corr.to_numpy()
    clean_corr = clean_corr[~np.isnan(clean_corr)]
    assert ((clean_corr >= -1.0) & (clean_corr <= 1.0)).all()
    assert (dispersion["cross_sectional_vol"] >= 0.0).all()
    assert (dispersion["interdecile_spread"] >= 0.0).all()
    assert (headline_volume.to_numpy() > 0.0).all()
    assert volume_share.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(volume_share)))
    assert (dollar_volume["trailing_median_dollar_volume_usd"] > 0.0).all()
    expected_btc_volume = float(
        stage1_panel.loc[stage1_panel["ticker"] == "BTC-USD"]
        .sort_values("date")
        .tail(30)["volume"]
        .median()
    )
    actual_btc_volume = float(
        dollar_volume.loc[dollar_volume["ticker"] == "BTC-USD", "trailing_median_dollar_volume_usd"]
        .iloc[0]
    )
    assert actual_btc_volume == pytest.approx(expected_btc_volume)


def test_make_stage2_figure_pack_exports_pngs(tmp_path: Path) -> None:
    stage1_panel, feature_panel = _synthetic_figure_inputs()

    outputs = make_stage2_figure_pack(
        stage1_panel,
        feature_panel,
        output_dir=tmp_path,
    )

    assert set(outputs) == {
        "headline_prices",
        "growth_of_one_dollar",
        "headline_drawdowns",
        "relative_to_btc",
        "rolling_corr_to_btc",
        "extreme_moves",
        "correlation_matrix",
        "volatility_ranking",
        "annualized_return_ranking",
        "sharpe_ranking",
        "sortino_ranking",
        "distribution_vs_normal",
        "tail_share_comparison",
    }
    for figure_paths in outputs.values():
        assert figure_paths["png"].exists()


def test_make_stage2_figure_pack_appendix_exports_pngs(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=220, freq="D")
    returns_rows: list[dict[str, object]] = []
    stage1_rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(HEADLINE_TICKERS):
        price = 120.0 + 25.0 * ticker_index
        for date_index, date in enumerate(dates):
            ret = 0.0008 * (ticker_index + 1) + ((date_index % 10) - 5) * 0.0025
            if date_index == 0:
                ret = np.nan
            if pd.notna(ret):
                price *= 1.0 + float(ret)
            returns_rows.append({"ticker": ticker, "date": date, "ret": ret})
            stage1_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adjClose": price,
                    "volume": 600_000 + 20_000 * ticker_index + 2_000 * date_index,
                }
            )
    returns_panel = pd.DataFrame(returns_rows)
    rfr_dates = pd.bdate_range(dates.min(), dates.max())
    rfr_path = tmp_path / "french_daily_rfr.parquet"
    pd.DataFrame({"date": rfr_dates, "rfr": np.full(len(rfr_dates), 0.0001)}).to_parquet(
        rfr_path,
        index=False,
    )
    feature_panel = build_feature_long_panel(returns_panel, rfr_path=rfr_path)
    stage1_panel = pd.DataFrame(stage1_rows)
    summary = summarize_stage2_metrics(feature_panel)

    outputs = make_stage2_figure_pack(
        stage1_panel,
        feature_panel,
        output_dir=tmp_path,
        summary=summary,
        include_appendix=True,
    )

    appendix_keys = {
        "cross_sectional_dispersion",
        "rolling_volatility_headline",
        "rolling_sharpe_headline",
        "max_drawdown_ranking",
        "risk_return_scatter",
        "headline_dollar_volume",
        "dollar_volume_concentration",
        "dollar_volume_ranking",
    }
    assert appendix_keys.issubset(outputs)
    for figure_paths in outputs.values():
        assert figure_paths["png"].exists()
