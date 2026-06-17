# ruff: noqa
"""Offline tests for the Week 4 API helpers."""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fins2026.week4.code.api_intro_ecb import build_ecb_url, tidy_ecb_exchange_rates
from fins2026.week4.code.equity_api_tiingo import (
    build_wide_price_table,
    load_tickers_from_file,
    parse_tiingo_prices,
    resolve_api_key,
)
from fins2026.week4.code.equity_api_yahoo import (
    build_yahoo_coverage_summary,
    load_yahoo_tickers_from_file,
    normalize_yahoo_chart_payload,
    to_epoch_seconds,
)
from fins2026.week4.code.risk_free_rate_french import (
    extract_first_csv_text,
    parse_french_daily_rfr,
)
from fins2026.week4.code.stage2_equity_returns import (
    assert_return_parity,
    build_adjusted_close_wide,
    build_feature_long_panel,
    compute_long_returns,
    compute_wide_returns,
    merge_daily_rfr,
)
from fins2026.week4.code.stage2_return_figures import make_stage2_figure_pack
from fins2026.week4.code.stage3_portfolio_figures import make_stage3_figure_pack
from fins2026.week4.code.stage3_portfolios import (
    PORTFOLIO_LABELS,
    build_balanced_stage3_sample,
    build_efficient_frontier,
    compute_portfolio_returns,
    estimate_portfolio_weights,
    minimum_variance_weights,
    solve_markowitz_system,
    summarize_asset_statistics,
    summarize_portfolio_metrics,
)


def test_build_ecb_url_matches_teaching_shape() -> None:
    url = build_ecb_url(
        base_currency="USD",
        quote_currency="EUR",
        frequency="D",
        start_date="2026-05-01",
        end_date="2026-05-15",
    )
    assert "https://data-api.ecb.europa.eu/service/data/EXR/" in url
    assert "D.USD.EUR.SP00.A" in url
    assert "startPeriod=2026-05-01" in url
    assert "endPeriod=2026-05-15" in url
    assert "format=csvdata" in url


def test_tidy_ecb_exchange_rates_keeps_teaching_columns() -> None:
    csv_text = "\n".join(
        [
            "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,TITLE",
            "D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-05-01,1.13,USD/EUR exchange rate",
            "D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-05-02,1.14,USD/EUR exchange rate",
        ]
    )
    tidy = tidy_ecb_exchange_rates(csv_text)
    assert list(tidy.columns) == [
        "date",
        "exchange_rate",
        "base_currency",
        "quote_currency",
        "series_title",
    ]
    assert tidy["exchange_rate"].tolist() == [1.13, 1.14]


def test_load_tickers_from_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("# comment\nAAPL\n\nmsft\n", encoding="utf-8")
    assert load_tickers_from_file(ticker_file) == ("AAPL", "MSFT")


def test_parse_tiingo_prices_makes_clean_long_panel() -> None:
    payload = [
        {"date": "2024-01-02T00:00:00.000Z", "close": 100.0, "open": 99.0},
        {"date": "2024-01-03T00:00:00.000Z", "close": 101.0, "open": 100.0},
    ]
    frame = parse_tiingo_prices(payload, "AAPL")
    assert list(frame.columns[:2]) == ["ticker", "date"]
    assert frame["ticker"].tolist() == ["AAPL", "AAPL"]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_build_wide_price_table_pivots_dates_and_tickers() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "close": [100.0, 200.0, 101.0, 201.0],
        }
    )
    wide = build_wide_price_table(panel, value_column="close")
    assert list(wide.columns) == ["date", "AAPL", "MSFT"]
    assert wide.loc[0, "AAPL"] == 100.0
    assert wide.loc[1, "MSFT"] == 201.0


def test_resolve_api_key_requires_explicit_or_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Missing Tiingo API key"):
        resolve_api_key()


def test_load_yahoo_tickers_from_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    ticker_file = tmp_path / "yahoo_tickers.txt"
    ticker_file.write_text("# comment\nAAPL\n\nmsft\n", encoding="utf-8")
    assert load_yahoo_tickers_from_file(ticker_file) == ("AAPL", "MSFT")


def test_to_epoch_seconds_matches_known_midnight_utc_value() -> None:
    assert to_epoch_seconds("2000-01-01") == 946684800


def test_normalize_yahoo_chart_payload_builds_expected_long_schema() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": "AAPL",
                        "exchangeName": "NMS",
                        "instrumentType": "EQUITY",
                        "firstTradeDate": 345479400,
                        "exchangeTimezoneName": "America/New_York",
                        "dataGranularity": "1d",
                    },
                    "timestamp": [1704171600, 1704258000, 1704344400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0, None],
                                "high": [101.0, 102.0, None],
                                "low": [99.0, 100.0, None],
                                "close": [100.5, 101.5, None],
                                "volume": [1_000, 1_100, None],
                            }
                        ],
                        "adjclose": [{"adjclose": [99.5, 100.5, None]}],
                    },
                    "events": {
                        "dividends": {
                            "evt0": {"amount": 0.25, "date": 1704171600},
                        },
                        "splits": {
                            "evt1": {
                                "date": 1704258000,
                                "numerator": 2,
                                "denominator": 1,
                                "splitRatio": "2/1",
                            }
                        },
                    },
                }
            ],
            "error": None,
        }
    }

    frame, metadata = normalize_yahoo_chart_payload("AAPL", payload)

    assert list(frame.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
        "dividend",
        "splitFactor",
    ]
    assert frame["ticker"].tolist() == ["AAPL", "AAPL"]
    assert len(frame) == 2
    assert frame["dividend"].iloc[0] == pytest.approx(0.25)
    assert frame["splitFactor"].iloc[1] == pytest.approx(2.0)
    assert metadata["currency"] == "USD"
    assert metadata["exchangeName"] == "NMS"


def test_build_yahoo_coverage_summary_flags_requested_start() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date": pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-04", "2000-01-05"]),
        }
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "currency": ["USD", "USD"],
            "exchangeName": ["NMS", "NMS"],
            "instrumentType": ["EQUITY", "EQUITY"],
            "firstTradeDate": [dt.date(1980, 12, 12), dt.date(1986, 3, 13)],
            "exchangeTimezoneName": ["America/New_York", "America/New_York"],
            "dataGranularity": ["1d", "1d"],
        }
    )

    coverage = build_yahoo_coverage_summary(
        panel,
        metadata,
        requested_start_date="2000-01-03",
    )

    assert coverage.loc[coverage["ticker"] == "AAPL", "covers_requested_start"].iloc[0]
    assert not coverage.loc[coverage["ticker"] == "MSFT", "covers_requested_start"].iloc[0]


def test_extract_first_csv_text_reads_zip_member() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("sample.csv", "Date,Mkt-RF,SMB,HML,RF\n19260701,0.09,0.0,0.0,0.01\n")

    text = extract_first_csv_text(buffer.getvalue())
    assert "Date,Mkt-RF,SMB,HML,RF" in text
    assert "19260701" in text


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


def test_stage2_wide_and_long_returns_match_on_toy_panel() -> None:
    panel = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "adjClose": [100.0, 101.0, 99.99, 200.0, 202.0, 204.02],
        }
    )
    wide_prices = build_adjusted_close_wide(panel, price_column="adjClose")
    wide_returns = compute_wide_returns(wide_prices)
    long_returns = compute_long_returns(panel, price_column="adjClose")

    max_abs_diff = assert_return_parity(long_returns, wide_returns)

    assert max_abs_diff <= 1e-12
    assert wide_returns["AAPL"].iloc[1] == pytest.approx(0.01)
    assert long_returns.loc[long_returns["ticker"] == "MSFT", "ret"].iloc[1] == pytest.approx(0.01)


def test_merge_daily_rfr_forward_fills_tail_dates(tmp_path: Path) -> None:
    rfr_path = tmp_path / "french_daily_rfr.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "rfr": [0.0001, 0.0002],
        }
    ).to_parquet(rfr_path, index=False)

    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ret": [np.nan, 0.01, -0.02],
        }
    )

    merged = merge_daily_rfr(frame, rfr_path=rfr_path)

    assert merged["rfr"].tolist() == pytest.approx([0.0001, 0.0002, 0.0002], nan_ok=True)


def test_build_feature_long_panel_adds_expected_columns(tmp_path: Path) -> None:
    dates = pd.bdate_range("2023-01-02", periods=130)
    base_pattern = np.array([0.01, -0.02, 0.005, -0.003, 0.004, -0.006])
    returns = np.resize(base_pattern, len(dates)).astype(float)
    returns[0] = np.nan
    returns[110] = 0.25
    returns[111] = -0.15

    long_returns = pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(dates),
            "date": dates,
            "ret": returns,
        }
    )
    rfr_path = tmp_path / "french_daily_rfr.parquet"
    pd.DataFrame({"date": dates, "rfr": np.full(len(dates), 0.0001)}).to_parquet(
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
    assert featured.loc[featured["date"] == dates[110], "is_large_move_10pct"].item()
    assert featured.loc[featured["date"] == dates[110], "is_large_move_20pct"].item()
    last_row = featured.iloc[-1]
    assert pd.notna(last_row["rolling_6m_avg_ret"])
    assert pd.notna(last_row["rolling_6m_vol"])
    assert pd.notna(last_row["rolling_6m_var_95"])
    assert pd.notna(last_row["rolling_6m_sharpe"])
    assert pd.notna(last_row["rolling_6m_sortino"])


def test_make_stage2_figure_pack_exports_pngs(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=40)
    rows: list[dict[str, object]] = []
    tickers = [f"T{i:02d}" for i in range(10)]
    for ticker_index, ticker in enumerate(tickers):
        for date_index, date in enumerate(dates):
            ret = ((ticker_index + 1) * 0.001) + ((date_index % 5) - 2) * 0.002
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "ret": ret,
                    "is_large_move_10pct": abs(ret) >= 0.10,
                }
            )
    feature_panel = pd.DataFrame(rows)

    outputs = make_stage2_figure_pack(
        feature_panel,
        provider="yahoo",
        output_dir=tmp_path,
    )

    assert set(outputs) == {
        "annualized_return_ranking",
        "distribution",
        "extreme_moves",
        "sharpe_ranking",
        "volatility_ranking",
        "top_bottom_growth",
        "extreme_move_count",
    }
    for figure_paths in outputs.values():
        assert figure_paths["png"].exists()


def _synthetic_stage3_feature_panel() -> pd.DataFrame:
    """Create a compact deterministic Stage 3 feature panel."""

    dates = pd.bdate_range("2020-01-02", periods=260)
    tickers = ["AAPL", "MSFT", "JPM", "KO"]
    chol = np.array(
        [
            [0.012, 0.0, 0.0, 0.0],
            [0.004, 0.010, 0.0, 0.0],
            [0.003, 0.002, 0.011, 0.0],
            [0.001, 0.001, 0.002, 0.008],
        ]
    )
    draws = np.random.default_rng(42).standard_normal((len(dates), len(tickers)))
    innovations = draws @ chol.T
    means = np.array([0.0012, 0.0010, 0.0008, 0.0006])
    returns = innovations + means
    rows: list[dict[str, object]] = []
    rfr = np.full(len(dates), 0.0001)
    for ticker_index, ticker in enumerate(tickers):
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "ret": returns[date_index, ticker_index],
                    "rfr": rfr[date_index],
                }
            )
    return pd.DataFrame(rows)


def test_build_balanced_stage3_sample_drops_missing_dates() -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    feature_panel.loc[
        (feature_panel["ticker"] == "AAPL")
        & (feature_panel["date"] == feature_panel["date"].iloc[10]),
        "ret",
    ] = np.nan

    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo",
        display_name="Yahoo Finance",
    )

    assert sample.returns_wide.index.is_monotonic_increasing
    assert not sample.returns_wide.isna().any().any()
    assert not sample.rfr.isna().any()
    assert feature_panel["date"].nunique() - sample.sample_days == 1


def test_stage3_portfolio_weights_sum_to_one_and_equal_weight_is_flat() -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo",
        display_name="Yahoo Finance",
    )

    weights, _methods = estimate_portfolio_weights(sample)
    matrix = weights.pivot(index="ticker", columns="portfolio", values="weight")

    assert matrix[PORTFOLIO_LABELS["equal_weight"]].tolist() == pytest.approx([0.25] * 4)
    assert matrix.sum(axis=0).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_minimum_variance_is_no_riskier_than_equal_weight_on_sample_covariance() -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo",
        display_name="Yahoo Finance",
    )

    weights, _methods = estimate_portfolio_weights(sample)
    matrix = weights.pivot(index="ticker", columns="portfolio", values="weight")
    covariance = np.cov(sample.returns_wide.to_numpy(dtype=float), rowvar=False, ddof=1)

    w_eq = matrix[PORTFOLIO_LABELS["equal_weight"]].to_numpy(dtype=float)
    w_mv = matrix[PORTFOLIO_LABELS["minimum_variance"]].to_numpy(dtype=float)
    var_eq = float(w_eq @ covariance @ w_eq)
    var_mv = float(w_mv @ covariance @ w_mv)

    assert var_mv <= var_eq + 1e-12


def test_tangency_sharpe_is_at_least_as_high_as_equal_weight_and_minimum_variance() -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo",
        display_name="Yahoo Finance",
    )

    weights, _methods = estimate_portfolio_weights(sample)
    portfolio_returns = compute_portfolio_returns(sample, weights)
    metrics = summarize_portfolio_metrics(portfolio_returns).set_index("portfolio")

    sharpe_eq = float(metrics.loc[PORTFOLIO_LABELS["equal_weight"], "sharpe_ratio"])
    sharpe_mv = float(metrics.loc[PORTFOLIO_LABELS["minimum_variance"], "sharpe_ratio"])
    sharpe_tan = float(metrics.loc[PORTFOLIO_LABELS["mean_variance_tangency"], "sharpe_ratio"])

    assert sharpe_tan >= sharpe_eq - 1e-10
    assert sharpe_tan >= sharpe_mv - 1e-10


def test_efficient_frontier_has_non_decreasing_volatility() -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo",
        display_name="Yahoo Finance",
    )
    weights, _methods = estimate_portfolio_weights(sample)
    frontier = build_efficient_frontier(sample, weights, n_points=80)

    diffs = np.diff(frontier["volatility_ann"].to_numpy(dtype=float))
    assert (diffs >= -1e-10).all()


def test_solve_markowitz_system_and_minimum_variance_handle_singular_covariance() -> None:
    covariance = np.array([[0.04, 0.04], [0.04, 0.04]], dtype=float)
    solution, method = solve_markowitz_system(covariance, np.array([1.0, 1.0]))
    weights, weight_method = minimum_variance_weights(covariance)

    assert method == "pseudoinverse"
    assert weight_method == "pseudoinverse"
    assert solution.tolist() == pytest.approx([12.5, 12.5])
    assert weights.tolist() == pytest.approx([0.5, 0.5])


def test_make_stage3_figure_pack_exports_pngs(tmp_path: Path) -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo",
        display_name="Yahoo Finance",
    )
    weights, _methods = estimate_portfolio_weights(sample)
    portfolio_returns = compute_portfolio_returns(sample, weights)
    metrics = summarize_portfolio_metrics(portfolio_returns)
    frontier = build_efficient_frontier(sample, weights, n_points=80)
    asset_summary = summarize_asset_statistics(sample)

    outputs = make_stage3_figure_pack(
        provider="yahoo",
        sample=sample,
        weights=weights,
        portfolio_returns=portfolio_returns,
        frontier=frontier,
        metrics=metrics,
        asset_summary=asset_summary,
        output_dir=tmp_path,
    )

    assert set(outputs) == {
        "weights",
        "growth_of_one",
        "drawdowns",
        "scorecard",
        "efficient_frontier",
    }
    for figure_paths in outputs.values():
        assert figure_paths["png"].exists()
