# ruff: noqa
"""Streamlit view rendering for the Week 3 companion U.S. macro app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fins2026.week3.us_app.app_config import (
    ALL_SPECS,
    APP_TITLE,
    BACKTEST_LABELS,
    DATA_LABELS,
    DATA_MODE_OPTIONS,
    DEFAULT_SAMPLE_PERIOD,
    FORECAST_LABELS,
    FORECAST_OUTPUT_LABELS,
    GDP_SPECS,
    MODEL_LABELS,
    SAMPLE_PERIODS,
    VIEW_OPTIONS,
)
from fins2026.week3.us_app.app_data import (
    apply_sample_period,
    load_market_data,
    source_status_text,
)
from fins2026.week3.us_app.app_insights import (
    build_stress_score,
    compact_table_height,
    current_stress_average,
    current_stress_percentile,
    forecast_and_backtest,
    forecast_level_change_percent,
    format_observation_date,
    format_percent,
    format_percentage_point,
    format_percentile_label,
    format_stress_value,
    gdp_growth_band,
    gdp_outlook_interpretation,
    latest_annualized_quarterly_growth,
    latest_snapshot,
    latest_year_over_year_growth,
    prior_stress_average,
    series_window_label,
    stress_overview_text,
    stress_percentile_help,
    stress_score_figure,
    stress_window_label,
    target_short_label,
    top_state_metrics,
    yield_curve_figure,
    yield_curve_frame,
    yield_curve_insights,
    yield_curve_interpretation,
)
from fintools.apps import (
    MetricCard,
    active_tab_label,
    availability_dates,
    backtest_figure,
    configure_page,
    forecast_series_spec,
    lazy_tabs,
    prepare_display_frame,
    query_choice,
    query_int,
    render_compact_metric_strip,
    render_csv_download,
    render_data_health,
    render_display_table,
    sync_query_params,
    tab_is_open,
    target_forecast_figure,
    target_name,
)


def render_equation(label: str, latex: str, explanation: str | None = None) -> None:
    """Render a labeled equation with Streamlit's native LaTeX support."""

    st.markdown(f"**{label}**")
    st.latex(latex)
    if explanation:
        st.caption(explanation)


def render_forecast_controls(
    *,
    prefix: str,
    forecast_default: str,
    model_default: str,
    horizon_default: int,
    gdp_horizon_default: int,
) -> tuple[str, str, int]:
    """Render local forecast controls and return selected label, model, horizon."""

    cols = st.columns([1.25, 1.0, 1.0])
    with cols[0]:
        forecast_label = st.selectbox(
            "Forecast series",
            list(FORECAST_LABELS),
            index=list(FORECAST_LABELS).index(forecast_default),
            key=f"{prefix}_forecast_label",
        )
    with cols[1]:
        model = st.selectbox(
            "Forecast model",
            list(MODEL_LABELS),
            index=list(MODEL_LABELS).index(model_default),
            format_func=lambda item: MODEL_LABELS[item],
            key=f"{prefix}_forecast_model",
        )
    selected_spec = FORECAST_LABELS[forecast_label]
    with cols[2]:
        if selected_spec.role == "macro":
            horizon = st.slider(
                "Forecast horizon (quarters)",
                1,
                8,
                gdp_horizon_default,
                step=1,
                key=f"{prefix}_forecast_horizon_quarters",
                help="Quarterly GDP horizons are counted in calendar quarters.",
            )
        else:
            horizon = st.slider(
                "Forecast horizon (market days)",
                21,
                252,
                horizon_default,
                step=21,
                key=f"{prefix}_forecast_horizon_market_days",
                help="Market horizons are counted in business days.",
            )
    return forecast_label, model, horizon


def main() -> None:
    """Render the U.S. macro app."""

    configure_page(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(
        "Monitor Treasury yields, credit spreads, equity volatility, and real GDP "
        "with public FRED data."
    )

    data_mode_default = query_choice("data", DATA_MODE_OPTIONS, default="Fixture")
    forecast_default = query_choice("forecast", list(FORECAST_LABELS), default="10-Year Treasury")
    model_default = query_choice("model", list(MODEL_LABELS), default="drift")
    horizon_default = query_int("horizon", default=126, minimum=21, maximum=252, step=21)
    gdp_horizon_default = query_int("gdp_horizon", default=4, minimum=1, maximum=8, step=1)
    sample_default = query_choice(
        "sample",
        list(SAMPLE_PERIODS),
        default=DEFAULT_SAMPLE_PERIOD,
    )
    view_default = query_choice("view", VIEW_OPTIONS, default="Overview")

    with st.sidebar:
        st.header("Controls")
        data_mode = st.radio(
            "Data source",
            DATA_MODE_OPTIONS,
            index=DATA_MODE_OPTIONS.index(data_mode_default),
            horizontal=True,
            captions=[
                "Stable built-in sample for repeatable analysis.",
                "Latest available observations from FRED; internet required.",
            ],
            key="data_mode",
        )

    data, active_data_mode, warning, loaded_at_utc = load_market_data(data_mode)
    if warning:
        st.warning("Live FRED could not be loaded, so the fixture data is shown.")
        with st.expander("Technical detail"):
            st.write(warning)

    sample_period = st.segmented_control(
        "Sample period",
        options=list(SAMPLE_PERIODS),
        default=sample_default,
        help="The selected period controls the chart, forecast, backtest, metrics, and data table.",
        key="sample_period",
    )
    sample_period = sample_period or DEFAULT_SAMPLE_PERIOD
    sample_data = apply_sample_period(data, sample_period)

    render_compact_metric_strip(
        top_state_metrics(
            sample_data,
            active_data_mode=active_data_mode,
            sample_period=sample_period,
        ),
        columns=3,
    )

    render_data_health(
        sample_data,
        source=active_data_mode,
        value_columns=list(ALL_SPECS),
    )
    st.caption(
        source_status_text(
            data,
            active_data_mode=active_data_mode,
            loaded_at_utc=loaded_at_utc,
            warning=warning,
        )
    )

    current_forecast_label = forecast_default
    current_model = model_default
    current_horizon = horizon_default
    current_gdp_horizon = gdp_horizon_default

    tabs = lazy_tabs(VIEW_OPTIONS, default=view_default, key="main_view")
    active_view = active_tab_label(VIEW_OPTIONS, tabs, default=view_default)
    (
        tab_overview,
        tab_stress,
        tab_curve,
        tab_forecast,
        tab_backtest,
        tab_gdp,
        tab_data,
        tab_method,
    ) = tabs

    if tab_is_open(tab_overview, fallback=active_view == "Overview"):
        with tab_overview:
            overview_score = build_stress_score(sample_data)
            st.subheader("Overview")
            overview_left, overview_right = st.columns([0.85, 1.35])
            with overview_left:
                st.markdown(
                    f"""
                    Treasury yields, credit spreads, volatility, and real GDP
                    over the **{sample_period}** sample.

                    **Current stress vs {sample_period}:** average z-score over
                    the latest 21 trading days, based on VIX (%), high-yield
                    OAS, and the inverted 10Y-2Y Treasury spread. The score is
                    standardized against the selected sample's mean and
                    standard deviation, so it changes when the sample period
                    changes; 0 is the selected-sample average.
                    """
                )
                render_equation(
                    "Stress formula",
                    r"s_t = \frac{1}{3}\left(z_{VIX,t} + z_{HY,t} - z_{10Y-2Y,t}\right)",
                    "Each z-score is computed within the selected sample.",
                )
                st.markdown(
                    f"""
                    {stress_overview_text(overview_score, sample_period)}

                    Forecast views focus on rates, credit spreads, and real GDP.
                    VIX is monitored as volatility context, not as a baseline
                    forecast target.
                    """
                )
            with overview_right:
                st.subheader("Latest market readings")
                snapshot = latest_snapshot(sample_data)
                if snapshot.empty:
                    st.info("No latest-state table is available for the selected sample.")
                else:
                    render_display_table(
                        snapshot,
                        labels={
                            "indicator": "Indicator",
                            "latest_date": "Latest date",
                            "latest_value": "Latest value",
                            "units": "Units",
                            "forecast_treatment": "Forecast treatment",
                            "percentile": "Sample standing (%)",
                        },
                        reset_index=False,
                        height=compact_table_height(snapshot),
                        column_alignments="center",
                    )

    if tab_is_open(tab_stress, fallback=active_view == "Stress Score"):
        with tab_stress:
            st.subheader("Stress Score")
            st.caption(
                "Composite of VIX (%), high-yield OAS, and the inverted 10Y-2Y "
                "Treasury spread. Higher values indicate more stress than usual "
                "for the selected sample; 0 is the selected-sample average. "
                "Changing the sample period changes the mean and standard "
                "deviation used for z-scores; it is not a data-availability change."
            )
            score = build_stress_score(sample_data)
            if score.empty:
                st.info("Not enough stress data for the selected sample.")
            else:
                current_stress = current_stress_average(score)
                prior_stress = prior_stress_average(score)
                current_change = (
                    None
                    if current_stress is None or prior_stress is None
                    else current_stress - prior_stress
                )
                current_percentile = current_stress_percentile(score)
                render_compact_metric_strip(
                    [
                        MetricCard(
                            f"Current stress vs {sample_period} (z)",
                            format_stress_value(current_stress),
                            help=(
                                f"Average over {stress_window_label(score)}; "
                                "0 is the selected-sample average. This z-score "
                                "is recalculated when the sample period changes."
                            ),
                        ),
                        MetricCard(
                            "Latest daily (z)",
                            f"{score.iloc[-1]:,.2f}",
                            help=(
                                "Most recent daily stress reading in z-score "
                                f"units, as of {format_observation_date(score.index[-1])}."
                            ),
                        ),
                        MetricCard(
                            "1M change (z)",
                            (
                                f"{current_change:+,.2f}"
                                if current_change is not None
                                else "n/a"
                            ),
                            help=(
                                "Latest 21-trading-day average minus the prior "
                                "21-trading-day average."
                            ),
                        ),
                        MetricCard(
                            "Current standing",
                            format_percentile_label(current_percentile),
                            help=stress_percentile_help(
                                current_percentile,
                                sample_period,
                            ),
                        ),
                    ],
                    columns=4,
                )
                st.caption(
                    f"Stress sample: {series_window_label(score)}. Current "
                    f"21-trading-day window: {stress_window_label(score)}. "
                    "Current standing compares the current 21-day average with "
                    "historical 21-day averages in this same sample."
                )
                st.caption(
                    "Daily readings are shown lightly; the red line is the "
                    "latest 21-trading-day average used for current stress."
                )
                st.plotly_chart(
                    stress_score_figure(score),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )

    if tab_is_open(tab_curve, fallback=active_view == "Yield Curve"):
        with tab_curve:
            st.subheader("Yield Curve")
            st.caption(
                "A point-in-time read on the Treasury term structure. The curve "
                "helps identify whether markets are pricing long-run yields above "
                "short-run policy-sensitive rates, or whether the curve is flat "
                "or inverted."
            )
            curve = yield_curve_frame(sample_data)
            curve_insights = yield_curve_insights(curve)
            render_compact_metric_strip(
                [
                    MetricCard(
                        "Curve shape",
                        curve_insights["shape"],
                        help="Based on the latest 10Y-2Y and 10Y-3M Treasury spreads.",
                    ),
                    MetricCard(
                        "10Y-2Y slope",
                        format_percentage_point(curve_insights["spread_10y_2y"]),
                        help="10-year yield minus 2-year yield.",
                    ),
                    MetricCard(
                        "10Y-3M slope",
                        format_percentage_point(curve_insights["spread_10y_3m"]),
                        help="10-year yield minus 3-month Treasury bill yield.",
                    ),
                    MetricCard(
                        "1Y curve shift",
                        curve_insights["shift"],
                        help=(
                            "Based on the one-year change in the 10Y-3M slope. "
                            f"3M moved {format_percentage_point(curve_insights['change_3m'])}; "
                            f"10Y moved {format_percentage_point(curve_insights['change_10y'])}."
                        ),
                    ),
                ],
                columns=4,
            )
            st.markdown(yield_curve_interpretation(curve))
            st.plotly_chart(
                yield_curve_figure(curve),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
            render_display_table(
                curve,
                labels={
                    "maturity": "Maturity",
                    "years": "Years",
                    "latest": "Latest",
                    "one_year_ago": "One year ago",
                },
                reset_index=False,
                column_alignments="center",
            )

    if tab_is_open(tab_forecast, fallback=active_view == "Forecasts"):
        with tab_forecast:
            st.subheader("Forecasts")
            st.caption(
                "Choose a forecastable series, model, and horizon. The chart shows "
                "the implied level path after the target transform is forecast."
            )
            forecast_label, model, horizon = render_forecast_controls(
                prefix="forecast_view",
                forecast_default=forecast_default,
                model_default=model_default,
                horizon_default=horizon_default,
                gdp_horizon_default=gdp_horizon_default,
            )
            current_forecast_label = forecast_label
            current_model = model
            spec = FORECAST_LABELS[forecast_label]
            if spec.role == "macro":
                current_gdp_horizon = horizon
            else:
                current_horizon = horizon
            series = sample_data[spec.series_id].dropna()
            if series.empty:
                st.info(f"No usable observations for {spec.label}.")
            else:
                forecast_result, backtest = forecast_and_backtest(
                    series,
                    forecast_label,
                    model=model,
                    horizon=horizon,
                )
                mae = (
                    backtest["absolute_level_error"].mean()
                    if not backtest.empty
                    else float("nan")
                )
                render_compact_metric_strip(
                    [
                        MetricCard(
                            "Latest",
                            f"{float(series.iloc[-1]):,.2f}",
                            help=f"Latest observation: {series.index[-1]:%Y-%m-%d}",
                        ),
                        MetricCard(
                            "Target",
                            target_short_label(forecast_label),
                            help=target_name(spec),
                        ),
                        MetricCard(
                            "Backtest MAE",
                            f"{mae:,.2f}" if pd.notna(mae) else "n/a",
                            help="Mean absolute one-step-ahead level error.",
                        ),
                    ],
                    columns=3,
                )
                st.plotly_chart(
                    target_forecast_figure(
                        forecast_result,
                        indicator_name=spec.label,
                        shade_recessions=True,
                        range_slider=False,
                    ),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )
                st.caption(
                    f"Forecast target: {target_name(spec)}. The chart displays the "
                    f"implied {spec.label.lower()} level path from that target."
                )
                download_cols = st.columns(2)
                with download_cols[0]:
                    render_csv_download(
                        prepare_display_frame(
                            forecast_result.display_forecast,
                            labels=FORECAST_OUTPUT_LABELS,
                        ),
                        label="Download displayed forecast CSV",
                        file_name=f"{spec.series_id.lower()}_display_forecast.csv",
                        key="download_display_forecast",
                    )
                with download_cols[1]:
                    render_csv_download(
                        prepare_display_frame(
                            forecast_result.target_forecast,
                            labels=FORECAST_OUTPUT_LABELS,
                        ),
                        label="Download target forecast CSV",
                        file_name=f"{spec.series_id.lower()}_target_forecast.csv",
                        key="download_target_forecast",
                    )

    if tab_is_open(tab_backtest, fallback=active_view == "Backtests"):
        with tab_backtest:
            st.subheader("Backtests")
            st.caption(
                "Backtests use one-step-ahead forecasts for the selected target and "
                "report errors on the displayed level path when available."
            )
            forecast_label, model, horizon = render_forecast_controls(
                prefix="backtest_view",
                forecast_default=forecast_default,
                model_default=model_default,
                horizon_default=horizon_default,
                gdp_horizon_default=gdp_horizon_default,
            )
            current_forecast_label = forecast_label
            current_model = model
            spec = FORECAST_LABELS[forecast_label]
            if spec.role == "macro":
                current_gdp_horizon = horizon
            else:
                current_horizon = horizon
            series = sample_data[spec.series_id].dropna()
            if series.empty:
                st.info(f"No usable observations for {spec.label}.")
                backtest = pd.DataFrame()
            else:
                _, backtest = forecast_and_backtest(
                    series,
                    forecast_label,
                    model=model,
                    horizon=horizon,
                )
            if backtest.empty:
                st.info("Not enough history for the configured backtest.")
            else:
                plot_backtest = pd.DataFrame(
                    {
                        "actual": backtest["actual_level"],
                        "forecast": backtest["forecast_level"],
                    },
                    index=backtest.index,
                )
                st.plotly_chart(
                    backtest_figure(plot_backtest, units=spec.units, shade_recessions=True),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )
                render_display_table(
                    backtest.tail(30),
                    labels=BACKTEST_LABELS,
                    height=420,
                    column_alignments="center",
                )
                render_csv_download(
                    prepare_display_frame(backtest, labels=BACKTEST_LABELS),
                    label="Download backtest CSV",
                    file_name=f"{spec.series_id.lower()}_backtest.csv",
                    key="download_backtest",
                )

    if tab_is_open(tab_gdp, fallback=active_view == "GDP Outlook"):
        with tab_gdp:
            st.subheader("GDP Outlook")
            st.caption(
                "GDP is quarterly, released with a lag, and revised. This view "
                "forecasts annualized quarterly growth, then converts that "
                "growth path back into an implied real-GDP level path."
            )
            gdp_spec = GDP_SPECS["GDPC1"]
            gdp_series = sample_data["GDPC1"].dropna()
            if gdp_series.empty:
                st.info("No quarterly GDP observations are available for the selected sample.")
            else:
                gdp_controls = st.columns([1.0, 1.0])
                with gdp_controls[0]:
                    gdp_model = st.selectbox(
                        "Forecast model",
                        list(MODEL_LABELS),
                        index=list(MODEL_LABELS).index(model_default),
                        format_func=lambda item: MODEL_LABELS[item],
                        key="gdp_outlook_model",
                    )
                with gdp_controls[1]:
                    gdp_horizon = st.slider(
                        "Forecast horizon (quarters)",
                        1,
                        8,
                        gdp_horizon_default,
                        step=1,
                        key="gdp_outlook_horizon",
                        help="Quarterly GDP horizons are counted in calendar quarters.",
                    )
                current_forecast_label = "Real GDP"
                current_model = gdp_model
                current_gdp_horizon = gdp_horizon
                gdp_result = forecast_series_spec(
                    gdp_series,
                    gdp_spec,
                    model=gdp_model,
                    horizon=gdp_horizon,
                )
                latest_growth = latest_annualized_quarterly_growth(gdp_series)
                yoy_growth = latest_year_over_year_growth(gdp_series)
                forecast_change = forecast_level_change_percent(
                    gdp_result,
                    float(gdp_series.iloc[-1]),
                )
                render_compact_metric_strip(
                    [
                        MetricCard(
                            "Latest real GDP",
                            f"{float(gdp_series.iloc[-1]):,.0f}",
                            help=f"Observation date: {gdp_series.index[-1]:%Y-%m-%d}",
                        ),
                        MetricCard(
                            "Latest growth",
                            format_percent(latest_growth),
                            help=(
                                "Annualized quarter-over-quarter real GDP growth; "
                                f"{gdp_growth_band(latest_growth).lower()} band."
                            ),
                        ),
                        MetricCard(
                            "YoY growth",
                            format_percent(yoy_growth),
                            help="Latest real GDP versus four quarters earlier.",
                        ),
                        MetricCard(
                            f"{gdp_horizon}Q forecast change",
                            format_percent(forecast_change, signed=True),
                            help=(
                                "Implied real-GDP level change over the selected "
                                f"{gdp_horizon}-quarter horizon."
                            ),
                        ),
                    ],
                    columns=4,
                )
                st.markdown(
                    gdp_outlook_interpretation(
                        gdp_series,
                        gdp_result,
                        model_label=MODEL_LABELS[gdp_model],
                        horizon=gdp_horizon,
                    )
                )
                st.plotly_chart(
                    target_forecast_figure(gdp_result, indicator_name="Real GDP"),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )
                availability = availability_dates(gdp_series.tail(1).index, gdp_spec)[0]
                st.caption(
                    "GDP is forecast as annualized quarterly growth and then converted "
                    "back into an implied real-GDP level path. The latest observation "
                    f"is treated as available around {availability:%Y-%m-%d}."
                )

    if tab_is_open(tab_data, fallback=active_view == "Data"):
        with tab_data:
            st.subheader("Data")
            st.caption("Selected sample used by the metrics, charts, forecasts, and downloads.")
            render_display_table(
                sample_data.tail(500),
                labels=DATA_LABELS,
                height=520,
                column_alignments="center",
            )
            render_csv_download(
                prepare_display_frame(sample_data, labels=DATA_LABELS),
                label="Download selected sample CSV",
                file_name="us_macro_selected_sample.csv",
                key="download_sample",
            )

    if tab_is_open(tab_method, fallback=active_view == "Methodology"):
        with tab_method:
            st.subheader("Methodology")
            st.markdown(
                f"""
                Public FRED market data feed a sample-relative stress score and
                transparent baseline forecasts. The active data source is
                **{active_data_mode}** and the selected comparison window is
                **{sample_period}**.

                **Stress score.** For each selected sample, the dashboard
                standardizes VIX, high-yield OAS, and the 10Y-2Y Treasury spread
                into z-scores. The current stress card is the latest
                21-trading-day average of that daily score. Changing the sample
                period changes the z-score baseline because the selected-sample
                means and standard deviations change.
                """
            )
            render_equation(
                "Daily stress score",
                r"s_t = \frac{1}{3}\left(z_{VIX,t} + z_{HY,t} - z_{10Y-2Y,t}\right)",
            )
            st.markdown(
                """
                **Forecast targets.** Rates, yield spreads, and high-yield OAS
                are forecast as daily changes, then converted back into implied
                level paths. Real GDP is forecast as annualized quarterly
                growth, then converted back into an implied GDP level path. VIX
                is volatility context, not a forecast target.
                """
            )
            render_equation("Rate and spread target", r"y_t = x_t - x_{t-1}")
            render_equation(
                "Implied level path",
                r"\hat{x}_{T+h} = x_T + \sum_{j=1}^{h}\hat{y}_{T+j}",
            )
            render_equation(
                "Annualized quarterly GDP growth",
                r"g_t = 100\left[\left(\frac{GDP_t}{GDP_{t-1}}\right)^4 - 1\right]",
            )
            st.markdown("**Baseline models.** All models operate on the transformed target.")
            render_equation("Naive", r"\hat{y}_{T+h} = y_T")
            render_equation(
                "Drift",
                r"y_k = a + bk + e_k,\qquad \hat{y}_{T+h} = a + b(k_T+h)",
            )
            render_equation(
                "AR(1)",
                r"y_t = c + \phi y_{t-1} + e_t",
                "Forecasts are projected recursively from the fitted persistence parameter.",
            )
            st.markdown(
                """
                **Backtest and uncertainty.** Backtest MAE is the average
                absolute one-step-ahead error for the displayed level when an
                implied level is available. Forecast bands are approximate
                uncertainty guides, not confidence guarantees. Gray vertical
                bands denote NBER recessions. GDP has a release lag and
                revisions, so serious real-time forecasting needs vintage-aware
                data.
                """
            )

    sync_query_params(
        data=data_mode,
        forecast=current_forecast_label,
        model=current_model,
        horizon=current_horizon,
        gdp_horizon=current_gdp_horizon,
        sample=sample_period,
        view=active_view,
    )
