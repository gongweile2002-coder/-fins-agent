# ruff: noqa
"""Streamlit view rendering for the Australia macro forecast app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fins2026.week3.app.app_config import (
    ALL_SPECS,
    APP_TITLE,
    AUSTRALIA_STATE_OPTIONS,
    BACKTEST_LABELS,
    DATA_MODE_OPTIONS,
    DEFAULT_FORECAST_SERIES,
    DEFAULT_MODEL,
    DEFAULT_MONTHLY_HORIZON,
    DEFAULT_QUARTERLY_HORIZON,
    DEFAULT_SAMPLE_PERIOD,
    FORECAST_LABELS,
    FORECAST_OUTPUT_LABELS,
    LEADERBOARD_LABELS,
    MODEL_LABELS,
    MODEL_SCOPE_NOTES,
    ONE_STEP_ONLY_MODELS,
    SAMPLE_PERIOD_OPTIONS,
    US_CONTEXT_LABEL_MAP,
    US_CONTEXT_OPTIONS,
    VIEW_OPTIONS,
)
from fins2026.week3.app.app_data import (
    apply_sample_period,
    load_benchmark_leaderboard,
    load_model_outputs,
    load_week3_data,
    source_status_text,
)
from fins2026.week3.app.app_insights import (
    australia_overview_text,
    compact_table_height,
    comparison_table,
    forecast_level_change_percent,
    forecast_summary_text,
    format_observation_date,
    format_percent,
    latest_observation,
    latest_snapshot,
    level_or_target_backtest_frame,
    line_figure,
    series_window_label,
    target_short_label,
    top_state_metrics,
    us_context_snapshot,
)
from fintools.apps import (
    MetricCard,
    active_tab_label,
    backtest_figure,
    configure_page,
    latest_delta,
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
    quarterly_horizon_default: int,
    include_horizon: bool = True,
) -> tuple[str, str, int]:
    """Render forecast controls and return series, model, and horizon."""

    cols = st.columns([1.25, 1.0, 1.0] if include_horizon else [1.5, 1.0])
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
    spec = FORECAST_LABELS[forecast_label]
    if not include_horizon:
        return forecast_label, model, 1
    with cols[2]:
        if model in ONE_STEP_ONLY_MODELS:
            st.caption("Horizon")
            st.write("**1 step only**")
            horizon = 1
        elif spec.frequency == "quarterly":
            horizon = st.slider(
                "Forecast horizon (quarters)",
                1,
                8,
                quarterly_horizon_default,
                step=1,
                key=f"{prefix}_forecast_horizon_quarters",
            )
        else:
            horizon = st.slider(
                "Forecast horizon (months)",
                1,
                12,
                horizon_default,
                step=1,
                key=f"{prefix}_forecast_horizon_months",
            )
    return forecast_label, model, horizon


def _selected_australia_series(
    selection: str,
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
) -> tuple[pd.Series, str]:
    if selection in australia_monthly and australia_monthly[selection].dropna().any():
        units = ALL_SPECS[selection].units if selection in ALL_SPECS else ""
        return australia_monthly[selection].dropna(), units
    if selection in australia_quarterly:
        units = ALL_SPECS[selection].units if selection in ALL_SPECS else ""
        return australia_quarterly[selection].dropna(), units
    raise KeyError(selection)


def main() -> None:
    """Render the Australia macro forecast app."""

    configure_page(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(
        "Forecast core Australian macro targets from release-lag-aware data, "
        "then benchmark them against transparent univariate and exogenous models."
    )

    data_mode_default = query_choice("data", DATA_MODE_OPTIONS, default="Fixture")
    forecast_default = query_choice(
        "forecast", list(FORECAST_LABELS), default=DEFAULT_FORECAST_SERIES
    )
    model_default = query_choice("model", list(MODEL_LABELS), default=DEFAULT_MODEL)
    horizon_default = query_int(
        "horizon", default=DEFAULT_MONTHLY_HORIZON, minimum=1, maximum=12, step=1
    )
    quarterly_horizon_default = query_int(
        "quarter_horizon",
        default=DEFAULT_QUARTERLY_HORIZON,
        minimum=1,
        maximum=8,
        step=1,
    )
    sample_default = query_choice(
        "sample", list(SAMPLE_PERIOD_OPTIONS), default=DEFAULT_SAMPLE_PERIOD
    )
    view_default = query_choice("view", VIEW_OPTIONS, default="Overview")
    australia_default = query_choice(
        "au_series", AUSTRALIA_STATE_OPTIONS, default=DEFAULT_FORECAST_SERIES
    )
    us_default = query_choice("us_series", US_CONTEXT_OPTIONS, default="FEDFUNDS")

    with st.sidebar:
        st.header("Controls")
        data_mode = st.radio(
            "Data source",
            DATA_MODE_OPTIONS,
            index=DATA_MODE_OPTIONS.index(data_mode_default),
            horizontal=True,
            captions=[
                "Frozen Australia and U.S. panels for repeatable benchmark results.",
                "Attempts a fresh Australia RBA and U.S. FRED refresh before "
                "falling back if needed.",
            ],
            key="data_mode",
        )

    bundle, active_data_mode, warning, loaded_at_utc = load_week3_data(data_mode)
    if warning:
        st.warning("Live data could not be rebuilt, so the fixture bundle is shown.")
        with st.expander("Technical detail"):
            st.write(warning)

    sample_period = st.segmented_control(
        "Sample period",
        options=list(SAMPLE_PERIOD_OPTIONS),
        default=sample_default,
        help=(
            "The selected period controls the app metrics, charts, benchmark "
            "comparisons, and downloads."
        ),
        key="sample_period",
    )
    sample_period = sample_period or DEFAULT_SAMPLE_PERIOD
    australia_monthly = apply_sample_period(bundle["australia_monthly"], sample_period)
    australia_quarterly = apply_sample_period(bundle["australia_quarterly"], sample_period)
    us_monthly = apply_sample_period(bundle["us_monthly"], sample_period)
    us_quarterly = apply_sample_period(bundle["us_quarterly"], sample_period)

    render_compact_metric_strip(
        top_state_metrics(
            australia_monthly,
            australia_quarterly,
            active_data_mode=active_data_mode,
            sample_period=sample_period,
        ),
        columns=3,
    )
    render_data_health(
        australia_monthly,
        source=active_data_mode,
        value_columns=list(ALL_SPECS),
    )
    st.caption(
        source_status_text(
            bundle,
            active_data_mode=active_data_mode,
            loaded_at_utc=loaded_at_utc,
            warning=warning,
        )
    )

    current_forecast_label = forecast_default
    current_model = model_default
    current_horizon = horizon_default
    current_quarter_horizon = quarterly_horizon_default
    current_au_series = australia_default
    current_us_series = us_default

    tabs = lazy_tabs(VIEW_OPTIONS, default=view_default, key="main_view")
    active_view = active_tab_label(VIEW_OPTIONS, tabs, default=view_default)
    (
        tab_overview,
        tab_state,
        tab_forecast,
        tab_compare,
        tab_backtest,
        tab_us,
        tab_data,
        tab_method,
    ) = tabs

    if tab_is_open(tab_overview, fallback=active_view == "Overview"):
        with tab_overview:
            st.subheader("Overview")
            st.markdown(
                """
                Australian macro data is the primary product surface.
                The app uses an observable month-end information set and benchmarks
                transparent forecast methods against one-step backtests.
                """
            )
            st.markdown(
                australia_overview_text(
                    australia_monthly,
                    australia_quarterly,
                    sample_period,
                )
            )
            st.caption(
                "Open Model Comparison to run or inspect the one-step benchmark leaderboard "
                "for the selected sample."
            )
            snapshot = latest_snapshot(australia_monthly, australia_quarterly)
            render_display_table(
                snapshot.head(8),
                labels={
                    "indicator": "Indicator",
                    "latest_date": "Latest date",
                    "latest_value": "Latest value",
                    "units": "Units",
                    "forecast_treatment": "Forecast treatment",
                },
                reset_index=False,
                height=compact_table_height(snapshot.head(8)),
                column_alignments="center",
            )

    if tab_is_open(tab_state, fallback=active_view == "Australia Snapshot"):
        with tab_state:
            st.subheader("Australia Snapshot")
            current_au_series = st.selectbox(
                "Australia series",
                AUSTRALIA_STATE_OPTIONS,
                index=AUSTRALIA_STATE_OPTIONS.index(australia_default),
                key="australia_state_series",
            )
            series, units = _selected_australia_series(
                current_au_series,
                australia_monthly,
                australia_quarterly,
            )
            value = float(series.iloc[-1])
            delta_periods = (
                4
                if current_au_series
                in {"Headline CPI inflation", "Wage Price Index growth", "Real GDP"}
                else 12
            )
            delta = latest_delta(series, periods=min(delta_periods, max(len(series) - 1, 1)))
            render_compact_metric_strip(
                [
                    MetricCard(
                        "Latest",
                        f"{value:,.2f}" if abs(value) < 10_000 else f"{value:,.0f}",
                        help=f"Latest observation: {format_observation_date(series.index[-1])}.",
                    ),
                    MetricCard(
                        "Window",
                        series_window_label(series),
                        help="Date span behind the displayed Australia series.",
                    ),
                    MetricCard(
                        "Change from earlier sample point",
                        format_percent(delta, signed=True)
                        if units == "Percent"
                        else (f"{delta:+,.2f}" if delta is not None else "n/a"),
                        help=(
                            "Computed over a longer comparison step so short "
                            "samples still show a meaningful move."
                        ),
                    ),
                ],
                columns=3,
            )
            st.plotly_chart(
                line_figure(series, indicator_name=current_au_series, units=units or "Level"),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
            if current_au_series in ALL_SPECS and ALL_SPECS[current_au_series].caveat:
                st.caption(ALL_SPECS[current_au_series].caveat)
            snapshot = latest_snapshot(australia_monthly, australia_quarterly)
            st.subheader("Latest Australia readings")
            render_display_table(
                snapshot,
                labels={
                    "indicator": "Indicator",
                    "latest_date": "Latest date",
                    "latest_value": "Latest value",
                    "units": "Units",
                    "forecast_treatment": "Forecast treatment",
                },
                reset_index=False,
                height=compact_table_height(snapshot),
                column_alignments="center",
            )

    if tab_is_open(tab_forecast, fallback=active_view == "Forecasts"):
        with tab_forecast:
            st.subheader("Forecasts")
            st.caption(
                "Univariate models support multi-step forward paths. Exogenous models are "
                "intentionally limited to one-step forecasts so the app never "
                "invents future driver paths."
            )
            forecast_label, model, horizon = render_forecast_controls(
                prefix="forecast_lab",
                forecast_default=forecast_default,
                model_default=model_default,
                horizon_default=horizon_default,
                quarterly_horizon_default=quarterly_horizon_default,
            )
            current_forecast_label = forecast_label
            current_model = model
            spec = FORECAST_LABELS[forecast_label]
            if spec.frequency == "quarterly":
                current_quarter_horizon = horizon
            else:
                current_horizon = horizon
            forecast_result, backtest = load_model_outputs(
                active_data_mode,
                sample_period,
                forecast_label,
                model,
                horizon,
            )
            latest_level = float(forecast_result.observed_level.iloc[-1])
            render_compact_metric_strip(
                [
                    MetricCard(
                        "Latest level",
                        f"{latest_level:,.2f}"
                        if abs(latest_level) < 10_000
                        else f"{latest_level:,.0f}",
                        help=(
                            "Latest observation: "
                            f"{format_observation_date(forecast_result.observed_level.index[-1])}."
                        ),
                    ),
                    MetricCard(
                        "Target",
                        target_short_label(forecast_label),
                        help=target_name(spec),
                    ),
                    MetricCard(
                        "Model scope",
                        "1-step only" if model in ONE_STEP_ONLY_MODELS else f"{horizon} steps",
                        help=MODEL_SCOPE_NOTES[model],
                    ),
                    MetricCard(
                        "1-step level change",
                        format_percent(
                            forecast_level_change_percent(forecast_result, latest_level),
                            signed=True,
                        ),
                        help=(
                            "Implied level change between the latest observation "
                            "and the forecast horizon endpoint."
                        ),
                    ),
                ],
                columns=4,
            )
            st.markdown(
                forecast_summary_text(
                    forecast_label,
                    forecast_result,
                    model_label=MODEL_LABELS[model],
                    horizon=horizon,
                )
            )
            st.plotly_chart(
                target_forecast_figure(
                    forecast_result,
                    indicator_name=spec.label,
                    shade_recessions=False,
                    range_slider=True,
                ),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
            if model in ONE_STEP_ONLY_MODELS:
                st.caption("This model is benchmarked and displayed as a one-step forecast only.")
            download_cols = st.columns(2)
            with download_cols[0]:
                render_csv_download(
                    prepare_display_frame(
                        forecast_result.display_forecast, labels=FORECAST_OUTPUT_LABELS
                    ),
                    label="Download displayed forecast CSV",
                    file_name=(f"{forecast_label.lower().replace(' ', '_')}_display_forecast.csv"),
                    key="download_display_forecast",
                )
            with download_cols[1]:
                render_csv_download(
                    prepare_display_frame(
                        forecast_result.target_forecast, labels=FORECAST_OUTPUT_LABELS
                    ),
                    label="Download target forecast CSV",
                    file_name=(f"{forecast_label.lower().replace(' ', '_')}_target_forecast.csv"),
                    key="download_target_forecast",
                )

    if tab_is_open(tab_compare, fallback=active_view == "Model Comparison"):
        with tab_compare:
            st.subheader("Model Comparison")
            series_label = st.selectbox(
                "Benchmark series",
                list(FORECAST_LABELS),
                index=list(FORECAST_LABELS).index(forecast_default),
                key="model_comparison_series",
            )
            current_forecast_label = series_label
            leaderboard = load_benchmark_leaderboard(active_data_mode, sample_period)
            table = comparison_table(leaderboard, series_label)
            if table.empty:
                st.info("No model comparison rows are available for the selected sample.")
            else:
                best_row = table.iloc[0]
                render_compact_metric_strip(
                    [
                        MetricCard("Best model", str(best_row["model_label"])),
                        MetricCard("Status", str(best_row["status"])),
                        MetricCard(
                            "Ranking metric",
                            f"{best_row['ranking_metric']:.3f}",
                            help=(
                                "Level MAE when an implied level path exists, otherwise target MAE."
                            ),
                        ),
                    ],
                    columns=3,
                )
                render_display_table(
                    table,
                    labels=LEADERBOARD_LABELS,
                    reset_index=False,
                    height=compact_table_height(table),
                    column_alignments="center",
                )

    if tab_is_open(tab_backtest, fallback=active_view == "Backtests"):
        with tab_backtest:
            st.subheader("Backtests")
            st.caption(
                "Model comparison uses one-step rolling backtests. "
                "Exogenous models stay on the same one-step footing as the "
                "saved benchmark leaderboard."
            )
            forecast_label, model, _ = render_forecast_controls(
                prefix="backtest_lab",
                forecast_default=forecast_default,
                model_default=model_default,
                horizon_default=1,
                quarterly_horizon_default=1,
                include_horizon=False,
            )
            current_forecast_label = forecast_label
            current_model = model
            spec = FORECAST_LABELS[forecast_label]
            _, backtest = load_model_outputs(
                active_data_mode,
                sample_period,
                forecast_label,
                model,
                1,
            )
            if backtest.empty:
                st.info("Not enough history is available for the selected backtest.")
            else:
                plot_backtest = level_or_target_backtest_frame(backtest)
                st.plotly_chart(
                    backtest_figure(plot_backtest, units=spec.units, shade_recessions=False),
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": False},
                )
                st.caption(
                    f"{MODEL_LABELS[model]} benchmark: "
                    f"{series_window_label(plot_backtest['actual'])}. "
                    "The plotted line is level-space when the target can be "
                    "reconstructed, otherwise the transformed target itself."
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
                    file_name=f"{forecast_label.lower().replace(' ', '_')}_{model}_backtest.csv",
                    key="download_backtest",
                )

    if tab_is_open(tab_us, fallback=active_view == "U.S. Context"):
        with tab_us:
            st.subheader("U.S. Context")
            current_us_series = st.selectbox(
                "U.S. context series",
                US_CONTEXT_OPTIONS,
                index=US_CONTEXT_OPTIONS.index(us_default),
                format_func=lambda item: US_CONTEXT_LABEL_MAP[item],
                key="us_context_series",
            )
            label = US_CONTEXT_LABEL_MAP[current_us_series]
            series = us_monthly[current_us_series].dropna()
            latest_value, latest_date = latest_observation(us_monthly, current_us_series)
            render_compact_metric_strip(
                [
                    MetricCard(
                        "Latest",
                        f"{latest_value:,.2f}" if latest_value is not None else "n/a",
                        help=f"Latest observation: {format_observation_date(latest_date)}.",
                    ),
                    MetricCard(
                        "Sample",
                        series_window_label(series),
                        help=("Visible date span for the selected U.S. context series."),
                    ),
                    MetricCard(
                        "Role",
                        "Secondary context",
                        help=(
                            "Used as comparison context and, for selected "
                            "models, as lagged exogenous input."
                        ),
                    ),
                ],
                columns=3,
            )
            st.plotly_chart(
                line_figure(
                    series,
                    indicator_name=label,
                    units=label.split("(")[-1].rstrip(")") if "(" in label else "Level",
                    shade_recessions=True,
                ),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": False},
            )
            snapshot = us_context_snapshot(us_monthly)
            render_display_table(
                snapshot,
                labels={
                    "indicator": "Indicator",
                    "latest_date": "Latest date",
                    "latest_value": "Latest value",
                },
                reset_index=False,
                height=compact_table_height(snapshot),
                column_alignments="center",
            )

    if tab_is_open(tab_data, fallback=active_view == "Data"):
        with tab_data:
            st.subheader("Data")
            st.caption(
                "The same sample window drives the Australia forecast panels, "
                "the U.S. context panels, and the saved benchmark comparisons."
            )
            table_choice = st.segmented_control(
                "Dataset",
                options=[
                    "Australia monthly",
                    "Australia quarterly",
                    "U.S. monthly",
                    "U.S. quarterly",
                ],
                default="Australia monthly",
                key="data_table_choice",
            )
            table_choice = table_choice or "Australia monthly"
            if table_choice == "Australia monthly":
                frame = australia_monthly
                file_name = "australia_monthly_sample.csv"
            elif table_choice == "Australia quarterly":
                frame = australia_quarterly
                file_name = "australia_quarterly_sample.csv"
            elif table_choice == "U.S. monthly":
                frame = us_monthly
                file_name = "us_monthly_context_sample.csv"
            else:
                frame = us_quarterly
                file_name = "us_quarterly_context_sample.csv"
            render_display_table(frame.tail(240), height=520, column_alignments="center")
            render_csv_download(
                prepare_display_frame(frame),
                label="Download selected data CSV",
                file_name=file_name,
                key="download_data_table",
            )

    if tab_is_open(tab_method, fallback=active_view == "Methodology"):
        with tab_method:
            st.subheader("Methodology")
            st.markdown(
                f"""
                This product is built around the **observable** Australia information set,
                not a hindsight reference panel. Monthly targets use the latest observable
                month-end panel. Quarterly targets use release-month observations from a
                committed release-timing dataset. U.S. data enters as secondary context and,
                for the exogenous models, as lagged predictors only.

                The active source is **{active_data_mode}** and the selected sample is
                **{sample_period}**.
                """
            )
            render_equation("Level-change target", r"y_t = x_t - x_{t-1}")
            render_equation(
                "Log-change target",
                r"y_t = 100\log\left(\frac{x_t}{x_{t-1}}\right)",
            )
            render_equation(
                "Annualized GDP growth target",
                r"g_t = 100\left[\left(\frac{GDP_t}{GDP_{t-1}}\right)^4 - 1\right]",
            )
            render_equation(
                "Implied level path from a change target",
                r"\hat{x}_{T+h} = x_T + \sum_{j=1}^{h}\hat{y}_{T+j}",
            )
            render_equation(
                "Implied level path from a log-change target",
                r"\hat{x}_{T+h} = x_T \exp\left(\sum_{j=1}^{h}\hat{y}_{T+j}/100\right)",
            )
            st.markdown("**Model families.**")
            render_equation("AR", r"y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + e_t")
            render_equation(
                "ARMA",
                (
                    r"y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + "
                    r"\sum_{j=1}^{q}\theta_j e_{t-j} + e_t"
                ),
            )
            render_equation(
                "ARMA + exogenous",
                (
                    r"y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \beta' z_t + "
                    r"\sum_{j=1}^{q}\theta_j e_{t-j} + e_t"
                ),
                (
                    "The exogenous block uses lag-safe Australia and U.S. "
                    "features, and this family is limited to one-step forecasts."
                ),
            )
            render_equation(
                "Elastic-net dynamic regression",
                (
                    r"y_t = c + \gamma' \ell_t + \beta' z_t + e_t,\qquad "
                    r"\min_{\beta,\gamma}\|e\|_2^2 + "
                    r"\lambda\left[\alpha\|\theta\|_1 + (1-\alpha)\|\theta\|_2^2\right]"
                ),
                (
                    "The OLS + elastic net design uses target lags plus lag-safe "
                    "Australia and U.S. features, and it stays on a "
                    "one-step footing."
                ),
            )
            st.markdown(
                """
                **Benchmark rule.** Saved leaderboards compare one-step rolling backtests.
                The ranking metric is level MAE when a transformed target can be rebuilt
                into an implied level path, otherwise target MAE. Forward multi-step paths
                are shown only for the univariate models so the app does not invent future
                exogenous scenarios.
                """
            )

    sync_query_params(
        data=data_mode,
        forecast=current_forecast_label,
        model=current_model,
        horizon=current_horizon,
        quarter_horizon=current_quarter_horizon,
        sample=sample_period,
        view=active_view,
        au_series=current_au_series,
        us_series=current_us_series,
    )
