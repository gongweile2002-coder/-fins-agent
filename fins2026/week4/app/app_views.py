# ruff: noqa
"""Streamlit layout and controls for the Week 4 portfolio app."""

from __future__ import annotations

import pandas as pd

from fins2026.week4.app.app_config import (
    APP_SUBTITLE,
    APP_TICKER_LABELS,
    APP_TICKER_OPTIONS,
    CONSTRAINT_OPTIONS,
    DEFAULT_CONSTRAINT_MODE,
    DEFAULT_PORTFOLIO_KEY,
    DEFAULT_SAMPLE_PERIOD,
    DEFAULT_VIEW,
    METHOD_NOTES,
    PORTFOLIO_OPTIONS,
    SAMPLE_PERIOD_OPTIONS,
    VIEW_OPTIONS,
)
from fins2026.week4.app.app_data import (
    apply_sample_period,
    load_week4_app_bundle,
    source_status_text,
)
from fins2026.week4.app.app_insights import (
    compact_table_height,
    cumulative_growth_figure,
    drawdown_figure,
    efficient_frontier_figure,
    portfolio_metric_table,
    portfolio_weight_figure,
    top_portfolio_metrics,
)
from fins2026.week4.code.stage4_app import (
    APP_PORTFOLIO_LABELS,
    asset_statistics_for_sample,
    build_app_frontier,
    build_stage4_sample,
    compute_named_portfolio_returns,
    estimate_app_portfolio_weights,
    normalize_custom_weights,
    summarize_named_portfolio_metrics,
)
from fintools.apps import (
    active_tab_label,
    configure_page,
    lazy_tabs,
    query_choice,
    render_csv_download,
    render_data_health,
    render_display_table,
    render_metric_strip,
    sync_query_params,
    tab_is_open,
)


def _selection_key(ticker: str) -> str:
    return f"week4_select_{ticker}"


def _weight_key(ticker: str) -> str:
    return f"week4_weight_{ticker}"
def _initialize_builder_state(st) -> None:
    for ticker in APP_TICKER_OPTIONS:
        st.session_state.setdefault(_selection_key(ticker), True)
        st.session_state.setdefault(_weight_key(ticker), 10.0)

    st.session_state.setdefault(
        "week4_sample_period",
        query_choice("sample", list(SAMPLE_PERIOD_OPTIONS), default=DEFAULT_SAMPLE_PERIOD),
    )
    st.session_state.setdefault(
        "week4_constraint_mode",
        query_choice("constraints", list(CONSTRAINT_OPTIONS), default=DEFAULT_CONSTRAINT_MODE),
    )
    st.session_state.setdefault(
        "week4_active_portfolio",
        query_choice("portfolio", list(PORTFOLIO_OPTIONS), default=DEFAULT_PORTFOLIO_KEY),
    )


def _selected_tickers(st) -> list[str]:
    return [
        ticker
        for ticker in APP_TICKER_OPTIONS
        if st.session_state.get(_selection_key(ticker), False)
    ]


def _custom_raw_weights(st, selected_tickers: list[str]) -> dict[str, float]:
    return {
        ticker: float(st.session_state.get(_weight_key(ticker), 0.0) or 0.0)
        for ticker in selected_tickers
    }


def _builder_summary_table(
    selected_tickers: list[str],
    raw_weights: dict[str, float],
    normalized_weights: pd.DataFrame,
) -> pd.DataFrame:
    mapping = normalized_weights.set_index("ticker")["weight"].mul(100.0).to_dict()
    return pd.DataFrame(
        {
            "Ticker": selected_tickers,
            "Company": [APP_TICKER_LABELS[ticker] for ticker in selected_tickers],
            "Requested weight (%)": [
                float(raw_weights.get(ticker, 0.0))
                for ticker in selected_tickers
            ],
            "Effective weight (%)": [
                float(mapping.get(ticker, 0.0))
                for ticker in selected_tickers
            ],
        }
    )


def _weight_matrix_table(weights: pd.DataFrame) -> pd.DataFrame:
    matrix = weights.pivot(index="ticker", columns="portfolio", values="weight").mul(100.0)
    matrix = matrix.reindex(index=APP_TICKER_OPTIONS)
    matrix = matrix.dropna(how="all").reset_index().rename(columns={"ticker": "Ticker"})
    return matrix


def _render_sidebar_controls(st) -> tuple[str, str, str, list[str], dict[str, float]]:
    with st.sidebar:
        st.header("Controls")
        sample_period = (
            st.segmented_control(
                "Sample window",
                list(SAMPLE_PERIOD_OPTIONS),
                key="week4_sample_period",
            )
            or st.session_state["week4_sample_period"]
        )
        constraint_mode = st.radio(
            "Optimization mode",
            list(CONSTRAINT_OPTIONS),
            format_func=lambda key: CONSTRAINT_OPTIONS[key],
            key="week4_constraint_mode",
        )
        active_portfolio_key = st.radio(
            "Selected portfolio",
            list(PORTFOLIO_OPTIONS),
            format_func=lambda key: PORTFOLIO_OPTIONS[key],
            key="week4_active_portfolio",
        )

        with st.expander("Stock universe", expanded=True):
            st.caption("Choose the names used to build the portfolio opportunity set.")
            for ticker in APP_TICKER_OPTIONS:
                st.checkbox(
                    APP_TICKER_LABELS[ticker],
                    key=_selection_key(ticker),
                )
        selected_tickers = _selected_tickers(st)

        with st.expander("Custom weights (%)", expanded=True):
            st.caption("Enter any non-negative weights. The app renormalizes them to 100%.")
            for ticker in selected_tickers:
                st.number_input(
                    APP_TICKER_LABELS[ticker],
                    min_value=0.0,
                    step=1.0,
                    key=_weight_key(ticker),
                )
        raw_weights = _custom_raw_weights(st, selected_tickers)

    return sample_period, constraint_mode, active_portfolio_key, selected_tickers, raw_weights


def main() -> None:
    """Run the Week 4 client-facing portfolio app."""

    st = configure_page("U.S. Equity Portfolio Builder")
    _initialize_builder_state(st)

    bundle, active_source, warning, loaded_at_utc = load_week4_app_bundle()
    sample_period, constraint_mode, active_portfolio_key, selected_tickers, raw_weights = (
        _render_sidebar_controls(st)
    )
    sampled_bundle = apply_sample_period(bundle, sample_period)
    active_portfolio_label = PORTFOLIO_OPTIONS[active_portfolio_key]

    st.title("U.S. Equity Portfolio Builder")
    st.caption(APP_SUBTITLE)
    if warning:
        st.warning(warning)
    st.caption(
        source_status_text(
            sampled_bundle,
            active_source=active_source,
            loaded_at_utc=loaded_at_utc,
            warning=warning,
        )
    )
    render_data_health(
        sampled_bundle.price_panel,
        source=active_source,
        date_column="date",
        value_columns=["adjClose"],
    )

    if len(selected_tickers) < 2:
        st.info(
            "Select at least two stocks to compare optimized portfolios and draw the "
            "efficient frontier."
        )
        custom_weights, custom_warning = normalize_custom_weights(
            selected_tickers or APP_TICKER_OPTIONS[:1],
            raw_weights,
        )
        if custom_warning:
            st.warning(custom_warning)
        st.subheader("Current custom allocation")
        builder_table = _builder_summary_table(
            selected_tickers or APP_TICKER_OPTIONS[:1],
            raw_weights,
            custom_weights,
        )
        render_display_table(
            builder_table,
            reset_index=False,
            height=compact_table_height(builder_table),
        )
        return

    sample = build_stage4_sample(sampled_bundle.feature_panel, selected_tickers=selected_tickers)
    custom_weights, custom_warning = normalize_custom_weights(selected_tickers, raw_weights)
    weights, solve_methods = estimate_app_portfolio_weights(
        sample,
        custom_weights=custom_weights,
        constraint_mode=constraint_mode,
    )
    portfolio_returns = compute_named_portfolio_returns(sample, weights)
    metrics = summarize_named_portfolio_metrics(portfolio_returns)
    frontier = build_app_frontier(sample, weights, constraint_mode=constraint_mode)
    asset_stats = asset_statistics_for_sample(sample)

    render_metric_strip(
        top_portfolio_metrics(
            metrics,
            active_portfolio_label=active_portfolio_label,
            sample=sample,
            constraint_mode=constraint_mode,
        ),
        columns=3,
    )

    view_default = query_choice("view", VIEW_OPTIONS, default=DEFAULT_VIEW)
    tabs = lazy_tabs(VIEW_OPTIONS, default=view_default, key="week4_app_view")
    active_view = active_tab_label(VIEW_OPTIONS, tabs, default=view_default)
    (
        tab_overview,
        tab_builder,
        tab_optimized,
        tab_history,
        tab_frontier,
        tab_data,
        tab_method,
    ) = tabs

    if tab_is_open(tab_overview, fallback=active_view == "Overview"):
        with tab_overview:
            st.subheader("Selected opportunity set")
            st.markdown(
                f"Using **{len(selected_tickers)} stocks** over the balanced in-sample window "
                f"from **{sample.start_date:%Y-%m-%d}** to **{sample.end_date:%Y-%m-%d}**."
            )
            if custom_warning:
                st.warning(custom_warning)
            st.plotly_chart(
                portfolio_weight_figure(weights, portfolio_label=active_portfolio_label),
                width="stretch",
            )
            st.subheader("Portfolio comparison")
            metric_table = portfolio_metric_table(metrics)
            render_display_table(
                metric_table,
                reset_index=False,
                height=compact_table_height(metric_table),
            )

    if tab_is_open(tab_builder, fallback=active_view == "Portfolio Builder"):
        with tab_builder:
            st.subheader("Custom allocation")
            st.markdown(
                "The custom portfolio uses the selected stocks and renormalizes the entered "
                "weights to 100%."
            )
            if custom_warning:
                st.warning(custom_warning)
            builder_table = _builder_summary_table(selected_tickers, raw_weights, custom_weights)
            render_display_table(
                builder_table,
                reset_index=False,
                height=compact_table_height(builder_table),
            )
            st.plotly_chart(
                portfolio_weight_figure(weights, portfolio_label=APP_PORTFOLIO_LABELS["custom"]),
                width="stretch",
            )

    if tab_is_open(tab_optimized, fallback=active_view == "Optimized Portfolios"):
        with tab_optimized:
            st.subheader("Portfolio weights")
            weight_focus = st.radio(
                "Inspect weights for",
                list(PORTFOLIO_OPTIONS.values()),
                index=list(PORTFOLIO_OPTIONS.values()).index(active_portfolio_label),
                horizontal=True,
                key="week4_weight_focus",
            )
            st.plotly_chart(
                portfolio_weight_figure(weights, portfolio_label=weight_focus),
                width="stretch",
            )
            st.caption(
                "Solve method: "
                + ", ".join(f"{label} = {method}" for label, method in solve_methods.items())
            )
            weight_table = _weight_matrix_table(weights)
            render_display_table(
                weight_table,
                reset_index=False,
                height=compact_table_height(weight_table, max_height=640),
            )

    if tab_is_open(tab_history, fallback=active_view == "Historical Performance"):
        with tab_history:
            st.subheader("Historical in-sample performance")
            st.plotly_chart(
                cumulative_growth_figure(portfolio_returns, sample=sample),
                width="stretch",
            )
            st.plotly_chart(
                drawdown_figure(portfolio_returns, sample=sample),
                width="stretch",
            )
            metric_table = portfolio_metric_table(metrics)
            render_display_table(
                metric_table,
                reset_index=False,
                height=compact_table_height(metric_table),
            )
            download_cols = st.columns(2)
            with download_cols[0]:
                render_csv_download(
                    portfolio_returns,
                    label="Download portfolio return panel",
                    file_name="week4_portfolio_returns.csv",
                    key="download_week4_portfolio_returns",
                )
            with download_cols[1]:
                render_csv_download(
                    metrics,
                    label="Download portfolio metrics",
                    file_name="week4_portfolio_metrics.csv",
                    key="download_week4_portfolio_metrics",
                )

    if tab_is_open(tab_frontier, fallback=active_view == "Efficient Frontier"):
        with tab_frontier:
            st.subheader("Efficient frontier")
            st.markdown(
                "The curve and points below are estimated in-sample only, "
                "using the selected historical window."
            )
            st.plotly_chart(
                efficient_frontier_figure(
                    frontier,
                    asset_stats,
                    metrics,
                    sample=sample,
                    active_portfolio_label=active_portfolio_label,
                ),
                width="stretch",
            )

    if tab_is_open(tab_data, fallback=active_view == "Data"):
        with tab_data:
            st.subheader("Data and downloads")
            table_choice = st.segmented_control(
                "Table",
                [
                    "Price panel",
                    "Feature panel",
                    "Balanced returns",
                    "Portfolio returns",
                    "Weights",
                ],
                key="week4_data_choice",
            )
            table_choice = table_choice or "Price panel"
            if table_choice == "Price panel":
                frame = sampled_bundle.price_panel.loc[
                    sampled_bundle.price_panel["ticker"].isin(selected_tickers)
                ].copy()
            elif table_choice == "Feature panel":
                frame = sampled_bundle.feature_panel.loc[
                    sampled_bundle.feature_panel["ticker"].isin(selected_tickers)
                ].copy()
            elif table_choice == "Balanced returns":
                frame = sample.returns_wide.reset_index()
            elif table_choice == "Portfolio returns":
                frame = portfolio_returns.copy()
            else:
                frame = _weight_matrix_table(weights)
            render_display_table(frame.tail(300), reset_index=False, height=520)
            render_csv_download(
                frame,
                label="Download displayed table",
                file_name="week4_displayed_table.csv",
                key=f"download_week4_{table_choice.lower().replace(' ', '_')}",
            )

    if tab_is_open(tab_method, fallback=active_view == "Methodology"):
        with tab_method:
            st.subheader("Methodology")
            st.markdown(
                "- Returns use adjusted Yahoo prices and the Kenneth French daily risk-free rate.\n"
                "- The app compares a custom portfolio with equal-weight, minimum-variance, "
                "and mean-variance allocations.\n"
                "- Results are **in-sample only** for the selected historical window.\n"
                f"- {METHOD_NOTES[constraint_mode]}"
            )
            st.markdown("Minimum-variance portfolio:")
            st.latex(
                r"w_{mv} = \arg\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad "
                r"\mathbf{1}^\top w = 1"
            )
            st.markdown("Mean-variance portfolio:")
            if constraint_mode == "unconstrained":
                st.latex(
                    r"w_{tan} = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}"
                    r"{\mathbf{1}^\top \Sigma^{-1}(\mu - r_f \mathbf{1})}"
                )
            else:
                st.latex(
                    r"w_{tan} = \arg\max_w \; "
                    r"\frac{w^\top(\mu-r_f\mathbf{1})}{\sqrt{w^\top \Sigma w}} "
                    r"\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; 0 \le w_i \le 1"
                )
            st.info(
                "Out-of-sample re-estimation is the next step in the course workflow, but it is "
                "not coded in this Week 4 app."
            )

    sync_query_params(
        view=active_view,
        sample=sample_period,
        constraints=constraint_mode,
        portfolio=active_portfolio_key,
    )
